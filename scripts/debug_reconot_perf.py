#!/usr/bin/env python3
"""
ReconOT 性能退化诊断工具。

用法示例::

  # 1) 解析 eval 日志（master.log / stdout / reconot.log）
  python scripts/debug_reconot_perf.py analyze-log \\
      --log ../autoresearch-rl-covenant/logs/eval-reconot-20260608-080647.master.log

  # 自动搜索常见路径
  python scripts/debug_reconot_perf.py analyze-log --auto-discover

  # 2) 对指定 mesh 做分阶段 profile（需 VQVAE，与 eval 同环境）
  python scripts/debug_reconot_perf.py profile \\
      --mesh-id 3deabb2bb43136108c1ef1705eceb4a7511b21070a2e1512cb56a97b51cc9552 \\
      --kr 0.75 --repeat 3

  # 3) 从日志中取最慢的 N 个 mesh 批量 profile
  python scripts/debug_reconot_perf.py profile-slowest --log ../output/logs/reconot.log --top 10

  # 4) 检测长跑漂移：同一 mesh 连续跑 N 次看是否变慢（内存/日志膨胀）
  python scripts/debug_reconot_perf.py drift-test --mesh-id <id> --repeat 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KEEP_RATIOS = (0.75, 0.5, 0.25)

# 行内任意位置匹配（兼容 master.log 前缀、tee stdout、reconot.log）
_RE_START = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}).*?"
    r"prune_start\s+kr=(?P<kr>[\d.]+)\s+tag=(?P<tag>\S+)"
)
_RE_DONE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}).*?"
    r"prune_done\s+kr=(?P<kr>[\d.]+)\s+tag=(?P<tag>\S+).*?"
    r"unique_kept=(?P<unique>\d+)"
)
_RE_CAP = re.compile(
    r"cap_needed=(?P<cap_needed>\d+)\s+cap_eff_final=(?P<cap_eff>\d+)\s+"
    r"per_id_cap=(?P<per_id_cap>\d+)"
)
_RE_PAIR_COS = re.compile(r"pair_cos_kept=(\S+)")
_RE_EVAL_PRUNE_DONE = re.compile(
    r"\[eval\].*prune reconot kr=(?P<kr>[\d.]+) done \((?P<sec>[\d.]+)s\)"
)
_RE_EVAL_PRUNE_START = re.compile(r"\[eval\].*prune reconot kr=(?P<kr>[\d.]+) \.\.\.")
_RE_MESH_TAG = re.compile(r"\[\d+/\d+\]\s+(?P<tag>[0-9a-f]{64})\b")
_RE_GLB_MESH = re.compile(r"/([0-9a-f]{64})\.glb")


def _parse_ts(ts: str, ms: str) -> float:
    from datetime import datetime

    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() + int(ms) / 1000.0


@dataclass
class PruneRecord:
    tag: str
    kr: float
    t_start: float
    t_end: float = 0.0
    duration_s: float = 0.0
    unique_kept: int = 0
    cap_needed: int = 0
    cap_eff_final: int = 0
    per_id_cap: int = 1
    fast_diagnostics: bool = True
    eval_wall_s: Optional[float] = None
    source_path: str = ""
    source_kind: str = "logger"  # logger | eval_wall
    run_id: str = ""
    log_mtime: float = 0.0

    @property
    def mesh_index_hint(self) -> Optional[int]:
        m = re.search(r"\[(\d+)/\d+\]", self.tag)
        return int(m.group(1)) if m else None


@dataclass
class MeshAggregate:
    tag: str
    total_prune_s: float = 0.0
    kr_records: Dict[float, PruneRecord] = field(default_factory=dict)


def _normalize_tag(tag: str, current_mesh: str) -> str:
    t = (tag or "").strip()
    if not t or t == "selector=dpp" or t == "selector=ot":
        return current_mesh
    return t


def _parse_run_id(log_path: Path) -> str:
    for part in log_path.parts:
        m = re.fullmatch(r"run-(\d+)", part)
        if m:
            return f"run-{m.group(1)}"
    return ""


def _record_priority(rec: PruneRecord) -> Tuple[int, float, float]:
    """Higher wins on dedupe: logger timestamps > eval wall; newer log/run."""
    kind_score = 2 if rec.source_kind == "logger" else 1
    run_score = int(rec.run_id.split("-")[-1]) if rec.run_id else 0
    return (kind_score, run_score, rec.log_mtime)


def discover_log_paths(repo_root: Path, *, run_filter: str = "") -> List[Path]:
    """常见 eval / autoresearch 日志路径（按修改时间倒序）。"""
    ar = repo_root.parent / "autoresearch-rl-covenant"
    output_root = repo_root.parent / "output"
    roots = [repo_root, output_root, ar] if ar.is_dir() else [repo_root, output_root]
    globs = [
        "**/eval_run.log",
        "**/logs/reconot.log",
        "logs/reconot.log",
        "logs/*.master.log",
        "eval_results*/**/eval_run.log",
        "eval_results*/**/logs/reconot.log",
        "artifacts/eval-reconot/runs/**/eval_run.log",
        "artifacts/eval-reconot/runs/**/logs/reconot.log",
    ]
    if ar.is_dir():
        globs.append("logs/*.master.log")

    seen: set[str] = set()
    out: List[Path] = []
    for root in roots:
        for pat in globs:
            for p in root.glob(pat):
                if not p.is_file() or p.stat().st_size == 0:
                    continue
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    if run_filter:
        rf = run_filter if run_filter.startswith("run-") else f"run-{run_filter}"
        out = [p for p in out if rf in p.parts]
    return out


def iter_log_prune_records(log_path: Path) -> Iterator[PruneRecord]:
    """解析 reconot.log / master.log / tee stdout（多格式）。"""
    pending: Dict[Tuple[str, float], PruneRecord] = {}
    current_mesh = ""
    eval_only: Dict[Tuple[str, float], PruneRecord] = {}
    n_logger = 0
    n_eval = 0
    src = str(log_path.resolve())
    run_id = _parse_run_id(log_path)
    log_mtime = log_path.stat().st_mtime

    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m_mesh = _RE_MESH_TAG.search(line)
            if m_mesh:
                current_mesh = m_mesh.group("tag")
            else:
                m_glb = _RE_GLB_MESH.search(line)
                if m_glb:
                    current_mesh = m_glb.group(1)

            m_es = _RE_EVAL_PRUNE_START.search(line)
            if m_es:
                kr = float(m_es.group("kr"))
                tag = current_mesh or f"unknown_{len(eval_only)}"
                eval_only[(tag, kr)] = PruneRecord(tag=tag, kr=kr, t_start=0.0)
                continue

            m_ed = _RE_EVAL_PRUNE_DONE.search(line)
            if m_ed:
                kr = float(m_ed.group("kr"))
                sec = float(m_ed.group("sec"))
                tag = current_mesh or f"unknown_{n_eval}"
                rec = eval_only.pop((tag, kr), None)
                if rec is None:
                    rec = PruneRecord(tag=tag, kr=kr, t_start=0.0)
                rec.duration_s = sec
                rec.eval_wall_s = sec
                rec.source_path = src
                rec.source_kind = "eval_wall"
                rec.run_id = run_id
                rec.log_mtime = log_mtime
                n_eval += 1
                yield rec
                continue

            m_s = _RE_START.search(line)
            if m_s:
                tag = _normalize_tag(m_s.group("tag"), current_mesh)
                key = (tag, float(m_s.group("kr")))
                pending[key] = PruneRecord(
                    tag=tag,
                    kr=float(m_s.group("kr")),
                    t_start=_parse_ts(m_s.group("ts"), m_s.group("ms")),
                )
                continue

            m_d = _RE_DONE.search(line)
            if m_d:
                tag = _normalize_tag(m_d.group("tag"), current_mesh)
                key = (tag, float(m_d.group("kr")))
                rec = pending.pop(key, None)
                if rec is None:
                    rec = PruneRecord(tag=tag, kr=float(m_d.group("kr")), t_start=0.0)
                rec.t_end = _parse_ts(m_d.group("ts"), m_d.group("ms"))
                if rec.t_start > 0:
                    rec.duration_s = rec.t_end - rec.t_start
                rec.unique_kept = int(m_d.group("unique"))
                cap_m = _RE_CAP.search(line)
                if cap_m:
                    rec.cap_needed = int(cap_m.group("cap_needed"))
                    rec.cap_eff_final = int(cap_m.group("cap_eff"))
                    rec.per_id_cap = int(cap_m.group("per_id_cap"))
                cos_m = _RE_PAIR_COS.search(line)
                if cos_m and cos_m.group(1) != "skipped":
                    rec.fast_diagnostics = False
                rec.source_path = src
                rec.source_kind = "logger"
                rec.run_id = run_id
                rec.log_mtime = log_mtime
                n_logger += 1
                yield rec

    _ = eval_only


def collect_prune_records(log_paths: List[Path]) -> List[PruneRecord]:
    out: List[PruneRecord] = []
    for p in log_paths:
        try:
            out.extend(iter_log_prune_records(p))
        except OSError as exc:
            print(f"WARN: skip {p}: {exc}", file=sys.stderr)
    return [
        r
        for r in out
        if r.duration_s > 0 and r.tag and not r.tag.startswith("unknown_")
    ]


def dedupe_prune_records(
    records: List[PruneRecord],
    *,
    prefer: str = "logger",
) -> Tuple[List[PruneRecord], int]:
    """按 (tag, kr) 去重；默认优先 reconot logger 时间戳，其次更新 run。"""
    best: Dict[Tuple[str, float], PruneRecord] = {}
    for rec in records:
        key = (rec.tag, rec.kr)
        cur = best.get(key)
        if cur is None or _record_priority(rec) > _record_priority(cur):
            best[key] = rec
        elif prefer == "logger" and rec.source_kind == "logger" and cur.source_kind == "eval_wall":
            best[key] = rec
    deduped = list(best.values())
    deduped.sort(key=lambda r: (r.tag, -r.kr))
    return deduped, len(records) - len(deduped)


def aggregate_by_mesh(records: List[PruneRecord]) -> List[MeshAggregate]:
    by_tag: Dict[str, MeshAggregate] = {}
    for r in records:
        if r.tag not in by_tag:
            by_tag[r.tag] = MeshAggregate(tag=r.tag)
        agg = by_tag[r.tag]
        agg.kr_records[r.kr] = r
    for agg in by_tag.values():
        agg.total_prune_s = sum(r.duration_s for r in agg.kr_records.values())
    return list(by_tag.values())


def _resolve_log_paths(args: argparse.Namespace) -> List[Path]:
    auto = bool(args.discover or getattr(args, "auto_discover", False) or not args.log)
    run_filter = getattr(args, "run", "") or ""
    if auto:
        found = discover_log_paths(REPO_ROOT, run_filter=run_filter)
        if found:
            print("Discovered log files (newest first):")
            for p in found[:10]:
                print(f"  {p}  ({p.stat().st_size // 1024} KiB)")
            return found[: args.max_logs]
    paths: List[Path] = []
    for raw in args.log or []:
        p = Path(raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(p.rglob("*.log"), key=lambda x: -x.stat().st_mtime))
        else:
            import glob as glob_mod

            paths.extend(
                Path(x)
                for x in glob_mod.glob(str(raw))
                if Path(x).is_file()
            )
    return paths


def cmd_analyze_log(args: argparse.Namespace) -> int:
    log_paths = _resolve_log_paths(args)
    if not log_paths:
        print("ERROR: no log files found.", file=sys.stderr)
        print(
            "Hints:\n"
            "  - eval 默认写 ../output/.../eval_run.log 与 ../output/.../logs/reconot.log\n"
            "  - autoresearch master: ../autoresearch-rl-covenant/logs/*.master.log\n"
            "  - 使用: python scripts/debug_reconot_perf.py analyze-log --discover\n"
            "  - 或:   --log ../autoresearch-rl-covenant/logs/eval-reconot-*.master.log",
            file=sys.stderr,
        )
        return 1

    raw_records = collect_prune_records(log_paths)
    if not raw_records:
        print("No prune records found. Tried:")
        for p in log_paths:
            print(f"  {p} ({p.stat().st_size} bytes)")
        print(
            "\nIf file contains [eval] prune reconot lines, ensure mesh id lines like "
            "'[412/3000] <64-hex>' appear before prune lines.\n"
            "Empty logs/reconot.log means eval wrote to output_dir/logs/ instead."
        )
        return 1

    if getattr(args, "no_dedupe", False):
        records = raw_records
        dropped = 0
    else:
        records, dropped = dedupe_prune_records(raw_records)
    dup_rate = dropped / max(1, len(raw_records))

    meshes = aggregate_by_mesh(records)
    meshes.sort(key=lambda m: m.total_prune_s, reverse=True)

    print(f"log_files={len(log_paths)}  primary={log_paths[0]}")
    print(
        f"raw_records={len(raw_records)} deduped_records={len(records)} "
        f"duplicate_dropped={dropped} duplicate_rate={dup_rate:.1%} "
        f"unique_meshes={len(meshes)}"
    )

    # 滚动均值：按出现顺序
    order: List[str] = []
    seen: set[str] = set()
    for r in records:
        if r.tag not in seen and r.kr == 0.75:
            seen.add(r.tag)
            order.append(r.tag)

    window = args.rolling_window
    rolling: List[Tuple[int, float]] = []
    for i, tag in enumerate(order):
        m = next(x for x in meshes if x.tag == tag)
        rolling.append((i + 1, m.total_prune_s))
    if rolling:
        print(f"\n--- rolling total_prune_s per mesh (sum of 3 kr), window={window} ---")
        for i in range(0, len(rolling), max(1, window // 5)):
            idx, _ = rolling[i]
            chunk = [t for j, t in rolling[max(0, i - window + 1) : i + 1]]
            if chunk:
                print(f"  mesh#{idx:4d}  rolling_mean={sum(chunk)/len(chunk):.2f}s  n={len(chunk)}")

        last_w = [t for _, t in rolling[-window:]]
        first_w = [t for _, t in rolling[:window]]
        if first_w and last_w:
            print(f"\n  first_{len(first_w)}_meshes mean={sum(first_w)/len(first_w):.2f}s")
            print(f"  last_{len(last_w)}_meshes mean={sum(last_w)/len(last_w):.2f}s")
            ratio = (sum(last_w) / len(last_w)) / max(1e-9, sum(first_w) / len(first_w))
            print(f"  slowdown_ratio={ratio:.2f}x")

    # 按 kr 分解
    by_kr: Dict[float, List[float]] = defaultdict(list)
    for r in records:
        if r.duration_s > 0:
            by_kr[r.kr].append(r.duration_s)
    print("\n--- per keep_ratio prune duration (deduped) ---")
    for kr in sorted(by_kr.keys(), reverse=True):
        xs = by_kr[kr]
        xs_sorted = sorted(xs)
        p50 = xs_sorted[len(xs_sorted) // 2]
        p95 = xs_sorted[int(len(xs_sorted) * 0.95)]
        p99 = xs_sorted[min(len(xs_sorted) - 1, int(len(xs_sorted) * 0.99))]
        print(
            f"  kr={kr:g}  n={len(xs)}  mean={sum(xs)/len(xs):.3f}s  "
            f"p50={p50:.3f}s  p95={p95:.3f}s  p99={p99:.3f}s  max={max(xs):.3f}s"
        )

    mesh_totals = sorted(m.total_prune_s for m in meshes)
    if mesh_totals:
        mt = mesh_totals
        print(
            f"\n--- per-mesh total (3 kr, deduped) n={len(mt)} ---\n"
            f"  mean={sum(mt)/len(mt):.3f}s  p50={mt[len(mt)//2]:.3f}s  "
            f"p95={mt[int(len(mt)*0.95)]:.3f}s  p99={mt[min(len(mt)-1, int(len(mt)*0.99))]:.3f}s  "
            f"max={mt[-1]:.3f}s"
        )

    # 相关性：慢的是否 cap_needed 高？
    print("\n--- slowest 15 meshes (total prune time) ---")
    print(
        f"{'tag':<20} {'total':>7} {'kr0.75':>7} {'kr0.5':>7} {'kr0.25':>7} "
        f"{'uniq':>5} {'cap_n':>5} {'cap_e':>5} {'pid':>3} {'fast':>4}"
    )
    for m in meshes[:15]:
        r75 = m.kr_records.get(0.75)
        r50 = m.kr_records.get(0.5)
        r25 = m.kr_records.get(0.25)
        ref = r75 or r50 or r25
        if ref is None:
            continue
        print(
            f"{m.tag[:20]:<20} {m.total_prune_s:7.2f} "
            f"{(r75.duration_s if r75 else 0):7.2f} "
            f"{(r50.duration_s if r50 else 0):7.2f} "
            f"{(r25.duration_s if r25 else 0):7.2f} "
            f"{ref.unique_kept:5d} {ref.cap_needed:5d} {ref.cap_eff_final:5d} "
            f"{ref.per_id_cap:3d} {str(ref.fast_diagnostics):>4}"
        )

    # 诊断提示
    slow_kr75 = [r for r in records if r.kr == 0.75 and r.duration_s > args.slow_threshold]
    fast_diag_off = sum(1 for r in records if not r.fast_diagnostics)
    print(f"\n--- heuristics ---")
    print(f"  kr=0.75 slower than {args.slow_threshold}s: {len(slow_kr75)} / {len(by_kr.get(0.75, []))}")
    print(f"  records with full diagnostics (pair_cos computed): {fast_diag_off}")
    if fast_diag_off > 0:
        print("  >> fast_diagnostics=false adds O(k^2) pairwise cosine each prune")
    cap_corr = [
        (
            m.total_prune_s,
            (m.kr_records.get(0.75) or m.kr_records.get(0.5) or next(iter(m.kr_records.values()))).cap_needed,
        )
        for m in meshes
    ]
    cap_corr.sort()
    if len(cap_corr) >= 10:
        low_cap = [t for t, c in cap_corr if c <= 10]
        hi_cap = [t for t, c in cap_corr if c > 50]
        if low_cap and hi_cap:
            print(
                f"  cap_needed<=10 mean_total={sum(low_cap)/len(low_cap):.2f}s  "
                f"cap_needed>50 mean_total={sum(hi_cap)/len(hi_cap):.2f}s"
            )
            print("  >> if high-cap meshes are NOT slower, bottleneck is likely DPP or environment")

    if args.json_out:
        out = {
            "meshes": [
                {
                    "tag": m.tag,
                    "total_prune_s": m.total_prune_s,
                    "kr": {str(k): v.duration_s for k, v in m.kr_records.items()},
                }
                for m in meshes
            ]
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    return 0


def _load_pruner_extra(fast_diagnostics: Optional[bool]) -> dict:
    from eval.config import load_pruner_extra_kwargs

    extra = load_pruner_extra_kwargs(REPO_ROOT / "configs" / "eval", "reconot")
    extra["profile_prune"] = True
    extra["prune_device"] = "cpu"
    if fast_diagnostics is not None:
        extra["fast_diagnostics"] = fast_diagnostics
    return extra


def _profile_one_mesh(
    mesh_id: str,
    *,
    glb_dir: Path,
    mesh_cache_dir: str,
    kr: float,
    extra: dict,
    vqvae: Any,
    vqvae_dev: torch.device,
    vq_emb: torch.nn.Embedding,
) -> Dict[str, Any]:
    from eval.data_loader import mesh_to_tokens
    from eval.pruners import get_pruner_class
    from eval.proposed.reconot import clear_mesh_score_cache

    glb = glb_dir / f"{mesh_id}.glb"
    if not glb.is_file():
        raise FileNotFoundError(f"GLB not found: {glb}")

    clear_mesh_score_cache()
    t0 = time.perf_counter()
    token_ids, voxel_grid = mesh_to_tokens(
        str(glb),
        vqvae,
        vqvae_dev,
        file_identifier=mesh_id,
        mesh_cache_dir=mesh_cache_dir,
    )
    t_tokens = time.perf_counter() - t0

    Pruner = get_pruner_class("reconot")
    timings: Dict[str, float] = {}
    meta_by_kr: Dict[str, Any] = {}
    dpp_stats: Dict[str, int] = {"greedy_fill_rounds": 0}

    import eval.proposed.reconot as reconot_mod

    orig_greedy_fill = reconot_mod._greedy_fill

    def _counting_greedy_fill(*a, **kw):
        kw["stats"] = dpp_stats
        return orig_greedy_fill(*a, **kw)

    reconot_mod._greedy_fill = _counting_greedy_fill
    try:
        for k in KEEP_RATIOS:
            clear_mesh_score_cache()
            pruner = Pruner(keep_ratio=k, seed=42, **extra)
            t1 = time.perf_counter()
            _, meta = pruner.prune(
                token_ids,
                voxel_grid,
                vq_embeddings=vq_emb,
                _log_tag=mesh_id,
                _log_keep_ratio=k,
            )
            timings[f"kr_{k}"] = time.perf_counter() - t1
            diag = meta.get("diagnostics", {})
            meta_by_kr[str(k)] = {
                "profile": diag.get("profile"),
                "cap_needed": diag.get("cap_needed"),
                "cap_eff_final": diag.get("cap_eff_final"),
                "cap_jump_count": diag.get("cap_jump_count"),
                "select_steps": diag.get("select_steps"),
                "unique_token_count": diag.get("unique_token_count"),
            }
    finally:
        reconot_mod._greedy_fill = orig_greedy_fill

    return {
        "mesh_id": mesh_id,
        "mesh_to_tokens_s": t_tokens,
        "prune_timings": timings,
        "total_prune_s": sum(timings.values()),
        "dpp_greedy_fill_rounds": dpp_stats.get("greedy_fill_rounds", 0),
        "meta": meta_by_kr,
        "n_unique_tokens": int(token_ids.unique().numel()),
    }


def _init_vqvae_only(vqvae_device: str) -> Tuple[Any, torch.device, torch.nn.Embedding]:
    from eval.cuda_env import init_cuda_for_eval, load_vqvae, warmup_vqvae, resolve_torch_device

    vqvae_dev = resolve_torch_device(vqvae_device)
    init_cuda_for_eval(vqvae_dev, resolve_torch_device("cpu"))
    print(f"Loading VQVAE on {vqvae_dev}...")
    vqvae = load_vqvae(vqvae_dev)
    warmup_vqvae(vqvae, vqvae_dev, vlm_dev=resolve_torch_device("cpu"))
    vq_emb = vqvae.quantize.embedding
    return vqvae, vqvae_dev, vq_emb


def cmd_profile(args: argparse.Namespace) -> int:
    logging.getLogger("eval.proposed.reconot").setLevel(logging.WARNING)
    fast = None if args.fast_diagnostics == "default" else args.fast_diagnostics == "true"
    extra = _load_pruner_extra(fast)

    vqvae, vqvae_dev, vq_emb = _init_vqvae_only(args.vqvae_device)
    glb_dir = Path(args.glb_dir)
    results: List[Dict[str, Any]] = []

    for rep in range(args.repeat):
        t0 = time.perf_counter()
        row = _profile_one_mesh(
            args.mesh_id,
            glb_dir=glb_dir,
            mesh_cache_dir=args.mesh_cache_dir,
            kr=args.kr,
            extra=extra,
            vqvae=vqvae,
            vqvae_dev=vqvae_dev,
            vq_emb=vq_emb,
        )
        row["repeat"] = rep
        row["wall_s"] = time.perf_counter() - t0
        results.append(row)
        print(f"\n=== repeat {rep} wall={row['wall_s']:.2f}s ===")
        print(f"  mesh_to_tokens: {row['mesh_to_tokens_s']:.3f}s")
        print(f"  total_prune:    {row['total_prune_s']:.3f}s")
        for k, v in row["prune_timings"].items():
            kr_key = k.replace("kr_", "")
            mkr = row["meta"].get(kr_key, {})
            prof = mkr.get("profile") or {}
            print(
                f"    {k}: {v:.3f}s  "
                f"embed={prof.get('embed_gather', 0):.3f} "
                f"rank={prof.get('rank_scores', 0):.3f} "
                f"select={prof.get('select_to_k', 0):.3f} "
                f"diag={prof.get('diagnostics', 0):.3f} "
                f"cap_jump={mkr.get('cap_jump_count')} "
                f"select_steps={mkr.get('select_steps')}"
            )
        print(f"  dpp_greedy_fill_extra_rounds: {row['dpp_greedy_fill_rounds']}")
        print(f"  n_unique_token_ids: {row['n_unique_tokens']}")

    if args.repeat > 1:
        walls = [r["wall_s"] for r in results]
        print(f"\n--- drift over {args.repeat} repeats: min={min(walls):.2f}s max={max(walls):.2f}s ---")
        if max(walls) > 1.5 * min(walls):
            print("  WARN: >1.5x spread — suspect memory pressure or thermal throttling")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def cmd_profile_slowest(args: argparse.Namespace) -> int:
    log_paths = _resolve_log_paths(
        argparse.Namespace(
            log=[args.log] if args.log else None,
            discover=not args.log,
            max_logs=getattr(args, "max_logs", 1),
            auto_discover=False,
            run=getattr(args, "run", ""),
        )
    )
    raw_records = collect_prune_records(log_paths)
    records, _ = dedupe_prune_records(raw_records)
    meshes = aggregate_by_mesh(records)
    meshes.sort(key=lambda m: m.total_prune_s, reverse=True)
    ids = [m.tag for m in meshes[: args.top] if m.tag and not m.tag.startswith("synth_")]
    print(f"Profiling slowest {len(ids)} meshes from log...")

    logging.getLogger("eval.proposed.reconot").setLevel(logging.ERROR)
    extra = _load_pruner_extra(None)
    vqvae, vqvae_dev, vq_emb = _init_vqvae_only(args.vqvae_device)
    glb_dir = Path(args.glb_dir)

    summary: List[Dict[str, Any]] = []
    for mid in ids:
        try:
            row = _profile_one_mesh(
                mid,
                glb_dir=glb_dir,
                mesh_cache_dir=args.mesh_cache_dir,
                kr=0.75,
                extra=extra,
                vqvae=vqvae,
                vqvae_dev=vqvae_dev,
                vq_emb=vq_emb,
            )
            prof75 = row["meta"].get("0.75", {}).get("profile", {}) or {}
            summary.append(
                {
                    "mesh_id": mid,
                    "total_prune_s": row["total_prune_s"],
                    "rank_scores_075": prof75.get("rank_scores", 0),
                    "select_to_k_075": prof75.get("select_to_k", 0),
                    "diagnostics_075": prof75.get("diagnostics", 0),
                    "cap_jump_075": row["meta"].get("0.75", {}).get("cap_jump_count"),
                    "dpp_fill_rounds": row["dpp_greedy_fill_rounds"],
                }
            )
            print(
                f"{mid[:16]}... total={row['total_prune_s']:.2f}s "
                f"rank={prof75.get('rank_scores', 0):.2f}s "
                f"select={prof75.get('select_to_k', 0):.2f}s "
                f"diag={prof75.get('diagnostics', 0):.2f}s "
                f"dpp_rounds={row['dpp_greedy_fill_rounds']}"
            )
        except FileNotFoundError as e:
            print(f"  SKIP {mid}: {e}")

    if summary:
        rank_times = [s["rank_scores_075"] for s in summary]
        print(f"\nkr=0.75 rank_scores: mean={sum(rank_times)/len(rank_times):.2f}s max={max(rank_times):.2f}s")
        if max(rank_times) > 3.0:
            print("  >> DPP rank_scores dominates — check dpp_fill_rounds and fast_diagnostics")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def cmd_drift_test(args: argparse.Namespace) -> int:
    args.repeat = args.repeat
    args.mesh_id = args.mesh_id
    args.fast_diagnostics = "default"
    args.json_out = args.json_out
    return cmd_profile(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="ReconOT performance debugger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_log = sub.add_parser("analyze-log", help="Parse eval_run.log / reconot.log / master.log")
    p_log.add_argument(
        "--log",
        nargs="*",
        default=None,
        help="Log file(s), directory, or glob (default: --discover)",
    )
    p_log.add_argument(
        "--discover",
        action="store_true",
        help="Auto-find eval_run.log / reconot.log / master.log under artifacts & autoresearch",
    )
    p_log.add_argument(
        "--auto-discover",
        action="store_true",
        dest="discover",
        help="Alias for --discover",
    )
    p_log.add_argument("--max-logs", type=int, default=1, help="With --discover, max files to merge")
    p_log.add_argument(
        "--run",
        default="",
        help="Only logs under artifacts/.../runs/run-XXXX (e.g. run-0001 or 0001)",
    )
    p_log.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate (tag, kr) records from merged logs",
    )
    p_log.add_argument("--rolling-window", type=int, default=50)
    p_log.add_argument("--slow-threshold", type=float, default=5.0)
    p_log.add_argument("--json-out", default="")

    p_prof = sub.add_parser("profile", help="Profile one mesh with phase breakdown")
    p_prof.add_argument("--mesh-id", required=True)
    p_prof.add_argument("--kr", type=float, default=0.75)
    p_prof.add_argument("--repeat", type=int, default=1)
    p_prof.add_argument("--glb-dir", default=str(REPO_ROOT.parent / "data"))
    p_prof.add_argument("--mesh-cache-dir", default=str(REPO_ROOT.parent / "data" / "mesh_voxel_cache"))
    p_prof.add_argument("--vqvae-device", default="cuda:0")
    p_prof.add_argument(
        "--fast-diagnostics",
        choices=("default", "true", "false"),
        default="default",
    )
    p_prof.add_argument("--json-out", default="")

    p_slow = sub.add_parser("profile-slowest", help="Profile top-N slowest from log")
    p_slow.add_argument("--log", default=str(REPO_ROOT.parent / "output" / "logs" / "reconot.log"))
    p_slow.add_argument("--top", type=int, default=10)
    p_slow.add_argument("--max-logs", type=int, default=1)
    p_slow.add_argument("--run", default="")
    p_slow.add_argument("--glb-dir", default=str(REPO_ROOT.parent / "data"))
    p_slow.add_argument("--mesh-cache-dir", default=str(REPO_ROOT.parent / "data" / "mesh_voxel_cache"))
    p_slow.add_argument("--vqvae-device", default="cuda:0")
    p_slow.add_argument("--json-out", default="")

    p_drift = sub.add_parser("drift-test", help="Repeat same mesh to detect runtime drift")
    p_drift.add_argument("--mesh-id", required=True)
    p_drift.add_argument("--repeat", type=int, default=20)
    p_drift.add_argument("--glb-dir", default=str(REPO_ROOT.parent / "data"))
    p_drift.add_argument("--mesh-cache-dir", default=str(REPO_ROOT.parent / "data" / "mesh_voxel_cache"))
    p_drift.add_argument("--vqvae-device", default="cuda:0")
    p_drift.add_argument("--json-out", default="")

    args = parser.parse_args()
    if args.cmd == "analyze-log":
        return cmd_analyze_log(args)
    if args.cmd == "profile":
        return cmd_profile(args)
    if args.cmd == "profile-slowest":
        return cmd_profile_slowest(args)
    if args.cmd == "drift-test":
        return cmd_drift_test(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
