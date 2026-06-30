"""
Merge ``results.json`` from multiple partial eval runs (e.g. one directory per GPU) and
recompute ``summary.csv`` and ``by_pruner/`` like ``run_eval``.

Optimized for large outputs: parallel reads across inputs, optional ``orjson``
serialization, compact (non-pretty) merged JSON, and parallel per-pruner writes.

Usage (from ``3d-token-prune-eval-main``)::

    python -m eval.merge_eval_results \\
      --inputs ../output/eval_results/gpu0 ../output/eval_results/gpu1 \\
      --output-dir ../output/eval_results/merged
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.utils import (  # noqa: E402
    _HAS_ORJSON,
    aggregate_summary,
    compute_pct_of_full,
    load_json,
    save_json_fast,
    split_results_by_pruner,
    write_summary_csv,
)


def _log(msg: str, t0: float) -> None:
    print(f"[merge +{time.time() - t0:7.2f}s] {msg}", flush=True)


def _load_one(path: str) -> list:
    part = load_json(path)
    if not isinstance(part, list):
        raise ValueError(f"Expected list in {path}, got {type(part).__name__}")
    return part


def _load_one_with_meta(path: str) -> tuple:
    t = time.time()
    part = _load_one(path)
    return path, part, time.time() - t


def _results_json_parent_name(results_json_path: str) -> str:
    """e.g. ``.../gpu2/results.json`` -> ``gpu2``."""
    return os.path.basename(os.path.dirname(os.path.abspath(results_json_path)))


def _filter_gpu2_loco3d_when_gpu3_has_loco3d(
    input_paths: list[str], parts: list[list], t0: float
) -> list[list]:
    """
    合并时若 gpu3 含 loco3d（V7 重跑），则剔除 gpu2 的 loco3d（V6 旧版），避免 n_samples=198 混版本。
    若 gpu3 无 loco3d，则保留 gpu2 结果以免丢方法。
    """
    gpu3_idx = next((i for i, p in enumerate(input_paths) if _results_json_parent_name(p) == "gpu3"), None)
    gpu3_has_loco = False
    if gpu3_idx is not None:
        gpu3_has_loco = any(isinstance(r, dict) and r.get("pruner") == "loco3d" for r in parts[gpu3_idx])
    out: list[list] = []
    for i, path in enumerate(input_paths):
        part = parts[i]
        if _results_json_parent_name(path) == "gpu2" and gpu3_has_loco:
            before = len(part)
            part = [r for r in part if not (isinstance(r, dict) and r.get("pruner") == "loco3d")]
            if before != len(part):
                _log(f"filtered loco3d from gpu2 (keep gpu3 V7): {path} rows {before}->{len(part)}", t0)
        out.append(part)
    return out


def _dedupe_key(row: dict) -> tuple | None:
    """Stable identity for one eval attempt; rows without enough fields are not deduped."""
    if not isinstance(row, dict):
        return None
    fid = row.get("file_identifier")
    pruner = row.get("pruner")
    keep_ratio = row.get("keep_ratio")
    if fid is None or pruner is None or keep_ratio is None:
        return None
    try:
        kr = round(float(keep_ratio), 12)
    except (TypeError, ValueError):
        return None
    return str(fid), str(pruner), kr


def _duplicate_stats(rows: list[dict]) -> tuple[int, int]:
    seen: dict[tuple, int] = {}
    duplicate_keys = 0
    duplicate_excess = 0
    for row in rows:
        key = _dedupe_key(row)
        if key is None:
            continue
        if key in seen:
            if seen[key] == 1:
                duplicate_keys += 1
            seen[key] += 1
            duplicate_excess += 1
        else:
            seen[key] = 1
    return duplicate_keys, duplicate_excess


def _dedupe_rows(rows: list[dict], *, keep: str) -> list[dict]:
    keyed: dict[tuple, dict] = {}
    passthrough: list[dict] = []
    order: list[tuple] = []
    for row in rows:
        key = _dedupe_key(row)
        if key is None:
            passthrough.append(row)
            continue
        if keep == "first" and key in keyed:
            continue
        if key not in keyed:
            order.append(key)
        keyed[key] = row
    return [keyed[key] for key in order] + passthrough


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge partial eval results.json directories")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Directories each containing a results.json",
    )
    p.add_argument("--output-dir", type=str, required=True, help="Merged output directory")
    p.add_argument(
        "--indent",
        type=int,
        default=None,
        help=(
            "JSON indentation for the big merged results.json and per-pruner files. "
            "Default None (compact, fastest). Pass 2 for human-pretty at much higher cost."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for reading inputs and writing by_pruner files. "
        "Default = min(#tasks, os.cpu_count()).",
    )
    p.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel read/write; do everything serially.",
    )
    p.add_argument(
        "--skip-merged-json",
        action="store_true",
        help="Do not write the big merged results.json (summary.csv and by_pruner/ still written).",
    )
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate (file_identifier, pruner, keep_ratio) rows before writing outputs.",
    )
    p.add_argument(
        "--dedupe-keep",
        choices=("first", "last"),
        default="first",
        help="Which duplicate row to keep when --dedupe is set.",
    )
    args = p.parse_args(argv)

    t0 = time.time()
    _log(
        f"start; inputs={len(args.inputs)} indent={args.indent} "
        f"orjson={'yes' if _HAS_ORJSON else 'no'} "
        f"workers={args.workers or 'auto'} parallel={not args.no_parallel}",
        t0,
    )

    input_paths: list[str] = []
    for d in args.inputs:
        path = os.path.join(d, "results.json")
        if not os.path.isfile(path):
            print(f"Missing {path}", file=sys.stderr)
            return 1
        input_paths.append(path)

    parts: list[list] = [None] * len(input_paths)  # type: ignore[list-item]
    if args.no_parallel or len(input_paths) == 1:
        for i, path in enumerate(input_paths):
            t = time.time()
            parts[i] = _load_one(path)
            _log(f"loaded {path} rows={len(parts[i])} in {time.time() - t:.2f}s", t0)
    else:
        max_workers = args.workers or min(len(input_paths), (os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_load_one_with_meta, path): i for i, path in enumerate(input_paths)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                path, part, dt = fut.result()
                parts[i] = part
                _log(f"loaded {path} rows={len(part)} in {dt:.2f}s", t0)

    parts = _filter_gpu2_loco3d_when_gpu3_has_loco3d(input_paths, parts, t0)

    merged: list[dict] = []
    for part in parts:
        merged.extend(part)
    _log(f"merged in-memory total rows={len(merged)}", t0)
    duplicate_keys, duplicate_excess = _duplicate_stats(merged)
    if duplicate_keys:
        _log(
            f"duplicate eval keys detected: keys={duplicate_keys} excess_rows={duplicate_excess}",
            t0,
        )
        if args.dedupe:
            before = len(merged)
            merged = _dedupe_rows(merged, keep=args.dedupe_keep)
            _log(
                f"deduped merged rows {before}->{len(merged)} keep={args.dedupe_keep}",
                t0,
            )
        else:
            _log("duplicates kept; pass --dedupe to drop repeated eval attempts", t0)

    os.makedirs(args.output_dir, exist_ok=True)

    out_json = os.path.join(args.output_dir, "results.json")
    if args.skip_merged_json:
        _log(f"skipping merged results.json (per --skip-merged-json)", t0)
    else:
        t = time.time()
        save_json_fast(out_json, merged, indent=args.indent)
        size_mb = os.path.getsize(out_json) / (1024 * 1024)
        _log(f"wrote {out_json} ({size_mb:.1f} MB) in {time.time() - t:.2f}s", t0)

    t = time.time()
    ok = [r for r in merged if "generated_caption" in r]
    summary = aggregate_summary(ok)
    summary = compute_pct_of_full(summary)
    out_csv = os.path.join(args.output_dir, "summary.csv")
    write_summary_csv(out_csv, summary)
    _log(
        f"wrote {out_csv} rows={len(summary)} (ok_rows={len(ok)}) in {time.time() - t:.2f}s",
        t0,
    )

    t = time.time()
    written = split_results_by_pruner(
        args.output_dir,
        merged,
        parallel=not args.no_parallel,
        indent=args.indent,
        max_workers=args.workers,
    )
    _log(
        f"wrote {len(written)} files under {os.path.join(args.output_dir, 'by_pruner')} "
        f"in {time.time() - t:.2f}s",
        t0,
    )

    _log(f"done. total elapsed {time.time() - t0:.2f}s", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
