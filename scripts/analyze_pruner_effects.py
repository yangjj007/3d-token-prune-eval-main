#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small-sample diagnostics for ShapeLLM mesh token pruners."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ["SHAPELLM_EVAL_LIGHT"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings, target_keep_count  # noqa: E402
from eval.config import EvalConfig, load_pruner_extra_kwargs, resolve_repo_paths  # noqa: E402
from eval.cuda_env import init_cuda_for_eval, resolve_torch_device  # noqa: E402
from eval.data_loader import iter_dataset, mesh_to_tokens  # noqa: E402
from eval.metrics import compute_text_metrics  # noqa: E402
from eval.pruners import ensure_pruners_loaded, get_pruner_class  # noqa: E402
from eval.proposed._spatial import latent_surface_mask  # noqa: E402
from eval.run_eval import load_llm, load_vqvae  # noqa: E402
from eval.utils import tokens_to_mesh_string  # noqa: E402

ensure_pruners_loaded()

DEFAULT_METHODS = (
    "no_pruning",
    "random",
    "uniform",
    "otprune",
    "apet",
    "divprune",
    "fastv_mesh",
    "tome",
    "loco3d",
    "loco3d_dpp",
    "loco3d_nonempty_dpp",
    "runlength_curve",
    "octree_merge",
    "reconot",
)

SUBSET_METHODS = {
    "no_pruning",
    "random",
    "uniform",
    "otprune",
    "apet",
    "divprune",
    "fastv_mesh",
    "loco3d",
    "loco3d_dpp",
    "loco3d_nonempty_dpp",
    "runlength_curve",
    "octree_merge",
    "reconot",
}


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _indices_from_meta(meta: Dict[str, Any]) -> List[int] | None:
    raw = meta.get("indices")
    if not isinstance(raw, list):
        return None
    out = []
    for value in raw:
        try:
            i = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= i < MESH_SEQ_LEN:
            out.append(i)
    return out


def _mask(indices: Sequence[int]) -> np.ndarray:
    arr = np.zeros(MESH_SEQ_LEN, dtype=bool)
    for i in indices:
        if 0 <= int(i) < MESH_SEQ_LEN:
            arr[int(i)] = True
    return arr


def _jaccard(a: Sequence[int], b: Sequence[int]) -> Tuple[float, int, int]:
    ma, mb = _mask(a), _mask(b)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return (inter / union if union else 0.0, inter, union)


def _token_id_jaccard(ids_a: Sequence[int], ids_b: Sequence[int]) -> float:
    a, b = set(int(x) for x in ids_a), set(int(x) for x in ids_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _occupancy_metrics(indices: Sequence[int] | None, voxel_grid: torch.Tensor | None) -> Dict[str, float | None]:
    if indices is None or voxel_grid is None:
        return {
            "kept_surface_fraction": None,
            "kept_filled_fraction": None,
            "kept_empty_fraction": None,
            "surface_recall": None,
            "nonempty_recall": None,
        }
    is_surface, is_empty, is_filled, _ = latent_surface_mask(voxel_grid.cpu())
    idx = torch.tensor(indices, dtype=torch.long)
    if idx.numel() == 0:
        return {
            "kept_surface_fraction": 0.0,
            "kept_filled_fraction": 0.0,
            "kept_empty_fraction": 0.0,
            "surface_recall": 0.0,
            "nonempty_recall": 0.0,
        }
    kept_surface = is_surface[idx]
    kept_empty = is_empty[idx]
    kept_filled = is_filled[idx]
    nonempty = ~is_empty
    return {
        "kept_surface_fraction": float(kept_surface.float().mean().item()),
        "kept_filled_fraction": float(kept_filled.float().mean().item()),
        "kept_empty_fraction": float(kept_empty.float().mean().item()),
        "surface_recall": float(kept_surface.sum().item() / max(int(is_surface.sum().item()), 1)),
        "nonempty_recall": float((~kept_empty).sum().item() / max(int(nonempty.sum().item()), 1)),
    }


def _embedding_metrics(
    indices: Sequence[int] | None,
    pruned_ids: torch.Tensor,
    full_token_ids: torch.Tensor,
    vq_embeddings: torch.nn.Embedding,
) -> Dict[str, float | None]:
    if indices is None:
        feats = gather_embeddings(vq_embeddings, pruned_ids.detach().long().view(-1))
    else:
        all_feats = gather_embeddings(vq_embeddings, full_token_ids.detach().long().view(-1))
        idx = torch.tensor(indices, dtype=torch.long, device=all_feats.device)
        feats = all_feats.index_select(0, idx) if idx.numel() else all_feats[:0]
    if feats.numel() == 0:
        return {
            "embedding_pair_cos_mean": None,
            "embedding_pair_cos_std": None,
            "embedding_unique_token_ids": float(pruned_ids.unique().numel()),
            "embedding_mean_min_dist_to_kept": None,
        }

    f = feats.float()
    f_norm = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if f_norm.size(0) > 1:
        cos = f_norm @ f_norm.t()
        off_diag = cos[~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)]
        pair_mean = float(off_diag.mean().item())
        pair_std = float(off_diag.std(unbiased=False).item())
    else:
        pair_mean = None
        pair_std = None

    full_feats = gather_embeddings(vq_embeddings, full_token_ids.detach().long().view(-1)).float()
    if f.size(0) > 0:
        dist = torch.cdist(full_feats, f, p=2.0)
        mean_min_dist = float(dist.min(dim=1).values.mean().item())
    else:
        mean_min_dist = None
    return {
        "embedding_pair_cos_mean": pair_mean,
        "embedding_pair_cos_std": pair_std,
        "embedding_unique_token_ids": float(pruned_ids.unique().numel()),
        "embedding_mean_min_dist_to_kept": mean_min_dist,
    }


def _run_pruner(
    method: str,
    keep_ratio: float,
    seed: int,
    token_ids: torch.Tensor,
    voxel_grid: torch.Tensor | None,
    vq_embeddings: torch.nn.Embedding,
    eval_config_dir: Path,
    file_identifier: str,
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    extra = load_pruner_extra_kwargs(eval_config_dir, method)
    pruner = get_pruner_class(method)(keep_ratio=keep_ratio, seed=seed, **extra)
    t0 = time.perf_counter()
    pruned, meta = pruner.prune(
        token_ids,
        voxel_grid,
        vq_embeddings=vq_embeddings,
        _log_tag=file_identifier,
        _log_keep_ratio=float(keep_ratio),
    )
    return pruned.detach().long().view(-1).cpu(), meta, time.perf_counter() - t0


def _csv_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    preferred = [
        "event",
        "sample_idx",
        "file_identifier",
        "keep_ratio",
        "method",
        "trial",
        "selection_type",
        "k_target",
        "k_actual",
        "unique_indices",
        "unique_token_ids",
        "prune_time_sec",
        "bleu_1",
        "bleu_4",
        "rouge_l",
        "sentence_bert",
        "simcse",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_method(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event") == "method_stats":
            groups[(str(row["method"]), float(row["keep_ratio"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (method, kr), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        numeric_keys = sorted(
            k for row in items for k, v in row.items() if _safe_float(v) is not None
        )
        rec: Dict[str, Any] = {
            "event": "aggregate_method",
            "method": method,
            "keep_ratio": kr,
            "n_rows": len(items),
        }
        for key in numeric_keys:
            if key in {"sample_idx", "keep_ratio", "trial"}:
                continue
            rec[f"mean_{key}"] = _mean(_safe_float(row.get(key)) for row in items)
        out.append(rec)
    return out


def _pairwise_rows(
    sample_idx: int,
    file_identifier: str,
    keep_ratio: float,
    runs: Dict[str, Dict[str, Any]],
    token_ids: torch.Tensor,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    names = sorted(runs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            idx_a = runs[a].get("indices")
            idx_b = runs[b].get("indices")
            if idx_a is None or idx_b is None:
                continue
            jac, inter, union = _jaccard(idx_a, idx_b)
            kr = len(idx_a) / MESH_SEQ_LEN
            rand_expected = (kr * kr) / (2 * kr - kr * kr + 1e-12)
            ids_a = token_ids[torch.tensor(idx_a, dtype=torch.long)].tolist()
            ids_b = token_ids[torch.tensor(idx_b, dtype=torch.long)].tolist()
            out.append(
                {
                    "event": "pairwise",
                    "sample_idx": sample_idx,
                    "file_identifier": file_identifier,
                    "keep_ratio": keep_ratio,
                    "method_a": a,
                    "method_b": b,
                    "jaccard": jac,
                    "intersection": inter,
                    "union": union,
                    "jaccard_minus_random_expected": jac - rand_expected,
                    "token_id_jaccard": _token_id_jaccard(ids_a, ids_b),
                }
            )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose ShapeLLM 3D token pruning behavior.")
    parser.add_argument("--data-csv", type=str, default="data/metadata.csv")
    parser.add_argument("--glb-dir", type=str, default="sampled_objaverse_data")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--keep-ratios", type=str, default="1.0,0.75,0.5,0.25,0.1")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--vqvae-device", type=str, default="cuda:0")
    parser.add_argument("--device", type=str, default="cuda:0", help="VLM device, only used with --with-vlm.")
    parser.add_argument("--eval-config-dir", type=str, default="configs/eval")
    parser.add_argument("--mesh-cache-dir", type=str, default="")
    parser.add_argument("--mesh-cache-readonly", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "pruner_effects")
    parser.add_argument("--with-vlm", action="store_true")
    parser.add_argument("--model-id", type=str, default="yejunliang23/ShapeLLM-7B-omni")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load VLM in 4-bit NF4 for single-GPU smoke tests.")
    parser.add_argument("--vlm-torch-dtype", type=str, default="bfloat16", choices=("auto", "float16", "bfloat16"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=8192)
    parser.add_argument("--caption-prompt", type=str, default=EvalConfig.caption_prompt)
    args = parser.parse_args(argv)

    cfg = EvalConfig(
        data_csv=args.data_csv,
        glb_dir=args.glb_dir,
        eval_config_dir=args.eval_config_dir,
        mesh_cache_dir=args.mesh_cache_dir,
        mesh_cache_readonly=args.mesh_cache_readonly,
    )
    cfg = resolve_repo_paths(cfg, REPO_ROOT)
    methods = _parse_str_list(args.methods)
    keep_ratios = _parse_float_list(args.keep_ratios)
    for method in methods:
        get_pruner_class(method)

    vqvae_dev = resolve_torch_device(args.vqvae_device)
    vlm_dev = resolve_torch_device(args.device)
    init_cuda_for_eval(vqvae_dev, vlm_dev if args.with_vlm else vqvae_dev)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading VQVAE on {vqvae_dev}...")
    vqvae = load_vqvae(vqvae_dev)
    vq_embeddings = getattr(vqvae.vq, "embeddings", None)
    if vq_embeddings is None:
        raise SystemExit("VQVAE has no vq.embeddings")

    model = processor = tokenizer = None
    if args.with_vlm:
        print(f"Loading VLM on {vlm_dev}...")
        model, processor, tokenizer = load_llm(
            args.model_id,
            vlm_dev,
            load_in_4bit=args.load_in_4bit,
            vlm_torch_dtype=args.vlm_torch_dtype,
        )
        from eval.generator import generate_caption
    else:
        generate_caption = None  # type: ignore[assignment]

    samples = list(
        iter_dataset(
            cfg.data_csv,
            cfg.glb_dir,
            num_samples=args.num_samples,
            skip_missing_glb=True,
        )
    )
    if not samples:
        raise SystemExit("No samples found; check --data-csv and --glb-dir.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"pruner_effects_{ts}.jsonl"
    csv_path = out_dir / f"pruner_effects_{ts}.csv"
    rows: List[Dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        def emit(record: Dict[str, Any]) -> None:
            rows.append(record)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()

        emit(
            {
                "event": "run_config",
                "methods": methods,
                "keep_ratios": keep_ratios,
                "num_samples": len(samples),
                "seed": args.seed,
                "random_trials": args.random_trials,
                "with_vlm": args.with_vlm,
                "temperature": args.temperature,
            }
        )

        for sample_idx, sample in enumerate(samples):
            print(f"[{sample_idx + 1}/{len(samples)}] {sample.file_identifier}")
            try:
                token_ids, voxel_grid = mesh_to_tokens(
                    sample.glb_path,
                    vqvae,
                    vqvae_dev,
                    file_identifier=sample.file_identifier,
                    mesh_cache_dir=cfg.mesh_cache_dir,
                    mesh_cache_readonly=cfg.mesh_cache_readonly,
                    vlm_device=vlm_dev if args.with_vlm else None,
                )
            except Exception as exc:
                emit(
                    {
                        "event": "sample_error",
                        "sample_idx": sample_idx,
                        "file_identifier": sample.file_identifier,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            token_ids_cpu = token_ids.detach().long().view(-1).cpu()
            token_counts = Counter(int(x) for x in token_ids_cpu.tolist())
            emit(
                {
                    "event": "sample_stats",
                    "sample_idx": sample_idx,
                    "file_identifier": sample.file_identifier,
                    "unique_token_ids_full": len(token_counts),
                    "top1_token_fraction": token_counts.most_common(1)[0][1] / MESH_SEQ_LEN if token_counts else 0.0,
                }
            )

            for kr in keep_ratios:
                k_target = target_keep_count(kr)
                primary_runs: Dict[str, Dict[str, Any]] = {}
                for method in methods:
                    trials = args.random_trials if method == "random" else 1
                    for trial in range(trials):
                        seed = args.seed + trial
                        try:
                            pruned, meta, prune_time = _run_pruner(
                                method,
                                kr,
                                seed,
                                token_ids_cpu,
                                voxel_grid,
                                vq_embeddings,
                                Path(cfg.eval_config_dir),
                                sample.file_identifier,
                            )
                        except Exception as exc:
                            emit(
                                {
                                    "event": "method_error",
                                    "sample_idx": sample_idx,
                                    "file_identifier": sample.file_identifier,
                                    "keep_ratio": kr,
                                    "method": method,
                                    "trial": trial,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            continue

                        indices = _indices_from_meta(meta)
                        selection_type = "subset" if indices is not None and method in SUBSET_METHODS else "merge_or_recode"
                        unique_indices = len(set(indices)) if indices is not None else None
                        row: Dict[str, Any] = {
                            "event": "method_stats",
                            "sample_idx": sample_idx,
                            "file_identifier": sample.file_identifier,
                            "keep_ratio": kr,
                            "method": method,
                            "trial": trial,
                            "selection_type": selection_type,
                            "k_target": k_target,
                            "k_actual": int(pruned.numel()),
                            "unique_indices": unique_indices,
                            "unique_token_ids": int(pruned.unique().numel()),
                            "prune_time_sec": prune_time,
                            **_occupancy_metrics(indices, voxel_grid),
                            **_embedding_metrics(indices, pruned, token_ids_cpu, vq_embeddings),
                        }
                        if int(pruned.numel()) != k_target:
                            row["k_delta"] = int(pruned.numel()) - k_target
                        if unique_indices is not None:
                            row["duplicate_index_count"] = int(pruned.numel()) - unique_indices
                        emit(row)

                        if trial == 0:
                            primary_runs[method] = {
                                "indices": indices,
                                "pruned": pruned,
                                "selection_type": selection_type,
                            }

                        if args.with_vlm and generate_caption is not None and model is not None:
                            mesh_str = tokens_to_mesh_string(pruned)
                            try:
                                caption, elapsed, n_in, n_out = generate_caption(
                                    model,
                                    processor,
                                    tokenizer,
                                    mesh_str,
                                    args.caption_prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    top_k=args.top_k,
                                    device=vlm_dev,
                                )
                                scores = compute_text_metrics(caption, sample.captions)
                                emit(
                                    {
                                        "event": "vlm_caption",
                                        "sample_idx": sample_idx,
                                        "file_identifier": sample.file_identifier,
                                        "keep_ratio": kr,
                                        "method": method,
                                        "trial": trial,
                                        "caption": caption,
                                        "generation_time_sec": elapsed,
                                        "num_input_tokens": n_in,
                                        "num_output_tokens": n_out,
                                        **scores,
                                    }
                                )
                            except Exception as exc:
                                emit(
                                    {
                                        "event": "vlm_error",
                                        "sample_idx": sample_idx,
                                        "file_identifier": sample.file_identifier,
                                        "keep_ratio": kr,
                                        "method": method,
                                        "trial": trial,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )

                for row in _pairwise_rows(sample_idx, sample.file_identifier, kr, primary_runs, token_ids_cpu):
                    emit(row)

        for row in _aggregate_method(rows):
            emit(row)

    _write_csv(csv_path, rows)
    print(f"wrote {jsonl_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
