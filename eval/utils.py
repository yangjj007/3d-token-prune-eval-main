"""Mesh token string helpers and result I/O."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

try:
    import orjson as _orjson  # type: ignore

    _HAS_ORJSON = True
except Exception:
    _orjson = None
    _HAS_ORJSON = False


def tokens_to_mesh_string(token_ids: Sequence[int] | torch.Tensor) -> str:
    """
    Build ``<mesh-start><mesh{id}>...<mesh-end>`` from a variable-length
    list of VQ codebook indices (matches training format; length may be < 1024 after pruning).
    """
    if isinstance(token_ids, torch.Tensor):
        ids = token_ids.detach().cpu().long().view(-1).tolist()
    else:
        ids = [int(x) for x in token_ids]
    parts = ["<mesh-start>"]
    for j in ids:
        parts.append(f"<mesh{j}>")
    parts.append("<mesh-end>")
    return "".join(parts)


def tokens_to_mesh_string_fixed1024(token_list: Sequence[int]) -> str:
    """Original app.py behavior: exactly 1024 mesh slots (legacy)."""
    assert len(token_list) == 1024
    return tokens_to_mesh_string(token_list)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_json_fast(path: str, data: Any, *, indent: Optional[int] = None) -> None:
    """
    Faster JSON writer for large result lists.

    Prefers ``orjson`` when available (binary write, no Python-level per-item loop).
    Falls back to stdlib ``json``. ``indent=None`` (compact) avoids the
    pretty-printer which is the main bottleneck on huge arrays.
    """
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    if _HAS_ORJSON and (indent is None or indent == 2):
        opts = 0
        if indent == 2:
            opts |= _orjson.OPT_INDENT_2  # type: ignore[attr-defined]
        opts |= _orjson.OPT_SERIALIZE_NUMPY | _orjson.OPT_NON_STR_KEYS  # type: ignore[attr-defined]
        with open(path, "wb") as f:
            f.write(_orjson.dumps(data, option=opts))  # type: ignore[attr-defined]
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _save_json_fast_worker(args: tuple) -> str:
    path, data, indent = args
    save_json_fast(path, data, indent=indent)
    return path


def save_json_many_parallel(
    items: List[tuple],
    *,
    indent: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[str]:
    """
    Write many JSON files in parallel using processes (CPU-bound serialization).

    ``items`` is a list of ``(path, data)`` tuples.
    """
    if not items:
        return []
    tasks = [(p, d, indent) for (p, d) in items]
    if max_workers is None:
        max_workers = min(len(tasks), (os.cpu_count() or 4))
    written: List[str] = []
    if max_workers <= 1 or len(tasks) == 1:
        for t in tasks:
            written.append(_save_json_fast_worker(t))
        return written
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for path in ex.map(_save_json_fast_worker, tasks):
            written.append(path)
    return written


def load_json(path: str) -> Any:
    if _HAS_ORJSON:
        with open(path, "rb") as f:
            return _orjson.loads(f.read())  # type: ignore[attr-defined]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by (model_backend, pruner, keep_ratio): mean text metrics / time / token counts.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    include_backend = any("model_backend" in r for r in rows)
    for r in rows:
        backend = r.get("model_backend", "shapellm") if include_backend else None
        key = (backend, r.get("pruner"), float(r.get("keep_ratio", 0)))
        groups[key].append(r)

    out: List[Dict[str, Any]] = []
    for (model_backend, pruner, keep_ratio), items in sorted(
        groups.items(), key=lambda x: (x[0][0] or "", x[0][1] or "", x[0][2])
    ):
        n = len(items)

        def mean(key: str) -> float | None:
            vals = [x.get(key) for x in items if x.get(key) is not None]
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        def std(key: str) -> float | None:
            vals = [x.get(key) for x in items if x.get(key) is not None]
            if len(vals) < 2:
                return 0.0
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            return float(var**0.5)

        row = {
            "pruner": pruner,
            "keep_ratio": keep_ratio,
            "n_samples": n,
            "bleu_1_mean": mean("bleu_1"),
            "bleu_2_mean": mean("bleu_2"),
            "bleu_3_mean": mean("bleu_3"),
            "bleu_4_mean": mean("bleu_4"),
            "rouge_l_mean": mean("rouge_l"),
            "sentence_bert_mean": mean("sentence_bert"),
            "simcse_mean": mean("simcse"),
            "generation_time_sec_mean": mean("generation_time_sec"),
            "generation_time_sec_std": std("generation_time_sec"),
            "num_input_tokens_mean": mean("num_input_tokens"),
            "num_output_tokens_mean": mean("num_output_tokens"),
            "num_tokens_pruned_mean": mean("num_tokens_pruned"),
            "pruner_tflops_mean": mean("pruner_tflops"),
            "llm_prefill_tflops_mean": mean("llm_prefill_tflops"),
            "llm_decode_tflops_mean": mean("llm_decode_tflops"),
            "llm_total_tflops_mean": mean("llm_total_tflops"),
            "total_tflops_mean": mean("total_tflops"),
        }
        if include_backend:
            row = {"model_backend": model_backend or "shapellm", **row}
        out.append(row)
    return out


def compute_pct_of_full(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each (keep_ratio), use ``no_pruning`` row as 100% reference for quality and TFLOP columns.
    Appends ``*_pct_of_full`` fields (percentage, 0--100+).
    """
    include_backend = any("model_backend" in row for row in summary_rows)
    ref_by_key: Dict[tuple, Dict[str, Any]] = {}
    for row in summary_rows:
        if row.get("pruner") == "no_pruning":
            backend = row.get("model_backend", "shapellm") if include_backend else None
            ref_by_key[(backend, float(row.get("keep_ratio", 1.0)))] = row

    def pct(val: float | None, base_val: float | None) -> float | None:
        if val is None or base_val is None:
            return None
        if base_val == 0.0:
            return None
        return 100.0 * float(val) / float(base_val)

    metric_keys = (
        "bleu_1_mean",
        "bleu_2_mean",
        "bleu_3_mean",
        "bleu_4_mean",
        "rouge_l_mean",
        "sentence_bert_mean",
        "simcse_mean",
        "pruner_tflops_mean",
        "llm_prefill_tflops_mean",
        "llm_decode_tflops_mean",
        "llm_total_tflops_mean",
        "total_tflops_mean",
        "num_input_tokens_mean",
        "num_output_tokens_mean",
    )

    out: List[Dict[str, Any]] = []
    for row in summary_rows:
        new_row = dict(row)
        kr = float(row.get("keep_ratio", 0.0))
        backend = row.get("model_backend", "shapellm") if include_backend else None
        base = ref_by_key.get((backend, kr)) or ref_by_key.get((backend, 1.0))
        if base is None:
            out.append(new_row)
            continue
        for mk in metric_keys:
            if mk not in row:
                continue
            suffix = mk.replace("_mean", "")
            new_row[f"{suffix}_pct_of_full"] = pct(row.get(mk), base.get(mk))
        out.append(new_row)
    return out


def split_results_by_pruner(
    output_dir: str,
    results: List[Dict[str, Any]],
    *,
    parallel: bool = False,
    indent: Optional[int] = 2,
    max_workers: Optional[int] = None,
) -> List[str]:
    """
    Write ``{output_dir}/by_pruner/<pruner_name>.json`` (one file per method) and
    ``index.json`` listing pruner names. Makes per-baseline results easy to open
    without searching the monolithic ``results.json``.

    When ``parallel=True``, per-method files are written via a process pool
    (one file per worker). ``indent`` controls pretty-printing of the per-method
    JSON (``None`` = compact, fastest).
    """
    by_dir = os.path.join(output_dir, "by_pruner")
    ensure_dir(by_dir)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        p = r.get("pruner")
        key = str(p) if p is not None else "unknown"
        groups[key].append(r)

    names_sorted = sorted(groups.keys())
    tasks: List[tuple] = []
    for name in names_sorted:
        safe = name.replace(os.sep, "_").replace("/", "_")
        path = os.path.join(by_dir, f"{safe}.json")
        tasks.append((path, groups[name]))

    if parallel and len(tasks) > 1:
        written = save_json_many_parallel(tasks, indent=indent, max_workers=max_workers)
    else:
        written = []
        for path, data in tasks:
            save_json_fast(path, data, indent=indent)
            written.append(path)

    index = {
        "pruners": names_sorted,
        "n_rows_per_pruner": {k: len(v) for k, v in groups.items()},
        "files": {k: f"{k.replace(os.sep, '_').replace('/', '_')}.json" for k in groups},
    }
    index_path = os.path.join(by_dir, "index.json")
    save_json_fast(index_path, index, indent=2)
    written.append(index_path)
    return written


def write_summary_csv(path: str, summary_rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(summary_rows)
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
