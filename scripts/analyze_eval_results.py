#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit merged ShapeLLM pruning eval results for comparability issues."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.utils import aggregate_summary, compute_pct_of_full, load_json  # noqa: E402

METRIC_KEYS = (
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "rouge_l",
    "sentence_bert",
    "simcse",
    "generation_time_sec",
    "num_input_tokens",
    "num_output_tokens",
    "num_tokens_pruned",
    "total_tflops",
)


def _row_key(row: Dict[str, Any]) -> Tuple[str, str, float]:
    return (
        str(row.get("file_identifier", "")),
        str(row.get("pruner", "")),
        round(float(row.get("keep_ratio", 0.0)), 12),
    )


def _group_key(row: Dict[str, Any]) -> Tuple[str, float]:
    return (str(row.get("pruner", "")), round(float(row.get("keep_ratio", 0.0)), 12))


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _load_summary_counts(path: Path | None) -> Dict[Tuple[str, float], int]:
    if path is None or not path.is_file():
        return {}
    out: Dict[Tuple[str, float], int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["pruner"], round(float(row["keep_ratio"]), 12))] = int(float(row["n_samples"]))
    return out


def _success_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if isinstance(r, dict) and "generated_caption" in r]


def _failure_reason(row: Dict[str, Any]) -> str:
    return str(row.get("error_type") or row.get("skip_reason") or row.get("error") or "unknown")


def _build_group_audit(rows: Sequence[Dict[str, Any]], summary_counts: Dict[Tuple[str, float], int]) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    failures: Dict[Tuple[str, float], Counter] = defaultdict(Counter)
    for row in rows:
        key = _group_key(row)
        if "generated_caption" in row:
            by_group[key].append(row)
        else:
            failures[key][_failure_reason(row)] += 1

    out: List[Dict[str, Any]] = []
    for key in sorted(set(by_group) | set(failures), key=lambda x: (x[0], x[1])):
        items = by_group.get(key, [])
        file_ids = {str(r.get("file_identifier", "")) for r in items}
        record: Dict[str, Any] = {
            "pruner": key[0],
            "keep_ratio": key[1],
            "ok_rows": len(items),
            "unique_files": len(file_ids),
            "failed_rows": int(sum(failures.get(key, Counter()).values())),
            "failure_reasons": dict(failures.get(key, Counter())),
        }
        if key in summary_counts:
            record["summary_csv_n_samples"] = summary_counts[key]
            record["summary_count_delta"] = len(items) - summary_counts[key]
        for metric in METRIC_KEYS:
            value = _mean(items, metric)
            if value is not None:
                record[f"{metric}_mean"] = value
        out.append(record)
    return out


def _duplicate_audit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(_row_key(r) for r in rows if r.get("file_identifier") and r.get("pruner") is not None)
    duplicate_keys = {k: v for k, v in counts.items() if v > 1}
    examples = [
        {"file_identifier": k[0], "pruner": k[1], "keep_ratio": k[2], "count": v}
        for k, v in list(duplicate_keys.items())[:20]
    ]
    return {
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_row_excess": int(sum(v - 1 for v in duplicate_keys.values())),
        "examples": examples,
    }


def _common_sample_audit(ok: Sequence[Dict[str, Any]], pruners: Sequence[str] | None) -> Dict[str, Any]:
    by_group: Dict[Tuple[str, float], set[str]] = defaultdict(set)
    for row in ok:
        if pruners and row.get("pruner") not in pruners:
            continue
        by_group[_group_key(row)].add(str(row.get("file_identifier", "")))
    if not by_group:
        return {"common_file_count": 0, "groups": []}

    common = set.intersection(*by_group.values()) if by_group else set()
    filtered = [r for r in ok if str(r.get("file_identifier", "")) in common]
    if pruners:
        filtered = [r for r in filtered if r.get("pruner") in pruners]
    summary = compute_pct_of_full(aggregate_summary(filtered))
    return {
        "common_file_count": len(common),
        "groups": [
            {"pruner": p, "keep_ratio": kr, "file_count": len(files)}
            for (p, kr), files in sorted(by_group.items(), key=lambda x: (x[0][0], x[0][1]))
        ],
        "summary_on_common_files": summary,
    }


def _no_pruning_vs_random(ok: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = { _row_key(r): r for r in ok }
    no_rows = {
        str(r.get("file_identifier")): r
        for r in ok
        if r.get("pruner") == "no_pruning" and abs(float(r.get("keep_ratio", 0.0)) - 1.0) < 1e-9
    }
    rand_rows = {
        str(r.get("file_identifier")): r
        for r in ok
        if r.get("pruner") == "random" and abs(float(r.get("keep_ratio", 0.0)) - 1.0) < 1e-9
    }
    del by_key
    common = sorted(set(no_rows) & set(rand_rows))
    diffs: Dict[str, float | None] = {}
    for metric in METRIC_KEYS:
        vals = []
        for fid in common:
            a = no_rows[fid].get(metric)
            b = rand_rows[fid].get(metric)
            if a is not None and b is not None:
                vals.append(float(b) - float(a))
        diffs[f"random_minus_no_pruning_{metric}_mean"] = float(sum(vals) / len(vals)) if vals else None

    caption_equal = 0
    for fid in common:
        if str(no_rows[fid].get("generated_caption", "")) == str(rand_rows[fid].get("generated_caption", "")):
            caption_equal += 1
    return {
        "common_file_count": len(common),
        "caption_equal_count": caption_equal,
        "caption_equal_fraction": caption_equal / len(common) if common else None,
        **diffs,
    }


def _print_group_table(group_rows: Sequence[Dict[str, Any]]) -> None:
    print("\nGROUP AUDIT")
    print(
        "pruner,keep_ratio,ok_rows,unique_files,failed_rows,"
        "bleu_1_mean,rouge_l_mean,sentence_bert_mean,simcse_mean"
    )
    for row in group_rows:
        print(
            f"{row['pruner']},{row['keep_ratio']},"
            f"{row['ok_rows']},{row['unique_files']},{row['failed_rows']},"
            f"{row.get('bleu_1_mean', '')},{row.get('rouge_l_mean', '')},"
            f"{row.get('sentence_bert_mean', '')},{row.get('simcse_mean', '')}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ShapeLLM merged eval results.")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=REPO_ROOT.parent / "output" / "eval_results_4gpu_fp16" / "merged" / "results.json",
        help="Merged results.json path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT.parent / "output" / "eval_results_4gpu_fp16" / "merged" / "summary.csv",
        help="Optional summary.csv path used to cross-check n_samples.",
    )
    parser.add_argument(
        "--common-pruners",
        type=str,
        default="",
        help="Comma-separated pruners for common-sample recomputation. Default: all groups.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON audit output path.")
    args = parser.parse_args(argv)

    if not args.results_json.is_file():
        raise SystemExit(f"Missing results json: {args.results_json}")

    rows = load_json(str(args.results_json))
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list in {args.results_json}, got {type(rows).__name__}")

    ok = _success_rows(rows)
    summary_counts = _load_summary_counts(args.summary_csv)
    common_pruners = [p.strip() for p in args.common_pruners.split(",") if p.strip()] or None

    audit = {
        "results_json": str(args.results_json),
        "summary_csv": str(args.summary_csv) if args.summary_csv else None,
        "total_rows": len(rows),
        "ok_rows": len(ok),
        "failed_rows": len(rows) - len(ok),
        "duplicates": _duplicate_audit(rows),
        "groups": _build_group_audit(rows, summary_counts),
        "common_samples": _common_sample_audit(ok, common_pruners),
        "no_pruning_vs_random_1p0": _no_pruning_vs_random(ok),
    }

    print(f"rows total={audit['total_rows']} ok={audit['ok_rows']} failed={audit['failed_rows']}")
    dup = audit["duplicates"]
    print(
        "duplicates "
        f"keys={dup['duplicate_key_count']} excess_rows={dup['duplicate_row_excess']}"
    )
    nvr = audit["no_pruning_vs_random_1p0"]
    print(
        "no_pruning vs random@1.0 "
        f"common={nvr['common_file_count']} caption_equal_fraction={nvr['caption_equal_fraction']}"
    )
    print(f"common sample count={audit['common_samples']['common_file_count']}")
    _print_group_table(audit["groups"])

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
