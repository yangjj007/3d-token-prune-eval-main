# -*- coding: utf-8 -*-
"""Compute composite eval_score from eval run summary.csv (autoresearch-rl integration)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _float_or_none(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_summary_rows(summary_path: Path, pruner: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("pruner") != pruner:
                continue
            kr = _float_or_none(row.get("keep_ratio", ""))
            if kr is None:
                continue
            rows.append(
                {
                    "keep_ratio": kr,
                    "rouge_l_mean": _float_or_none(row.get("rouge_l_mean", "")) or 0.0,
                    "bleu_4_mean": _float_or_none(row.get("bleu_4_mean", "")) or 0.0,
                    "bleu_1_mean": _float_or_none(row.get("bleu_1_mean", "")) or 0.0,
                    "sentence_bert_mean": _float_or_none(row.get("sentence_bert_mean", "")) or 0.0,
                    "simcse_mean": _float_or_none(row.get("simcse_mean", "")) or 0.0,
                    "generation_time_sec_mean": _float_or_none(
                        row.get("generation_time_sec_mean", "")
                    )
                    or 0.0,
                    "total_tflops_mean": _float_or_none(row.get("total_tflops_mean", "")) or 0.0,
                }
            )
    return rows


def row_composite(
    row: Dict[str, Any],
    *,
    w_rl: float = 0.5,
    w_b4: float = 0.3,
    w_b1: float = 0.2,
) -> float:
    return (
        w_rl * float(row["rouge_l_mean"])
        + w_b4 * float(row["bleu_4_mean"])
        + w_b1 * float(row["bleu_1_mean"])
    )


def compute_eval_score(
    rows: List[Dict[str, Any]],
    *,
    w_rl_at_03: float = 1.0,
    kr_low_focus: float = 0.3,
) -> Tuple[float, Dict[str, float]]:
    """Mean composite over keep_ratio rows; optional extra weight at low keep_ratio."""
    if not rows:
        return 0.0, {}

    weights: List[float] = []
    composites: List[float] = []
    per_kr: Dict[str, float] = {}

    for row in rows:
        c = row_composite(row)
        kr = float(row["keep_ratio"])
        w = w_rl_at_03 if abs(kr - kr_low_focus) < 1e-6 else 1.0
        weights.append(w)
        composites.append(c)
        per_kr[f"score_kr_{kr:g}"] = c

    total_w = sum(weights)
    score = sum(c * w for c, w in zip(composites, weights)) / max(total_w, 1e-9)

    agg_rl = sum(r["rouge_l_mean"] for r in rows) / len(rows)
    agg_b4 = sum(r["bleu_4_mean"] for r in rows) / len(rows)
    agg_b1 = sum(r["bleu_1_mean"] for r in rows) / len(rows)
    agg_sbert = sum(r.get("sentence_bert_mean", 0.0) for r in rows) / len(rows)
    agg_simcse = sum(r.get("simcse_mean", 0.0) for r in rows) / len(rows)

    extras = {
        "rouge_l_mean": agg_rl,
        "bleu_4_mean": agg_b4,
        "bleu_1_mean": agg_b1,
        "sentence_bert_mean": agg_sbert,
        "simcse_mean": agg_simcse,
        **per_kr,
    }
    return score, extras


def score_from_summary_file(
    summary_path: Path,
    pruner: str,
    *,
    w_rl_at_03: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    rows = load_summary_rows(summary_path, pruner)
    return compute_eval_score(rows, w_rl_at_03=w_rl_at_03)
