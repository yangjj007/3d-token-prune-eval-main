# -*- coding: utf-8 -*-
"""Shared logging + deep JSONL dumps for proposed mesh pruners."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

import numpy as np
import torch

_LOGGERS_CONFIGURED: set[str] = set()


def _zeros_stats() -> Dict[str, float]:
    return {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "p05": 0.0,
        "p25": 0.0,
        "p50": 0.0,
        "p75": 0.0,
        "p90": 0.0,
        "p95": 0.0,
    }


def summarize_vec(x: Union[torch.Tensor, Sequence[float], None]) -> Dict[str, float]:
    """Summary stats for a 1D float vector (torch, list, or numpy-friendly sequence)."""
    if x is None:
        return _zeros_stats()
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return _zeros_stats()
        t = x.detach().float().cpu().view(-1)
        qs = torch.quantile(
            t,
            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.9, 0.95], dtype=torch.float32),
        )
        return {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "p05": float(qs[0].item()),
            "p25": float(qs[1].item()),
            "p50": float(qs[2].item()),
            "p75": float(qs[3].item()),
            "p90": float(qs[4].item()),
            "p95": float(qs[5].item()),
        }
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return _zeros_stats()
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return _zeros_stats()
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }
    # scalar or other iterable
    try:
        return summarize_vec(list(x))  # type: ignore[arg-type]
    except Exception:
        return _zeros_stats()


def histogram(
    xs: Iterable[float],
    bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")),
) -> Dict[str, int]:
    """Count values into half-open bins [bins[i], bins[i+1]). Last bin upper is +inf."""
    edges = list(bins)
    if not edges or len(edges) < 2:
        return {}
    counts: Dict[str, int] = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        hi_label = "+inf" if math.isinf(hi) else str(hi)
        counts[f"[{lo},{hi_label})"] = 0
    for v in xs:
        if not math.isfinite(v):
            continue
        placed = False
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if math.isinf(hi):
                if v >= lo:
                    hi_label = "+inf"
                    counts[f"[{lo},{hi_label})"] += 1
                    placed = True
                    break
            elif lo <= v < hi:
                counts[f"[{lo},{hi})"] += 1
                placed = True
                break
        if not placed and not math.isinf(edges[-1]) and v >= edges[-1]:
            counts[f"[{edges[-2]},{edges[-1]})"] += 1
    return counts


def should_deep_dump(sample_idx: int | None) -> bool:
    """
    SHAPELLM_EVAL_LOG_DEEP_EVERY: default 20.
    0 = never, negative = every sample, positive = sample_idx % n == 0.
    """
    if sample_idx is None:
        return False
    try:
        si = int(sample_idx)
    except (TypeError, ValueError):
        return False
    try:
        every = int(os.environ.get("SHAPELLM_EVAL_LOG_DEEP_EVERY", "20"))
    except ValueError:
        every = 20
    if every == 0:
        return False
    if every < 0:
        return True
    return si % every == 0


def _log_dir() -> Path:
    root = os.environ.get("SHAPELLM_EVAL_LOG_DIR", "logs")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_pruner_logger(method: str) -> logging.Logger:
    """File + stdout logger for one pruner name; idempotent."""
    name = f"eval.proposed.{method}"
    logger = logging.getLogger(name)
    if method in _LOGGERS_CONFIGURED:
        return logger
    level_name = os.environ.get("SHAPELLM_EVAL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    log_root = _log_dir()
    fh = logging.FileHandler(log_root / f"{method}.log", mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    _LOGGERS_CONFIGURED.add(method)
    return logger


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        x = obj.detach().float().cpu().view(-1).tolist()
        return [round(float(v), 6) for v in x]
    if isinstance(obj, np.ndarray):
        return [round(float(v), 6) for v in obj.astype(np.float64).flatten().tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_deep_dump(method: str, record: dict[str, Any]) -> None:
    """Append one JSON line to ``{log_dir}/{method}.deep.jsonl``."""
    path = _log_dir() / f"{method}.deep.jsonl"
    line = json.dumps(record, ensure_ascii=False, default=_json_default)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
