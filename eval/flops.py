"""
Rough FLOP / TFLOP estimates for mesh pruning + decoder-only LLM inference.

These are **analytic proxies** for cross-method comparison (not hardware-accurate).
"""

from __future__ import annotations

import math
from typing import Any, Dict

MESH_SEQ_LEN = 1024


def get_model_config(model_id: str) -> Dict[str, int]:
    """
    Return decoder-only-ish hyperparameters for FLOP scaling.
    Qwen2.5-VL-7B text tower ≈ Qwen2.5-7B; vision stack ignored here (mesh tokens dominate text).
    """
    s = model_id.lower()
    if "72b" in s or "70b" in s:
        return {"d_model": 8192, "n_layers": 80, "d_ff": 28672, "n_heads": 64, "n_kv_heads": 8}
    if "32b" in s:
        return {"d_model": 5120, "n_layers": 64, "d_ff": 27648, "n_heads": 40, "n_kv_heads": 8}
    if "14b" in s:
        return {"d_model": 5120, "n_layers": 48, "d_ff": 13824, "n_heads": 40, "n_kv_heads": 8}
    if "2b" in s:
        # Rough Qwen-VL 2B-class proxy; used for EVA01 reporting only.
        return {"d_model": 2048, "n_layers": 28, "d_ff": 11008, "n_heads": 16, "n_kv_heads": 8}
    if "3b" in s:
        return {"d_model": 2048, "n_layers": 36, "d_ff": 11008, "n_heads": 16, "n_kv_heads": 8}
    # default: Qwen2.5-7B class
    return {"d_model": 3584, "n_layers": 28, "d_ff": 18944, "n_heads": 28, "n_kv_heads": 4}


def _decoder_layer_flops_prefill(
    seq_len: int,
    d_model: int,
    d_ff: int,
    n_heads: int,
    n_kv_heads: int,
) -> float:
    """Approximate FLOPs for one transformer layer, prefill (batch=1)."""
    # QKV projections: 2 * seq * d * (d + 2 * d_kv_ratio * d) with kv heads
    kv_ratio = n_kv_heads / max(1, n_heads)
    qkv = 2.0 * seq_len * d_model * (d_model + 2.0 * kv_ratio * d_model)
    # Attention scores + weighted sum: 2 * seq^2 * d each (approx)
    attn = 4.0 * (seq_len**2) * d_model
    # Out proj
    out_p = 2.0 * seq_len * d_model * d_model
    # SwiGLU FFN: 3 matmuls of seq x d x d_ff scale
    ffn = 3.0 * 2.0 * seq_len * d_model * d_ff
    return qkv + attn + out_p + ffn


def _decoder_layer_flops_decode_step(
    ctx_len: int,
    d_model: int,
    d_ff: int,
    n_heads: int,
    n_kv_heads: int,
) -> float:
    """One autoregressive step (KV cache length = ctx_len)."""
    kv_ratio = n_kv_heads / max(1, n_heads)
    qkv = 2.0 * d_model * (d_model + 2.0 * kv_ratio * d_model)
    attn = 4.0 * ctx_len * d_model
    out_p = 2.0 * d_model * d_model
    ffn = 3.0 * 2.0 * d_model * d_ff
    return qkv + attn + out_p + ffn


def estimate_llm_tflops(num_input_tokens: int, num_output_tokens: int, model_id: str) -> Dict[str, float]:
    """
    Prefill on ``num_input_tokens`` + decode for ``num_output_tokens`` new tokens.
    Returns TFLOPs (1e12 FLOPs).
    """
    cfg = get_model_config(model_id)
    d_model = cfg["d_model"]
    n_layers = cfg["n_layers"]
    d_ff = cfg["d_ff"]
    n_heads = cfg["n_heads"]
    n_kv_heads = cfg["n_kv_heads"]

    prefill_layer = _decoder_layer_flops_prefill(num_input_tokens, d_model, d_ff, n_heads, n_kv_heads)
    prefill = n_layers * prefill_layer

    decode = 0.0
    for t in range(max(0, num_output_tokens)):
        ctx = num_input_tokens + t
        decode += n_layers * _decoder_layer_flops_decode_step(ctx, d_model, d_ff, n_heads, n_kv_heads)

    total = prefill + decode
    return {
        "llm_prefill_tflops": prefill / 1e12,
        "llm_decode_tflops": decode / 1e12,
        "llm_total_tflops": total / 1e12,
    }


def _estimate_pruner_flops_heuristic(
    pruner_name: str,
    meta: Dict[str, Any],
    *,
    embed_dim: int,
    codebook_size: int,
) -> float:
    """Return estimated pruner FLOPs (not TFLOPs)."""
    n = int(
        meta.get("num_tokens_original")
        or meta.get("num_eva_patches_original")
        or MESH_SEQ_LEN
    )
    d = max(1, embed_dim)
    v = max(1, codebook_size)
    k = int(meta.get("k", n))

    # Embedding gather + pairwise distances baseline
    base = 2.0 * n * d + 4.0 * n * d  # gather + L2-ish

    name = (pruner_name or "").lower()
    if name == "runlength_curve":
        runs = int(meta.get("num_runs_initial", n))
        curve_ops = 4.0 * n * d + 2.0 * n
        topk = float(n * math.log(max(n, 2)))
        return base + curve_ops + topk + 2.0 * runs * d
    if name == "octree_merge":
        blocks = 128
        e8 = blocks * 8 * d * 4
        per_flat = 6.0 * n * d + 2.0 * n * d
        return base + e8 + per_flat + 2.0 * k * d
    if name == "loco3d":
        # causal scan + optional 6n + nearest codebook per cell
        nn = 6.0 * n * d + float(n) * (2.0 * v * d)
        return base + nn + 4.0 * n * d
    if name in ("tome",):
        # iterative merge; use rounds if present
        rounds = int(meta.get("merge_rounds", max(1, n - k)))
        return base + float(rounds) * (4.0 * n * d + 2.0 * n * n * d)
    if name in ("divprune",):
        greedy = float(k) * (2.0 * n * d + n * n)
        return base + greedy
    if name in ("apet",):
        basis = int(meta.get("basis_token_num", 32))
        solve = float(basis**3) + 2.0 * n * basis * d
        return base + solve + 2.0 * n * d
    if name in ("otprune", "fastv_mesh"):
        return base + 4.0 * n * n * d
    # random / uniform / no_pruning
    return base + 2.0 * k


def enrich_pruner_metadata_flops(
    pruner_name: str,
    meta: Dict[str, Any],
    *,
    embed_dim: int,
    codebook_size: int,
) -> None:
    """Mutates ``meta`` in-place: sets ``diagnostics['pruner_tflops']`` (creates diagnostics dict)."""
    flops = _estimate_pruner_flops_heuristic(pruner_name, meta, embed_dim=embed_dim, codebook_size=codebook_size)
    diag = meta.get("diagnostics")
    if not isinstance(diag, dict):
        diag = {}
        meta["diagnostics"] = diag
    diag["pruner_tflops"] = float(flops / 1e12)
