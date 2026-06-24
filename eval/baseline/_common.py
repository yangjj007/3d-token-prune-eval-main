"""Shared helpers for mesh VQ token pruning baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

MESH_SEQ_LEN = 1024


def target_keep_count(keep_ratio: float, seq_len: int = MESH_SEQ_LEN) -> int:
    k = max(1, int(round(float(keep_ratio) * seq_len)))
    return min(k, seq_len)


def gather_embeddings(
    embed: nn.Embedding,
    token_ids: torch.Tensor,
    *,
    device: torch.device | str | None = "cpu",
) -> torch.Tensor:
    """
    Return ``[N, D]`` continuous embeddings for discrete indices.

    Defaults to CPU so pruning does not allocate large kernels on the VQVAE GPU
    (``cuda:1`` in dual-GPU eval), which otherwise leaks until OOM over long runs.
    Pass ``device=None`` to use ``embed.weight.device``.
    """
    t = token_ids.detach().long().view(-1)
    if device is None:
        dev = embed.weight.device
        return embed(t.to(dev))
    dev = torch.device(device)
    w = embed.weight.detach()
    if w.device != dev:
        w = w.to(dev)
    return torch.nn.functional.embedding(t.to(dev), w)


def nearest_codebook_ids(embed: nn.Embedding, features: torch.Tensor) -> torch.Tensor:
    """
    Map continuous rows ``[M, D]`` to nearest codebook indices.

    ``features`` and ``embed.weight`` use Euclidean distance in embedding space.
    """
    w = embed.weight  # [V, D]
    dev = w.device
    f = features.float().to(dev)
    # [M, V]
    d2 = torch.cdist(f, w.float(), p=2.0) ** 2
    return d2.argmin(dim=-1).long()


def require_vq_embeddings(vq_embeddings: Any) -> nn.Embedding:
    if vq_embeddings is None:
        raise ValueError("This pruner requires vq_embeddings (VQ codebook nn.Embedding).")
    if not isinstance(vq_embeddings, nn.Embedding):
        raise TypeError(f"vq_embeddings must be nn.Embedding, got {type(vq_embeddings)}")
    return vq_embeddings
