"""Shared diagnostics helpers for proposed mesh pruners (V7)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from eval.proposed._spatial import _is_grid_boundary, flat_index_to_coord


def boundary_interior_counts(
    flat_indices: List[int],
    surface_mask: Optional[torch.Tensor] = None,
) -> Tuple[int, int]:
    """
    Return (boundary_like, interior_like) for given flat indices.

    If ``surface_mask`` is provided (bool ``[1024]``), boundary = true voxel surface;
    otherwise legacy: latent grid outer shell via ``_is_grid_boundary``.
    """
    if surface_mask is not None:
        b = sum(1 for fi in flat_indices if bool(surface_mask[int(fi)].item()))
        return b, len(flat_indices) - b
    b = i = 0
    for fi in flat_indices:
        x, y, z = flat_index_to_coord(fi)
        if _is_grid_boundary(x, y, z):
            b += 1
        else:
            i += 1
    return b, i


def tensor_score_stats(t: torch.Tensor) -> Dict[str, float]:
    """Summary stats for a 1D float tensor (on any device)."""
    if t.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    x = t.detach().float().cpu().view(-1)
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "median": float(q[1].item()),
        "p25": float(q[0].item()),
        "p75": float(q[2].item()),
    }
