# -*- coding: utf-8 -*-
"""v4_perf3 golden reference for parity (DPP rank_scores path)."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from eval.baseline.otprune import greedy_select as _greedy_select_dpp
from eval.proposed.reconot import _greedy_fill


def _diversity_kernel_v4perf3(feats: Tensor, gamma: float) -> Tensor:
    g = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = g.t() @ g
    gram_sq = (g @ h) @ g.t()
    eye = torch.eye(feats.size(0), device=feats.device, dtype=feats.dtype)
    return eye + float(gamma) * gram_sq


def _select_dpp_on_subset_v4perf3(
    feats: Tensor,
    quality: Tensor,
    k_pick: int,
    *,
    gamma: float,
    epsilon: float,
) -> List[int]:
    n = feats.size(0)
    k_pick = min(k_pick, n)
    k_mat = _diversity_kernel_v4perf3(feats, gamma)
    q = quality.clamp_min(1e-12)
    l_kernel = q.unsqueeze(1) * k_mat * q.unsqueeze(0)
    selected_set = set(_greedy_fill(l_kernel, k_pick, epsilon))
    if len(selected_set) < k_pick:
        sel_mask = torch.ones(n, dtype=torch.bool, device=feats.device)
        if selected_set:
            sel_mask[list(selected_set)] = False
        remaining = torch.masked_select(
            torch.arange(n, device=feats.device), sel_mask
        )
        order = torch.argsort(quality[remaining], descending=True)
        for j in order.tolist():
            selected_set.add(int(remaining[j].item()))
            if len(selected_set) >= k_pick:
                break
    return sorted(selected_set)[:k_pick]


def dpp_rank_scores_v4perf3(
    feats: Tensor,
    q: Tensor,
    k_eff: int,
    *,
    gamma: float,
    epsilon: float,
) -> Tensor:
    n = feats.size(0)
    device = feats.device
    m = min(n, max(2 * k_eff, k_eff + 64, 384))
    cand = torch.topk(q, m).indices
    sub_feats = feats[cand]
    sub_q = q[cand]
    sub_k = min(int(cand.numel()), k_eff)
    sub_sel = _select_dpp_on_subset_v4perf3(
        sub_feats, sub_q, sub_k, gamma=gamma, epsilon=epsilon
    )
    global_sel = cand[sub_sel]

    rank_scores = torch.zeros(n, dtype=torch.float32, device=device)
    q_order = torch.argsort(q, descending=True)
    ranks = torch.arange(n, dtype=torch.float32, device=device)
    rank_scores[q_order] = (n - ranks) / max(n, 1)
    rank_scores[global_sel] = rank_scores[global_sel] + float(n)
    return rank_scores
