"""OTPrune: DPP-style greedy on kernel K = I + γ G G^T (mesh VQ)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from eval.baseline._common import gather_embeddings, require_vq_embeddings, target_keep_count
from eval.pruners import BasePruner, register_pruner


def greedy_select(kernel_matrix: Tensor, max_length: int, epsilon: float = 1e-10) -> List[int]:
    device = kernel_matrix.device
    dtype = kernel_matrix.dtype
    item_size = kernel_matrix.size(0)
    cis = torch.zeros((max_length, item_size), device=device, dtype=dtype)
    di2s = torch.clone(torch.diag(kernel_matrix))
    selected_items: List[int] = []
    selected_item = torch.argmax(di2s).item()
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        if k > 0:
            correction = (ci_optimal.unsqueeze(0) @ cis[:k, :]).squeeze(0)
        else:
            correction = 0.0
        eis = (elements - correction) / di_optimal.clamp_min(1e-12)
        cis[k, :] = eis
        di2s = di2s - eis.square()
        selected_item = torch.argmax(di2s).item()
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def otprune_select(
    visual_tokens: Tensor,
    threshold_ratio: float,
    gamma: float = 0.01,
    epsilon: float = 1e-10,
) -> Tensor:
    if visual_tokens.dim() != 2:
        raise ValueError(f"otprune_select expects [N, D], got {tuple(visual_tokens.shape)}")
    device = visual_tokens.device
    dtype = visual_tokens.dtype
    num_tokens, _ = visual_tokens.shape
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    visual_tokens_norm = visual_tokens / visual_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gram = visual_tokens_norm @ visual_tokens_norm.t()
    eye = torch.eye(num_tokens, device=device, dtype=dtype)
    kernel_matrix = eye + gamma * (gram @ gram.t())

    max_length = int(round(threshold_ratio * num_tokens))
    max_length = max(1, min(max_length, num_tokens))

    selected_items = greedy_select(kernel_matrix=kernel_matrix, max_length=max_length, epsilon=epsilon)
    selected_set = set(selected_items)
    km = kernel_matrix.clone()
    while len(selected_set) < max_length:
        for idx in selected_set:
            km[idx, :] = 0
            km[:, idx] = 0
        remain = max_length - len(selected_set)
        if remain <= 0:
            break
        new_items = greedy_select(kernel_matrix=km, max_length=remain, epsilon=epsilon)
        if not new_items:
            break
        selected_set.update(new_items)

    out = torch.tensor(sorted(selected_set), dtype=torch.long, device=device)
    return out


@register_pruner("otprune")
class OTPrunePruner(BasePruner):
    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embed = require_vq_embeddings(kwargs.get("vq_embeddings"))
        gamma = float(self.extra.get("gamma", 0.01))
        epsilon = float(self.extra.get("epsilon", 1e-10))

        t = token_ids.detach().long().view(-1)
        if t.numel() <= 0:
            raise ValueError("otprune received an empty token sequence")
        feats = gather_embeddings(embed, t)
        idx = otprune_select(feats, self.keep_ratio, gamma=gamma, epsilon=epsilon)
        idx = idx.sort().values
        pruned = t[idx.cpu()]
        meta = {
            "method": "otprune",
            "indices": idx.cpu().tolist(),
            "k": int(pruned.numel()),
            "gamma": gamma,
        }
        return pruned, meta
