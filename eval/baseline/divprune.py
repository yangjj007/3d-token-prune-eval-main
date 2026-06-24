"""DivPrune: diversity greedy selection on cosine distance (mesh VQ adaptation)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings, require_vq_embeddings, target_keep_count
from eval.pruners import BasePruner, register_pruner


def pairwise_cosine_similarity(matrix: torch.Tensor) -> torch.Tensor:
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return norm_matrix @ norm_matrix.t()


def divprune_indices(
    visual_features: torch.Tensor,
    image_feature_length: int,
    cosine_matrix: torch.Tensor | None,
    keep_count: int,
) -> torch.Tensor:
    """Greedy max-min on distance = 1 - cos_sim; returns ``[K]`` long indices (DivPrune subset)."""
    if cosine_matrix is None:
        cosine_matrix = 1.0 - pairwise_cosine_similarity(visual_features)

    threshold_terms = max(1, min(int(keep_count), image_feature_length))
    device = visual_features.device
    s = torch.empty(threshold_terms, dtype=torch.long, device=device)
    selected_mask = torch.zeros(image_feature_length, dtype=torch.bool, device=device)
    for i in range(threshold_terms):
        if i == 0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(
                cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i, device=device))
            )

        if i == 0:
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
        else:
            scores = torch.min(m2, dim=0).values

        scores = scores.clone()
        scores[selected_mask] = -torch.inf
        phrase_to_add_idx = torch.argmax(scores)
        if bool(selected_mask[phrase_to_add_idx].item()) or not torch.isfinite(scores[phrase_to_add_idx]):
            remaining = torch.nonzero(~selected_mask, as_tuple=False).view(-1)
            if remaining.numel() == 0:
                break
            phrase_to_add_idx = remaining[0]
        s[i] = phrase_to_add_idx
        selected_mask[phrase_to_add_idx] = True

    return s[: int(selected_mask.sum().item())]


@register_pruner("divprune")
class DivPrunePruner(BasePruner):
    """Diversity pruning on codebook embeddings (LLaVA DivPrune, mesh segment only)."""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embed = require_vq_embeddings(kwargs.get("vq_embeddings"))
        t = token_ids.detach().long().view(-1)
        assert t.numel() == MESH_SEQ_LEN
        k = target_keep_count(self.keep_ratio)
        feats = gather_embeddings(embed, t)
        idx = divprune_indices(feats, MESH_SEQ_LEN, None, k)
        idx = torch.unique(idx, sorted=True)
        if idx.numel() < k:
            selected = torch.zeros(MESH_SEQ_LEN, dtype=torch.bool, device=idx.device)
            selected[idx] = True
            fill = torch.nonzero(~selected, as_tuple=False).view(-1)[: k - idx.numel()]
            idx = torch.cat([idx, fill]).sort().values
        else:
            idx = idx[:k].sort().values
        pruned = t[idx.cpu()]
        meta = {
            "method": "divprune",
            "indices": idx.cpu().tolist(),
            "k": int(pruned.numel()),
        }
        return pruned, meta
