"""
FastV mesh proxy: no LLM attention; use a pseudo-query on codebook embeddings + top-k.

Aligns with FastV's ``topk`` on an importance row, using ``dot(emb_i, query)`` as scores.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from eval.baseline._common import gather_embeddings, require_vq_embeddings, target_keep_count
from eval.pruners import BasePruner, register_pruner


@register_pruner("fastv_mesh")
class FastVMeshPruner(BasePruner):
    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embed = require_vq_embeddings(kwargs.get("vq_embeddings"))
        query_mode = str(self.extra.get("query_mode", "mean"))

        t = token_ids.detach().long().view(-1)
        if t.numel() <= 0:
            raise ValueError("fastv_mesh received an empty token sequence")
        k = target_keep_count(self.keep_ratio, int(t.numel()))
        feats = gather_embeddings(embed, t)

        if query_mode == "last":
            q = feats[-1]
        else:
            q = feats.mean(dim=0)

        d = float(feats.shape[-1])
        scores = (feats * q.unsqueeze(0)).sum(dim=-1) / (d**0.5)
        top_rel = scores.topk(k, dim=-1).indices
        idx = top_rel.sort().values
        pruned = t[idx.cpu()]
        meta = {
            "method": "fastv_mesh",
            "indices": idx.cpu().tolist(),
            "k": int(pruned.numel()),
            "query_mode": query_mode,
        }
        return pruned, meta
