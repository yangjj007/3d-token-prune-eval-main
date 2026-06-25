"""ToMe-style bipartite token merging on VQ embeddings, then nearest-codebook quantization."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import torch

from eval.baseline._common import gather_embeddings, nearest_codebook_ids, require_vq_embeddings, target_keep_count
from eval.pruners import BasePruner, register_pruner


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src_m = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_m, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    return merge, merge


@register_pruner("tome")
class ToMePruner(BasePruner):
    """
    Iterative ToMe merging until token count <= K, then map rows to nearest codebook ids.
    """

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embed = require_vq_embeddings(kwargs.get("vq_embeddings"))
        merge_mode = str(self.extra.get("merge_mode", "mean"))
        max_rounds = int(self.extra.get("max_rounds", 256))

        t = token_ids.detach().long().view(-1)
        if t.numel() <= 0:
            raise ValueError("tome received an empty token sequence")
        k_target = target_keep_count(self.keep_ratio, int(t.numel()))

        emb = gather_embeddings(embed, t).unsqueeze(0)
        rounds = 0
        while emb.shape[1] > k_target and rounds < max_rounds:
            n = emb.shape[1]
            need_remove = n - k_target
            r = min(n // 2, max(1, need_remove))
            merge_fn, _ = bipartite_soft_matching(emb, r, class_token=False, distill_token=False)
            if merge_fn is do_nothing:
                break
            emb = merge_fn(emb, mode=merge_mode)
            rounds += 1

        merged = emb.squeeze(0)
        # If still above k (edge case), keep first k by magnitude or truncate
        if merged.shape[0] > k_target:
            merged = merged[:k_target]
        ids_out = nearest_codebook_ids(embed, merged).cpu()
        meta = {
            "method": "tome",
            "k": int(ids_out.numel()),
            "merge_rounds": rounds,
            "merge_mode": merge_mode,
        }
        return ids_out, meta
