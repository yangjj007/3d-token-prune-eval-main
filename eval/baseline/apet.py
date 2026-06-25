"""ApET-style: FPS basis + linear reconstruction error ranking (mesh VQ)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import torch

from eval.baseline._common import gather_embeddings, require_vq_embeddings, target_keep_count
from eval.pruners import BasePruner, register_pruner


def fps(x: torch.Tensor, k: int) -> torch.LongTensor:
    """x: [B, N, D] -> centroids [B, k] indices."""
    B, N, D = x.shape
    centroids = torch.zeros(B, k, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), 1e10, device=x.device)
    farthest = torch.randint(0, N, (B,), device=x.device)
    for i in range(k):
        centroids[:, i] = farthest
        centroid = x[torch.arange(B, device=x.device), farthest].unsqueeze(1)
        dist_cur = ((x - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, dist_cur)
        farthest = dist.max(dim=1)[1]
    return centroids


def linear_reconstruct_with_basis(
    tokens: torch.Tensor, basis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """tokens, basis: [B, N, D] and [B, K, D] -> recon, errors [B, N]."""
    B, N, D = tokens.shape
    _, K, _ = basis.shape
    tokens_f = tokens.float()
    basis_f = basis.float()
    g = torch.matmul(basis_f, basis_f.transpose(-1, -2))
    g.diagonal(dim1=-2, dim2=-1).add_(1e-5)
    b_rhs = torch.matmul(basis_f, tokens_f.transpose(-1, -2))
    w = torch.linalg.solve(g, b_rhs).transpose(-1, -2)
    recon = torch.matmul(w, basis_f)
    errors = torch.norm(tokens_f - recon, dim=-1)
    return recon, errors


@register_pruner("apet")
class ApETPruner(BasePruner):
    """
    Keep tokens with highest linear reconstruction error w.r.t. FPS basis.
    ``basis_token_num`` and ``selection_method`` come from JSON (extra kwargs).
    """

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embed = require_vq_embeddings(kwargs.get("vq_embeddings"))
        t = token_ids.detach().long().view(-1)
        if t.numel() <= 0:
            raise ValueError("apet received an empty token sequence")
        k_keep = target_keep_count(self.keep_ratio, int(t.numel()))
        basis_token_num = int(self.extra.get("basis_token_num", max(1, min(32, k_keep // 4))))
        basis_token_num = max(1, min(basis_token_num, k_keep))
        selection_method: Literal["fps", "random"] = self.extra.get("selection_method", "fps")
        if selection_method not in ("fps", "random"):
            selection_method = "fps"

        x = gather_embeddings(embed, t).unsqueeze(0)  # [1, N, D]
        B, N, D = x.shape
        Kb = basis_token_num

        if selection_method == "fps":
            fps_idx = fps(x, Kb)
            idx_exp = fps_idx.unsqueeze(-1).expand(-1, -1, D)
            seed_features = x.gather(1, idx_exp)
        else:
            g = torch.Generator(device=x.device)
            g.manual_seed(self.seed)
            rand_idx = torch.randint(0, N, (B, Kb), device=x.device, generator=g)
            seed_features = x.gather(1, rand_idx.unsqueeze(-1).expand(-1, -1, D))

        _, errors = linear_reconstruct_with_basis(x, seed_features)
        _, top_idx = torch.topk(errors, k=k_keep, dim=1, largest=True)
        idx = top_idx[0].sort().values
        pruned = t[idx.cpu()]
        meta = {
            "method": "apet",
            "indices": idx.cpu().tolist(),
            "k": int(pruned.numel()),
            "basis_token_num": Kb,
            "selection_method": selection_method,
        }
        return pruned, meta
