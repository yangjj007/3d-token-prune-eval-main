"""Baseline mesh token pruners: no pruning, random, uniform downsampling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from eval.pruners import BasePruner, register_pruner

MESH_SEQ_LEN = 1024


def _target_keep_count(keep_ratio: float) -> int:
    k = max(1, int(round(MESH_SEQ_LEN * float(keep_ratio))))
    return min(k, MESH_SEQ_LEN)


@register_pruner("no_pruning")
class NoPruningPruner(BasePruner):
    """Keep all 1024 VQ tokens (performance upper bound). ``keep_ratio`` is ignored."""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t = token_ids.detach().long().view(-1)
        assert t.numel() == MESH_SEQ_LEN, f"Expected {MESH_SEQ_LEN} tokens, got {t.numel()}"
        meta = {"indices": list(range(MESH_SEQ_LEN)), "method": "no_pruning"}
        return t.clone(), meta


@register_pruner("random")
class RandomPruningPruner(BasePruner):
    """Randomly retain K tokens; indices sorted ascending to preserve sequence order."""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t = token_ids.detach().long().view(-1)
        assert t.numel() == MESH_SEQ_LEN
        k = _target_keep_count(self.keep_ratio)
        g = torch.Generator(device=t.device)
        g.manual_seed(self.seed)
        perm = torch.randperm(MESH_SEQ_LEN, generator=g, device=t.device)[:k]
        idx = perm.sort().values
        pruned = t[idx]
        meta = {
            "indices": idx.cpu().tolist(),
            "method": "random",
            "k": k,
        }
        return pruned, meta


@register_pruner("uniform")
class UniformDownsamplingPruner(BasePruner):
    """Uniformly spaced indices over [0, 1023]."""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t = token_ids.detach().long().view(-1)
        assert t.numel() == MESH_SEQ_LEN
        k = _target_keep_count(self.keep_ratio)
        if k == MESH_SEQ_LEN:
            idx = torch.arange(MESH_SEQ_LEN, device=t.device, dtype=torch.long)
        elif k == 1:
            idx = torch.tensor([MESH_SEQ_LEN // 2], device=t.device, dtype=torch.long)
        else:
            # Evenly spaced indices in [0, MESH_SEQ_LEN - 1]
            step = (MESH_SEQ_LEN - 1) / (k - 1)
            raw = [round(i * step) for i in range(k)]
            seen: set[int] = set()
            ordered: list[int] = []
            for x in raw:
                x = max(0, min(MESH_SEQ_LEN - 1, int(x)))
                if x not in seen:
                    seen.add(x)
                    ordered.append(x)
            # Pad if duplicates removed
            p = 0
            while len(ordered) < k and p < MESH_SEQ_LEN:
                if p not in seen:
                    ordered.append(p)
                    seen.add(p)
                p += 1
            ordered = sorted(ordered)[:k]
            idx = torch.tensor(ordered, device=t.device, dtype=torch.long)
        pruned = t[idx]
        meta = {
            "indices": idx.cpu().tolist(),
            "method": "uniform",
            "k": int(pruned.numel()),
        }
        return pruned, meta
