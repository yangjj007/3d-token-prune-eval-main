"""Baseline mesh token pruners: no pruning, random, uniform downsampling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from eval.baseline._common import target_keep_count
from eval.pruners import BasePruner, register_pruner


@register_pruner("no_pruning")
class NoPruningPruner(BasePruner):
    """Keep all input tokens (performance upper bound). ``keep_ratio`` is ignored."""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t = token_ids.detach().long().view(-1)
        n = int(t.numel())
        if n <= 0:
            raise ValueError("no_pruning received an empty token sequence")
        meta = {"indices": list(range(n)), "method": "no_pruning", "k": n}
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
        n = int(t.numel())
        if n <= 0:
            raise ValueError("random received an empty token sequence")
        k = target_keep_count(self.keep_ratio, n)
        g = torch.Generator(device=t.device)
        g.manual_seed(self.seed)
        perm = torch.randperm(n, generator=g, device=t.device)[:k]
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
    """Uniformly spaced indices over the input sequence."""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t = token_ids.detach().long().view(-1)
        n = int(t.numel())
        if n <= 0:
            raise ValueError("uniform received an empty token sequence")
        k = target_keep_count(self.keep_ratio, n)
        if k == n:
            idx = torch.arange(n, device=t.device, dtype=torch.long)
        elif k == 1:
            idx = torch.tensor([n // 2], device=t.device, dtype=torch.long)
        else:
            # Evenly spaced indices in [0, n - 1]
            step = (n - 1) / (k - 1)
            raw = [round(i * step) for i in range(k)]
            seen: set[int] = set()
            ordered: list[int] = []
            for x in raw:
                x = max(0, min(n - 1, int(x)))
                if x not in seen:
                    seen.add(x)
                    ordered.append(x)
            # Pad if duplicates removed
            p = 0
            while len(ordered) < k and p < n:
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
