"""
Token pruner registry and base class for ShapeLLM-Omni mesh token pruning experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

PRUNER_REGISTRY: Dict[str, type] = {}

# 所有剪枝模块：惰性加载，避免仅 import eval.proposed 时漏注册 proposed 子模块
_PRUNER_MODULES: tuple[str, ...] = (
    "eval.baseline",
    "eval.pruners.baseline",
    "eval.proposed.loco3d",
    "eval.proposed.octree_merge",
    "eval.proposed.reconot",
    "eval.proposed.runlength_curve",
)

_PROPOSED_PRUNERS: frozenset[str] = frozenset(
    {
        "loco3d",
        "loco3d_dpp",
        "loco3d_nonempty_dpp",
        "octree_merge",
        "reconot",
        "runlength_curve",
    }
)

_LOAD_ATTEMPTED = False
_LOAD_ERRORS: Dict[str, str] = {}


def ensure_pruners_loaded() -> None:
    """Import all pruner modules so ``@register_pruner`` decorators run."""
    global _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return
    _LOAD_ATTEMPTED = True
    import importlib

    for mod in _PRUNER_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            _LOAD_ERRORS[mod] = f"{type(exc).__name__}: {exc}"


def pruner_load_errors() -> Dict[str, str]:
    """Return import errors from the last ``ensure_pruners_loaded()`` call."""
    ensure_pruners_loaded()
    return dict(_LOAD_ERRORS)


def register_pruner(name: str):
    """Decorator to register a pruner class under ``name``."""

    def decorator(cls):
        if name in PRUNER_REGISTRY:
            raise ValueError(f"Pruner '{name}' is already registered.")
        PRUNER_REGISTRY[name] = cls
        return cls

    return decorator


def get_pruner_class(name: str) -> type:
    ensure_pruners_loaded()
    if name not in PRUNER_REGISTRY:
        available = ", ".join(sorted(PRUNER_REGISTRY.keys()))
        hint = _pruner_lookup_hint(name)
        raise KeyError(f"Unknown pruner '{name}'. Available: {available}{hint}")
    return PRUNER_REGISTRY[name]


def _pruner_lookup_hint(name: str) -> str:
    mod = f"eval.proposed.{name}"
    if mod in _LOAD_ERRORS:
        return f"\nImport failed for {mod}: {_LOAD_ERRORS[mod]}"
    if name in _PROPOSED_PRUNERS:
        return (
            f"\nMissing or broken eval/proposed/{name}.py — sync eval-main and run:\n"
            f'  python -c "import eval.proposed.{name}; from eval.pruners import PRUNER_REGISTRY; '
            f"print('{name}' in PRUNER_REGISTRY)\""
        )
    return ""


class BasePruner(ABC):
    """
    Args:
        keep_ratio: Target fraction of mesh tokens to keep in (0, 1].
        seed: RNG seed for stochastic pruners.
    """

    def __init__(self, keep_ratio: float = 1.0, seed: int = 42, **kwargs):
        self.keep_ratio = float(keep_ratio)
        self.seed = int(seed)
        self.extra = kwargs

    @abstractmethod
    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            token_ids: Long tensor of shape ``[1024]`` (VQ codebook indices).
            voxel_grid: Optional ``[64, 64, 64]`` occupancy (0/1) for spatial methods.
            **kwargs: e.g. ``vq_embeddings`` for codebook-aware methods.

        Returns:
            pruned_ids: Long tensor ``[K]`` with ``K <= 1024``.
            metadata: Extra diagnostics (indices kept, scores, etc.).
        """
        raise NotImplementedError
