"""Tier-1 proposed mesh token pruners (see ../../../docs/research/ShapeLLM-Omni-Token-Pruning-Proposals.md)."""

from __future__ import annotations

# loco3d 必须最先加载（RL campaign 默认可变文件）；子模块请保持显式 import。
from eval.proposed import loco3d  # noqa: F401
from eval.proposed import loco3d_dpp  # noqa: F401
from eval.proposed import loco3d_nonempty_dpp  # noqa: F401
from eval.proposed import octree_merge  # noqa: F401
from eval.proposed import reconot  # noqa: F401
from eval.proposed import runlength_curve  # noqa: F401

__all__ = [
    "loco3d",
    "loco3d_dpp",
    "loco3d_nonempty_dpp",
    "octree_merge",
    "reconot",
    "runlength_curve",
]
