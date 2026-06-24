#!/usr/bin/env python3
"""Synthetic smoke test for loco3d v21 quality-weighted DPP."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.baseline._common import MESH_SEQ_LEN, target_keep_count
from eval.pruners.baseline import RandomPruningPruner
from eval.proposed.loco3d import Loco3DPruner


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def main() -> int:
    torch.manual_seed(42)
    voxel = torch.zeros(64, 64, 64)
    voxel[24:40, 24:40, 24:40] = 1.0
    tokens = torch.randint(0, 400, (MESH_SEQ_LEN,))
    embed = nn.Embedding(512, 64)

    for kr in [0.75, 0.5, 0.25, 0.1]:
        k = target_keep_count(kr)
        p = Loco3DPruner(keep_ratio=kr, gamma=0.01, use_embedding=True)
        _, meta = p.prune(tokens, voxel, vq_embeddings=embed)
        ent = meta["diagnostics"]["entropy_stats"]
        assert meta["k"] == k, (meta["k"], k)
        assert math.isfinite(ent["mean"]), ent
        assert meta["selection_mode"] == "quality_weighted_dpp"
        rp = RandomPruningPruner(keep_ratio=kr, seed=42)
        _, rmeta = rp.prune(tokens, voxel, vq_embeddings=embed)
        j = jaccard(meta["indices"], rmeta["indices"])
        rand_exp = kr * kr / (2 * kr - kr * kr)
        uniq = len(set(tokens[meta["indices"]].tolist()))
        print(
            f"kr={kr} k={k} ent_mean={ent['mean']:.3f} "
            f"j_vs_rand={j:.3f} rand_exp~{rand_exp:.3f} delta={j - rand_exp:.3f} uniq={uniq}"
        )

    p2 = Loco3DPruner(keep_ratio=0.5, use_embedding=False)
    _, m2 = p2.prune(tokens, voxel)
    assert m2["selection_mode"] == "fps_quality_fallback"
    print("fps fallback ok", m2["k"])
    print("ALL SYNTHETIC CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
