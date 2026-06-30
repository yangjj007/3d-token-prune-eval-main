#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小样本对比 loco3d、otprune、reconot（及 random 参照）的剪枝结果差异。

仅剪枝（默认）：只加载 VQVAE，速度快。loco3d v20 用邻接 token 熵；otprune 用 embedding；
reconot 用重构误差引导的 q 加权 DPP/OT（纯 embedding，无空间先验）；
脚本对三者都算集合/空间指标，并分别用熵与 embedding 邻域误差做 rank/独占区域分析。
可选 --with-vlm：额外跑 ShapeLLM 生成 caption，并记录 BLEU/ROUGE（显存占用大）。

用法（在 3d-token-prune-eval-main 目录）::

python scripts/compare_loco3d_otprune.py \
    --data-csv ../data/metadata.csv \
    --glb-dir ../data \
    --num-samples 5 \
    --keep-ratios 0.75,0.5,0.25,0.1 \
    --vqvae-device cuda:0 \
    --log-file ../output/logs/compare_loco3d_otprune.log

双卡时 VLM 用 cuda:0、VQVAE 用 cuda:1::

    python scripts/compare_loco3d_otprune.py ... --with-vlm --device cuda:0 --vqvae-device cuda:1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ["SHAPELLM_EVAL_LIGHT"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _register_pruners_for_compare() -> None:
    """
    显式加载各剪枝模块以执行 @register_pruner。

    勿只 ``import eval.proposed``：若服务器上 ``eval/proposed/__init__.py`` 未
    ``import loco3d``，registry 里会只有 octree_merge / runlength_curve，没有 loco3d。
    """
    import eval.baseline  # noqa: F401
    import eval.pruners.baseline  # noqa: F401
    import eval.proposed.loco3d  # noqa: F401 — 必须直接导入子模块
    import eval.proposed.reconot  # noqa: F401 — 必须直接导入子模块


_register_pruners_for_compare()

from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings, target_keep_count  # noqa: E402
from eval.config import load_pruner_extra_kwargs, resolve_repo_paths  # noqa: E402
from eval.config import EvalConfig  # noqa: E402
from eval.cuda_env import init_cuda_for_eval, resolve_torch_device  # noqa: E402
from eval.data_loader import iter_dataset, mesh_to_tokens  # noqa: E402
from eval.metrics import compute_text_metrics  # noqa: E402
from eval.pruners import PRUNER_REGISTRY, get_pruner_class  # noqa: E402
from eval.proposed._spatial import (  # noqa: E402
    GRID_X,
    GRID_Y,
    GRID_Z,
    _is_grid_boundary,
    all_coords_tensor,
    flat_index_to_coord,
    latent_surface_mask,
)
from eval.run_eval import load_llm, load_vqvae  # noqa: E402
from eval.utils import tokens_to_mesh_string  # noqa: E402

METHODS_DEFAULT = ("loco3d", "otprune", "reconot", "random")


def _require_pruners(names: Sequence[str]) -> None:
    missing = [n for n in names if n not in PRUNER_REGISTRY]
    if not missing:
        return
    avail = ", ".join(sorted(PRUNER_REGISTRY.keys()))
    raise SystemExit(
        f"剪枝器未注册: {', '.join(missing)}\n"
        f"当前 registry: {avail}\n"
        "请检查:\n"
        "  1) eval/proposed/loco3d.py / reconot.py 是否存在且含 @register_pruner\n"
        "  2) eval/proposed/__init__.py 是否包含对应 import\n"
        "  3) 在 eval-main 根目录运行: python scripts/compare_loco3d_otprune.py ...\n"
        f"若导入失败，可先执行: python -c \"import eval.proposed.reconot\""
    )
PAIR_FOCUS = (
    ("loco3d", "otprune"),
    ("loco3d", "reconot"),
    ("loco3d", "random"),
    ("otprune", "reconot"),
    ("otprune", "random"),
    ("reconot", "random"),
)


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _indices_to_mask(indices: Sequence[int], n: int = MESH_SEQ_LEN) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for i in indices:
        if 0 <= int(i) < n:
            m[int(i)] = True
    return m


def _mask_to_indices(mask: np.ndarray) -> List[int]:
    return np.nonzero(mask)[0].astype(int).tolist()


@dataclass
class IndexSetStats:
    indices: List[int]
    k: int
    mask: np.ndarray = field(repr=False)

    @classmethod
    def from_indices(cls, indices: Sequence[int]) -> "IndexSetStats":
        idx = sorted(int(i) for i in indices)
        m = _indices_to_mask(idx)
        return cls(indices=idx, k=len(idx), mask=m)

def _six_neighbor_embedding_importance_measures(
    emb: torch.Tensor, is_empty: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    基于 VQ embedding 的 6‑邻居关系，计算多个 token 重要性度量。

    返回字典，每个值都是 shape [1024] 的 numpy float64 数组。
    - local_dissim: 1 - cosine_sim(token, mean_of_neighbors)
    - neighbor_var: 邻居 embedding 各维度的方差均值
    """
    d = emb.shape[-1]
    emb_grid = emb.view(GRID_X, GRID_Y, GRID_Z, d)  # [X, Y, Z, D]
    # 初始化累加器
    sum_nb = torch.zeros_like(emb_grid)
    sum_sq_nb = torch.zeros_like(emb_grid)
    cnt_nb = torch.zeros(GRID_X, GRID_Y, GRID_Z, device=emb_grid.device, dtype=torch.float32)

    for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
        # 根据偏移量切片
        if dx == 1:
            sum_nb[:-1] += emb_grid[1:]
            sum_sq_nb[:-1] += emb_grid[1:] ** 2
            cnt_nb[:-1] += 1
        elif dx == -1:
            sum_nb[1:] += emb_grid[:-1]
            sum_sq_nb[1:] += emb_grid[:-1] ** 2
            cnt_nb[1:] += 1
        # y, z 方向类似处理
        if dy == 1:
            sum_nb[:, :-1] += emb_grid[:, 1:]
            sum_sq_nb[:, :-1] += emb_grid[:, 1:] ** 2
            cnt_nb[:, :-1] += 1
        elif dy == -1:
            sum_nb[:, 1:] += emb_grid[:, :-1]
            sum_sq_nb[:, 1:] += emb_grid[:, :-1] ** 2
            cnt_nb[:, 1:] += 1
        if dz == 1:
            sum_nb[:, :, :-1] += emb_grid[:, :, 1:]
            sum_sq_nb[:, :, :-1] += emb_grid[:, :, 1:] ** 2
            cnt_nb[:, :, :-1] += 1
        elif dz == -1:
            sum_nb[:, :, 1:] += emb_grid[:, :, :-1]
            sum_sq_nb[:, :, 1:] += emb_grid[:, :, :-1] ** 2
            cnt_nb[:, :, 1:] += 1

    # 防止除零
    cnt_nb = cnt_nb.clamp(min=1.0)
    mean_nb = sum_nb / cnt_nb.unsqueeze(-1)
    # 邻居方差
    var_nb = (sum_sq_nb / cnt_nb.unsqueeze(-1)) - (mean_nb ** 2)
    # 局部不相似度: 1 - cosine(token, mean_neighbor)
    emb_grid_norm = emb_grid / (emb_grid.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    mean_nb_norm = mean_nb / (mean_nb.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    cosine_sim = (emb_grid_norm * mean_nb_norm).sum(dim=-1)
    local_dissim = 1.0 - cosine_sim

    # 展平
    local_dissim_flat = local_dissim.reshape(-1).detach().cpu().numpy()
    # 邻居方差：取所有维度的均值作为总方差度量
    neighbor_var_flat = var_nb.mean(dim=-1).reshape(-1).detach().cpu().numpy()

    # 对 is_empty 位置置零（或保留，视需求而定）
    if is_empty.device != emb.device:
        is_empty = is_empty.to(emb.device)
    empty_mask = is_empty.cpu().numpy()
    local_dissim_flat[empty_mask] = 0.0
    neighbor_var_flat[empty_mask] = 0.0

    return {
        "local_dissim": local_dissim_flat.astype(np.float64),
        "neighbor_var": neighbor_var_flat.astype(np.float64),
    }

def _set_overlap_metrics(a: IndexSetStats, b: IndexSetStats) -> Dict[str, float]:
    inter = int(np.logical_and(a.mask, b.mask).sum())
    union = int(np.logical_or(a.mask, b.mask).sum())
    only_a = int(np.logical_and(a.mask, ~b.mask).sum())
    only_b = int(np.logical_and(b.mask, ~a.mask).sum())
    sym_diff = only_a + only_b
    jacc = inter / union if union > 0 else 0.0
    dice = 2 * inter / (a.k + b.k) if (a.k + b.k) > 0 else 0.0
    prec_a_in_b = inter / a.k if a.k > 0 else 0.0
    prec_b_in_a = inter / b.k if b.k > 0 else 0.0
    # 随机两集（各 k）期望 Jaccard ≈ k/(2N-k) 当独立均匀抽样；报告 k/N 作参照
    kr_a = a.k / MESH_SEQ_LEN
    kr_b = b.k / MESH_SEQ_LEN
    rand_expected_jacc = (kr_a * kr_b) / (kr_a + kr_b - kr_a * kr_b + 1e-12)
    return {
        "intersection": float(inter),
        "union": float(union),
        "only_a": float(only_a),
        "only_b": float(only_b),
        "symmetric_diff": float(sym_diff),
        "jaccard": float(jacc),
        "dice": float(dice),
        "recall_a_from_b": prec_a_in_b,
        "recall_b_from_a": prec_b_in_a,
        "overlap_coef": float(inter / min(a.k, b.k)) if min(a.k, b.k) > 0 else 0.0,
        "rand_indep_expected_jaccard": float(rand_expected_jacc),
        "jaccard_minus_rand_expected": float(jacc - rand_expected_jacc),
    }


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _spatial_metrics(indices: Sequence[int], voxel_grid: torch.Tensor | None) -> Dict[str, float]:
    coords = all_coords_tensor(device=torch.device("cpu")).numpy()
    idx = np.array(sorted(int(i) for i in indices), dtype=np.int64)
    if idx.size == 0:
        return {}
    c = coords[idx]
    centroid = c.mean(axis=0)
    spread = float(np.linalg.norm(c - centroid, axis=1).mean())
    spread_std = float(np.linalg.norm(c - centroid, axis=1).std())
    boundary_frac = float(
        sum(1 for i in idx if _is_grid_boundary(*flat_index_to_coord(int(i)))) / len(idx)
    )
    # 沿 flat 顺序的 run-length（连续段）
    runs = 1
    for j in range(1, len(idx)):
        if idx[j] != idx[j - 1] + 1:
            runs += 1
    mean_run = len(idx) / max(runs, 1)
    out: Dict[str, float] = {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "mean_dist_to_centroid": spread,
        "std_dist_to_centroid": spread_std,
        "grid_boundary_fraction": boundary_frac,
        "num_runs_along_flat_order": float(runs),
        "mean_run_length_flat": float(mean_run),
    }
    if voxel_grid is not None:
        is_surface, is_empty, is_filled, occ_flat = latent_surface_mask(voxel_grid.cpu())
        surf = is_surface.numpy()
        empty = is_empty.numpy()
        filled = is_filled.numpy()
        occ = occ_flat.numpy()
        m = _indices_to_mask(idx)
        out["kept_surface_fraction"] = float(surf[m].mean()) if m.any() else 0.0
        out["kept_empty_fraction"] = float(empty[m].mean()) if m.any() else 0.0
        out["kept_filled_fraction"] = float(filled[m].mean()) if m.any() else 0.0
        out["kept_mean_occupancy"] = float(occ[m].mean()) if m.any() else 0.0
        out["all_surface_fraction"] = float(surf.mean())
        out["surface_recall_vs_all_surface"] = (
            float(surf[m].sum() / max(surf.sum(), 1)) if surf.any() else 0.0
        )
        out["empty_recall_vs_all_empty"] = (
            float(empty[m].sum() / max(empty.sum(), 1)) if empty.any() else 0.0
        )
    return out

def _otprune_spatial_analysis(
    indices: Sequence[int],
    token_ids: torch.Tensor,
    voxel_grid: torch.Tensor,
    vq_emb: torch.nn.Embedding | None = None,
) -> Dict[str, float]:
    """
    对 otprune 保留的 token 进行详细的空间特征分析。
    返回多个统计量，用于发现其与 random 的差异。
    """
    if len(indices) == 0:
        return {}
    coords = all_coords_tensor(device=torch.device("cpu")).numpy()
    idx = np.array(sorted(int(i) for i in indices), dtype=np.int64)
    c_kept = coords[idx]                     # [K, 3]
    c_all = coords                           # [1024, 3]

    # 1. 最近邻距离分布（保留集内部和全体到保留集的距离）
    from scipy.spatial import cKDTree
    tree_all = cKDTree(c_all)
    tree_kept = cKDTree(c_kept)

    # 保留 token 之间的最近邻距离
    if len(c_kept) > 1:
        dist_kept_to_kept, _ = tree_kept.query(c_kept, k=2)  # 第一个是自己，取第二个
        nn_dist_kept = dist_kept_to_kept[:, 1]
    else:
        nn_dist_kept = np.array([0.0])

    # 全体 token 到保留集的最近距离
    dist_all_to_kept, _ = tree_kept.query(c_all, k=1)
    
    # 2. 基于网格的均匀性：将空间离散成 (4,4,4) 的小块，统计保留 token 在每个块中的数量分布
    grid_bins = 4  # 4x4x4 = 64 个小方块
    bins = [np.linspace(0, GRID_X-1, grid_bins+1),
            np.linspace(0, GRID_Y-1, grid_bins+1),
            np.linspace(0, GRID_Z-1, grid_bins+1)]
    # 全体 token 的直方图
    hist_all, _ = np.histogramdd(c_all, bins=bins)
    # 保留 token 的直方图
    hist_kept, _ = np.histogramdd(c_kept, bins=bins)
    # 若某块没有 token，可能导致除法问题，我们只考虑有 token 的块
    mask = hist_all > 0
    # 计算每块保留比例，然后统计这些比例的分布
    ratios = np.zeros_like(hist_all, dtype=np.float64)
    ratios[mask] = hist_kept[mask] / hist_all[mask]
    # 均匀覆盖度：保留比例的标准差（越小越均匀）
    uniformity_std = float(np.std(ratios[mask]))
    # 被保留 token 覆盖的块数占总非空块数的比例
    covered_blocks = float(np.sum(hist_kept > 0) / max(np.sum(mask), 1))

    # 3. 边界偏好：距离网格边界的最小距离
    def _dist_to_boundary(coord):
        x, y, z = coord
        return min(x, GRID_X-1-x, y, GRID_Y-1-y, z, GRID_Z-1-z)
    d2b_kept = np.array([_dist_to_boundary(c_kept[i]) for i in range(len(c_kept))])
    d2b_all = np.array([_dist_to_boundary(c_all[i]) for i in range(len(c_all))])

    # 4. 表面 / 内部 / 空 token 的保留偏好
    if voxel_grid is not None:
        is_surface, is_empty, is_filled, _ = latent_surface_mask(voxel_grid.cpu())
        surf = is_surface.numpy()
        empty = is_empty.numpy()
        filled = is_filled.numpy()
        kept_mask = _indices_to_mask(indices)
        surf_kept = surf[kept_mask]
        empty_kept = empty[kept_mask]
        filled_kept = filled[kept_mask]
        # 保留的各类 token 数占总保留数的比例
        surf_frac = float(surf_kept.mean())
        empty_frac = float(empty_kept.mean())
        filled_frac = float(filled_kept.mean())
        # 各类 token 的召回率
        surf_recall = float(surf_kept.sum() / max(surf.sum(), 1))
        empty_recall = float(empty_kept.sum() / max(empty.sum(), 1))
        filled_recall = float(filled_kept.sum() / max(filled.sum(), 1))
    else:
        surf_frac = empty_frac = filled_frac = 0.0
        surf_recall = empty_recall = filled_recall = 0.0

    # 5. 保留 token 之间的成对距离的均值和标准差（衡量聚集程度）
    if len(c_kept) > 1:
        # 随机抽 500 对避免 O(N^2)
        n_sample = min(500, len(c_kept))
        idx1 = np.random.choice(len(c_kept), size=n_sample, replace=True)
        idx2 = np.random.choice(len(c_kept), size=n_sample, replace=True)
        pairwise_dists = np.linalg.norm(c_kept[idx1] - c_kept[idx2], axis=1)
        pair_dist_mean = float(pairwise_dists.mean())
        pair_dist_std = float(pairwise_dists.std())
    else:
        pair_dist_mean = pair_dist_std = 0.0

    return {
        "nn_dist_kept_mean": float(nn_dist_kept.mean()),
        "nn_dist_kept_std": float(nn_dist_kept.std()),
        "dist_all_to_kept_mean": float(dist_all_to_kept.mean()),
        "dist_all_to_kept_std": float(dist_all_to_kept.std()),
        "uniformity_std_ratio_per_block": uniformity_std,
        "covered_grid_blocks_ratio": covered_blocks,
        "boundary_dist_kept_mean": float(d2b_kept.mean()),
        "boundary_dist_kept_std": float(d2b_kept.std()),
        "boundary_dist_all_mean": float(d2b_all.mean()),
        "surf_frac_kept": surf_frac,
        "empty_frac_kept": empty_frac,
        "filled_frac_kept": filled_frac,
        "surf_recall": surf_recall,
        "empty_recall": empty_recall,
        "filled_recall": filled_recall,
        "pairwise_dist_kept_mean": pair_dist_mean,
        "pairwise_dist_kept_std": pair_dist_std,
    }


def _embedding_pruner_extra_metrics(
    name: str,
    indices: Sequence[int],
    token_ids: torch.Tensor,
    voxel_grid: torch.Tensor,
    vq_emb: torch.nn.Embedding,
    log: _TeeLogger,
) -> Dict[str, float]:
    """otprune / reconot 共用的 embedding 剪枝扩展指标（共现、图、多样性、BPE、空间）。"""
    extra: Dict[str, float] = {}
    cooc = _token_cooccurrence_change(list(range(MESH_SEQ_LEN)), indices, token_ids)
    log.line(f"  [{name} cooc] {cooc}")
    graph = _spatial_graph_metrics(indices)
    log.line(f"  [{name} graph] {graph}")
    div = _token_diversity_metrics(token_ids, indices)
    log.line(f"  [{name} diversity] {div}")
    bpe_pair_stats = _bpe_pair_retention_analysis(
        token_ids,
        indices,
        directions=((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)),
    )
    extra.update({f"bpe_{k}": float(v) for k, v in bpe_pair_stats.items()})
    log.line(f"  [BPE pair retention] {bpe_pair_stats}")
    ot_spatial_extra = _otprune_spatial_analysis(indices, token_ids, voxel_grid, vq_emb)
    log.line(f"  [{name} spatial analysis]")
    for k, v in sorted(ot_spatial_extra.items()):
        log.line(f"      otspatial.{k}={v:.6g}")
    extra.update({f"otspatial_{k}": float(v) for k, v in ot_spatial_extra.items()})
    return extra


def _bpe_pair_retention_analysis(
    full_token_ids: torch.Tensor,
    kept_indices: Sequence[int],
    directions: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    ),
    top_fractions: Tuple[float, ...] = (0.05, 0.10, 0.25),
) -> Dict[str, float]:
    """
    基于 token ID 共现频率模拟 3D BPE 高频对的保留情况。

    核心变更：不再使用网格坐标对，而是统计 token ID 对在 6 邻域内共同出现的次数。
    高频对的定义：在全量 token 序列中，相邻出现的 token ID 对（无向）频率最高的那一部分。

    返回:
      token_pair_freq_weighted_recall: 保留集内 token 对频率加权召回
      kept_token_pair_count / full_token_pair_count: 保留集内 / 全集中出现的 token ID 对种类数
      mean_pair_freq_kept / mean_pair_freq_full: 保留对 / 全对平均频率
      highfreq_type_recall_topXXpct: 高频 token 对中，两个 token ID 在保留集中都被保留下来的比例
      highfreq_instance_recall_topXXpct: 高频 token 对在全集中共现的总次数中，保留集内仍相邻的比例
      edge_density_kept: 保留集内部实际存在的 token 对种类数 / 最大可能（已弃用，保留为兼容）
    """
    tokens = full_token_ids.detach().cpu().numpy().astype(np.int64)
    coord_to_idx = {flat_index_to_coord(i): i for i in range(MESH_SEQ_LEN)}

    # 1. 构建全集的 token ID 对共现频率（无向，每次相邻出现 +1）
    pair_freq: Dict[Tuple[int, int], int] = defaultdict(int)
    for i in range(MESH_SEQ_LEN):
        x, y, z = flat_index_to_coord(i)
        ti = int(tokens[i])
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z:
                j = coord_to_idx[(nx, ny, nz)]
                if j != i:
                    tj = int(tokens[j])
                    # 统一为无向对，避免 (a,b) 和 (b,a) 重复
                    pair = tuple(sorted((ti, tj)))
                    pair_freq[pair] += 1  # 注意：6 方向会导致每条空间边被计数两次（作为 token 对）
    # 因为每条空间边被两个方向各记录一次，pair_freq 的频率会是偶数（多数为 2 的倍数），
    # 但同一个 token 对可能出现在不同位置，频率自然增加，从而区分高频对。

    if not pair_freq:
        return {}

    pair_list = list(pair_freq.keys())
    freq_array = np.array([pair_freq[p] for p in pair_list], dtype=np.float64)
    total_freq = freq_array.sum()

    # 2. 保留集中 token ID 的集合
    kept_set = set(int(i) for i in kept_indices)
    kept_token_ids = {tokens[i] for i in kept_indices}

    # 保留集内实际出现的 token 对（在保留 token 之间相邻）
    kept_pairs = set()
    kept_pair_freq_sum = 0.0
    kept_pair_count = 0
    for pair, freq in pair_freq.items():
        a, b = pair
        if a in kept_token_ids and b in kept_token_ids:
            kept_pairs.add(pair)
            kept_pair_freq_sum += freq
            kept_pair_count += 1

    # 更精细：只统计保留集内真正相邻的那些 token 对实例
    # 重新遍历保留集网格，构建保留集内部的 token 对
    kept_pair_instance_freq = defaultdict(int)
    for i in kept_indices:
        x, y, z = flat_index_to_coord(i)
        ti = int(tokens[i])
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z:
                j = coord_to_idx[(nx, ny, nz)]
                if j in kept_set and j != i:
                    tj = int(tokens[j])
                    pair = tuple(sorted((ti, tj)))
                    kept_pair_instance_freq[pair] += 1

    weighted_recall = kept_pair_freq_sum / total_freq if total_freq > 0 else 0.0

    # 3. 高频 token 对的保留（两种粒度）
    highfreq_type_recall = {}
    highfreq_instance_recall = {}
    for frac in top_fractions:
        if frac <= 0.0 or frac > 1.0:
            continue
        threshold = np.percentile(freq_array, 100.0 * (1.0 - frac))
        high_mask = freq_array >= threshold
        high_pairs = [pair_list[i] for i in np.where(high_mask)[0]]

        # 类型保留：两个 token 是否都被保留
        kept_type = sum(1 for p in high_pairs if p[0] in kept_token_ids and p[1] in kept_token_ids)
        highfreq_type_recall[f"top{int(frac*100):02d}pct_type"] = kept_type / len(high_pairs)

        # 实例保留：这些高频对在全集中共现的总次数，在保留集中仍相邻的次数占比
        total_instance_count = sum(pair_freq[p] for p in high_pairs)
        kept_instance_count = sum(kept_pair_instance_freq.get(p, 0) for p in high_pairs)
        highfreq_instance_recall[f"top{int(frac*100):02d}pct_instance"] = kept_instance_count / total_instance_count if total_instance_count > 0 else 0.0

    # 4. 平均频率
    mean_freq_full = float(freq_array.mean())
    mean_freq_kept = float(np.mean([pair_freq[p] for p in kept_pairs])) if kept_pairs else 0.0

    # 5. 保留集内部空间边的密度（沿用网格边密度，仅供参考）
    n_kept = len(kept_indices)
    max_possible_edges = n_kept * (n_kept - 1) // 2
    edge_count_kept = len(kept_pair_instance_freq)
    edge_density = edge_count_kept / max_possible_edges if max_possible_edges > 0 else 0.0

    return {
        "token_pair_freq_weighted_recall": weighted_recall,
        "kept_token_pair_types": float(len(kept_pairs)),
        "full_token_pair_types": float(len(pair_list)),
        "mean_pair_freq_kept": mean_freq_kept,
        "mean_pair_freq_full": mean_freq_full,
        "edge_density_kept": edge_density,
        **highfreq_type_recall,
        **highfreq_instance_recall,
    }

def _token_cooccurrence_change(
    all_indices: Sequence[int],
    kept_indices: Sequence[int],
    token_ids: torch.Tensor,
) -> Dict[str, float]:
    """
    分析剪枝前后 6‑邻域共现边的保留情况。
    返回：
    - cooccur_edge_recall: 原始高频共现边在保留子集中仍存在的比例
    - kept_pairwise_neighbor_rate: 保留 token 中彼此为邻居的比例
    - mean_neighbor_degree_kept: 保留 token 平均的保留邻居数
    """
    tokens = token_ids.detach().cpu().numpy()
    coord_to_idx = {flat_index_to_coord(i): i for i in range(MESH_SEQ_LEN)}
    
    # 构建全集的共现边（无序）频率
    full_edges = defaultdict(int)
    for i in range(MESH_SEQ_LEN):
        x, y, z = flat_index_to_coord(i)
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z:
                j = coord_to_idx[(nx, ny, nz)]
                if j != i:
                    edge = tuple(sorted((tokens[i], tokens[j])))
                    full_edges[edge] += 1
    
    # 保留集内部的共现边
    kept_set = set(kept_indices)
    kept_edges = defaultdict(int)
    for i in kept_indices:
        x, y, z = flat_index_to_coord(i)
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z:
                j = coord_to_idx[(nx, ny, nz)]
                if j in kept_set and j != i:
                    edge = tuple(sorted((tokens[i], tokens[j])))
                    kept_edges[edge] += 1

    # 取全集中频率 top 10% 的边，计算保留率
    if full_edges:
        min_freq = np.percentile(list(full_edges.values()), 90)
        top_edges = {e for e, c in full_edges.items() if c >= min_freq}
        recalled_edges = top_edges & set(kept_edges.keys())
        cooccur_recall = len(recalled_edges) / len(top_edges) if top_edges else 1.0
    else:
        cooccur_recall = 1.0

    # 保留 token 的邻居保留度
    neighbor_counts = []
    for i in kept_indices:
        cnt = 0
        x, y, z = flat_index_to_coord(i)
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z:
                j = coord_to_idx[(nx, ny, nz)]
                if j in kept_set:
                    cnt += 1
        neighbor_counts.append(cnt)
    mean_deg = np.mean(neighbor_counts) if neighbor_counts else 0.0

    # 保留 token 之间的邻居对数占总保留 token 对的比例
    n_kept = len(kept_indices)
    total_pairs = n_kept * (n_kept - 1) / 2
    neighbor_pairs = sum(neighbor_counts) / 2  # 每条边算两次
    kept_pairwise_rate = neighbor_pairs / total_pairs if total_pairs > 0 else 0.0

    return {
        "cooccur_edge_recall_top10": float(cooccur_recall),
        "kept_pairwise_neighbor_rate": float(kept_pairwise_rate),
        "mean_neighbor_degree_kept": float(mean_deg),
    }

def _spatial_graph_metrics(
    kept_indices: Sequence[int],
    all_indices: Sequence[int] = None,
) -> Dict[str, float]:
    """
    保留 token 在 3D 网格上的邻接图特征。
    """
    coord_to_idx = {flat_index_to_coord(i): i for i in range(MESH_SEQ_LEN)}
    kept_set = set(kept_indices)
    # 构建图：只考虑保留 token 之间的边
    edges = []
    for i in kept_indices:
        x, y, z = flat_index_to_coord(i)
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z:
                j = coord_to_idx[(nx, ny, nz)]
                if j in kept_set and j > i:  # 防止重复边
                    edges.append((i, j))
    # 计算连通分量
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    comp_sizes = []
    for node in kept_indices:
        if node not in visited:
            stack = [node]
            size = 0
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                size += 1
                for nb in adj[v]:
                    if nb not in visited:
                        stack.append(nb)
            comp_sizes.append(size)
    num_comp = len(comp_sizes)
    max_comp_ratio = max(comp_sizes) / len(kept_indices) if comp_sizes else 0.0
    mean_comp_size = np.mean(comp_sizes) if comp_sizes else 0.0
    # 平均度
    degrees = [len(adj[n]) for n in kept_indices]
    mean_deg = np.mean(degrees) if degrees else 0.0
    return {
        "num_connected_components": float(num_comp),
        "max_component_ratio": float(max_comp_ratio),
        "mean_component_size": float(mean_comp_size),
        "mean_degree": float(mean_deg),
    }

def _token_diversity_metrics(
    all_token_ids: torch.Tensor,
    kept_indices: Sequence[int],
) -> Dict[str, float]:
    t = all_token_ids.detach().cpu().numpy()
    all_ids = t
    kept_ids = t[list(kept_indices)]
    # 唯一 token 数量
    unique_all = len(set(all_ids))
    unique_kept = len(set(kept_ids))
    # 覆盖率
    coverage = unique_kept / unique_all if unique_all else 1.0
    # 频次分布的 KL 散度（近似）
    from collections import Counter
    cnt_all = Counter(all_ids)
    cnt_kept = Counter(kept_ids)
    # 归一化为概率
    prob_all = np.array([cnt_all[i] for i in range(max(all_ids)+1)], dtype=np.float64)
    prob_all /= prob_all.sum()
    prob_kept = np.array([cnt_kept[i] for i in range(max(all_ids)+1)], dtype=np.float64)
    prob_kept /= prob_kept.sum()
    # 避免 log(0)
    mask = prob_all > 0
    kl = (prob_all[mask] * np.log(prob_all[mask] / (prob_kept[mask] + 1e-12))).sum()
    # 保留 token 中低频 token（出现次数<=2）的比例
    low_freq_thresh = 2
    low_freq_all = sum(1 for c in cnt_all.values() if c <= low_freq_thresh)
    low_freq_kept = sum(1 for i in kept_ids if cnt_all[i] <= low_freq_thresh)
    low_freq_recall = low_freq_kept / low_freq_all if low_freq_all else 1.0

    return {
        "unique_token_coverage": float(coverage),
        "kl_divergence_all_vs_kept": float(kl),
        "low_freq_token_recall": float(low_freq_recall),
    }

def _six_neighbor_embedding_error_flat(emb: torch.Tensor, is_empty: torch.Tensor) -> torch.Tensor:
    """
    VQ embedding 的 6-邻域 L2 预测误差 [1024]（用于 otprune 侧 embedding 覆盖分析）。

    不依赖 loco3d 实现细节；当前 loco3d v20 使用 token 邻接熵，不用此量。
    """
    d = emb.shape[-1]
    emb_grid = emb.view(GRID_X, GRID_Y, GRID_Z, d)
    sum_nb = torch.zeros_like(emb_grid)
    cnt_nb = torch.zeros(
        GRID_X,
        GRID_Y,
        GRID_Z,
        device=emb_grid.device,
        dtype=torch.float32,
    )
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        if dx == 1:
            sum_nb[:-1] += emb_grid[1:]
            cnt_nb[:-1] += 1
        elif dx == -1:
            sum_nb[1:] += emb_grid[:-1]
            cnt_nb[1:] += 1
        if dy == 1:
            sum_nb[:, :-1] += emb_grid[:, 1:]
            cnt_nb[:, :-1] += 1
        elif dy == -1:
            sum_nb[:, 1:] += emb_grid[:, :-1]
            cnt_nb[:, 1:] += 1
        if dz == 1:
            sum_nb[:, :, :-1] += emb_grid[:, :, 1:]
            cnt_nb[:, :, :-1] += 1
        elif dz == -1:
            sum_nb[:, :, 1:] += emb_grid[:, :, :-1]
            cnt_nb[:, :, 1:] += 1
    mean_nb = sum_nb / cnt_nb.unsqueeze(-1).clamp(min=1.0)
    err = torch.norm(emb_grid - mean_nb, p=2, dim=-1).reshape(-1)
    # 确保 is_empty 与 err 在同一设备上
    if is_empty.device != err.device:
        is_empty = is_empty.to(err.device)
    return err.masked_fill(is_empty, 0.0)

def _loco3d_neighbor_entropy_np(
    token_ids: torch.Tensor,
    voxel_grid: torch.Tensor | None,
) -> np.ndarray:
    """计算 6‑邻域 token id 分布熵 [1024]（CPU numpy），与旧版 loco3d 一致。"""
    tokens = token_ids.detach().cpu().numpy().astype(np.int64)
    coord_to_idx = {}
    for i in range(MESH_SEQ_LEN):
        coord_to_idx[flat_index_to_coord(i)] = i

    neighbor_counts = [defaultdict(int) for _ in range(MESH_SEQ_LEN)]
    total_degree = np.zeros(MESH_SEQ_LEN, dtype=np.int32)

    for i in range(MESH_SEQ_LEN):
        x, y, z = flat_index_to_coord(i)
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx, ny, nz = x+dx, y+dy, z+dz
            if not (0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z):
                continue
            j = coord_to_idx[(nx, ny, nz)]
            if j != i:
                neighbor_counts[i][tokens[j]] += 1
                total_degree[i] += 1

    entropy = np.zeros(MESH_SEQ_LEN, dtype=np.float32)
    for i in range(MESH_SEQ_LEN):
        if total_degree[i] == 0:
            entropy[i] = 0.0
        else:
            cnts = np.array(list(neighbor_counts[i].values()), dtype=np.float32)
            probs = cnts / cnts.sum()
            entropy[i] = float(-np.sum(probs * np.log(probs + 1e-9)))

    return entropy.astype(np.float64)

def _embedding_metrics(
    indices: Sequence[int],
    embed: torch.nn.Embedding,
    token_ids: torch.Tensor,
    voxel_grid: torch.Tensor | None,
) -> Dict[str, float]:
    device = embed.weight.device
    t = token_ids.detach().long().view(-1).to(device)
    emb_all = gather_embeddings(embed, t)  # [1024, D]
    
    # 检查 emb_all 的设备
    emb_device = emb_all.device
    emb_norm = emb_all / emb_all.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    
    # 确保 idx_kept 与 emb_norm 在同一设备上
    idx_kept = torch.tensor(sorted(int(i) for i in indices), dtype=torch.long, device=emb_device)
    idx_all = torch.arange(MESH_SEQ_LEN, device=emb_device, dtype=torch.long)
    idx_disc = idx_all[~torch.isin(idx_all, idx_kept)]

    kept = emb_norm[idx_kept]
    disc = emb_norm[idx_disc]
    k = int(idx_kept.numel())

    # 保留集内平均余弦相似度（多样性：越低越分散）
    if k >= 2:
        gram_k = kept @ kept.t()
        triu = torch.triu_indices(k, k, offset=1, device=emb_device)
        pairwise_cos = gram_k[triu[0], triu[1]]
        mean_cos_kept = float(pairwise_cos.mean().item())
        std_cos_kept = float(pairwise_cos.std().item())
    else:
        mean_cos_kept = std_cos_kept = float("nan")

    # 丢弃 token 到最近保留 token 的余弦距离 (1 - cos)
    if disc.numel() > 0 and k > 0:
        sim = disc @ kept.t()
        max_cos = sim.max(dim=1).values
        mean_cos_disc_to_kept = float(max_cos.mean().item())
        mean_l2_disc_to_kept = float((1.0 - max_cos).mean().item())
        min_cos_disc = float(max_cos.min().item())
        p10_cos = float(torch.quantile(max_cos, 0.1).item())
    else:
        mean_cos_disc_to_kept = mean_l2_disc_to_kept = min_cos_disc = p10_cos = float("nan")

    # 与「高预测误差」oracle 的重叠
    is_empty = torch.zeros(MESH_SEQ_LEN, dtype=torch.bool, device=emb_device)
    if voxel_grid is not None:
        _, is_empty, _, _ = latent_surface_mask(voxel_grid.cpu())
        is_empty = is_empty.to(emb_device)
    err = _six_neighbor_embedding_error_flat(emb_all, is_empty)
    err_np = err.detach().cpu().numpy()
    nz = ~is_empty.cpu().numpy()
    if nz.any():
        thresh = float(np.median(err_np[nz]))
    else:
        thresh = float(np.median(err_np))
    high_err_mask = err_np >= thresh
    kept_mask = _indices_to_mask(indices)
    oracle_high_err_recall = (
        float(np.logical_and(kept_mask, high_err_mask).sum() / max(high_err_mask.sum(), 1))
    )
    mean_err_kept = float(err_np[kept_mask].mean()) if kept_mask.any() else 0.0
    mean_err_all_nonempty = float(err_np[nz].mean()) if nz.any() else float(err_np.mean())

    # 全局 embedding 偏离
    gmean = emb_all.mean(dim=0, keepdim=True)
    glob_dev = torch.norm(emb_all - gmean, p=2, dim=-1).detach().cpu().numpy()
    mean_glob_kept = float(glob_dev[kept_mask].mean()) if kept_mask.any() else 0.0

    uniq_tok = int(torch.unique(t[idx_kept]).numel()) if k > 0 else 0
    tok_repeat_rate = 1.0 - uniq_tok / max(k, 1)

    return {
        "mean_pairwise_cosine_kept": mean_cos_kept,
        "std_pairwise_cosine_kept": std_cos_kept,
        "mean_max_cosine_disc_to_kept": mean_cos_disc_to_kept,
        "mean_one_minus_cos_disc_to_kept": mean_l2_disc_to_kept,
        "min_max_cosine_disc_to_kept": min_cos_disc,
        "p10_max_cosine_disc_to_kept": p10_cos,
        "oracle_high_neighbor_err_recall": oracle_high_err_recall,
        "mean_neighbor_err_kept": mean_err_kept,
        "mean_neighbor_err_all_nonempty": mean_err_all_nonempty,
        "mean_global_dev_kept": mean_glob_kept,
        "unique_token_count": float(uniq_tok),
        "token_repeat_rate": float(tok_repeat_rate),
    }

def _token_id_overlap(
    indices_a: Sequence[int],
    indices_b: Sequence[int],
    token_ids: torch.Tensor,
) -> Dict[str, float]:
    t = token_ids.detach().long().view(-1).cpu().numpy()
    sa = {int(t[i]) for i in indices_a}
    sb = {int(t[i]) for i in indices_b}
    inter = len(sa & sb)
    union = len(sa | sb)
    return {
        "token_id_jaccard": inter / union if union else 0.0,
        "token_id_intersection": float(inter),
        "token_id_union": float(union),
    }


def _rank_agreement(
    indices_a: Sequence[int],
    indices_b: Sequence[int],
    score: np.ndarray,
    *,
    score_label: str = "score",
) -> Dict[str, float]:
    """用连续 score 向量比较两方法的「重要性」排序一致性。"""
    sel_a = _indices_to_mask(indices_a).astype(np.float64)
    sel_b = _indices_to_mask(indices_b).astype(np.float64)
    return {
        f"spearman_{score_label}_vs_sel_a": _spearman_rho(score, sel_a),
        f"spearman_{score_label}_vs_sel_b": _spearman_rho(score, sel_b),
        f"spearman_sel_a_vs_sel_b": _spearman_rho(sel_a, sel_b),
    }


class _TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")

    def line(self, msg: str = "") -> None:
        print(msg, flush=True)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _run_prune(
    name: str,
    keep_ratio: float,
    seed: int,
    token_ids: torch.Tensor,
    voxel_grid: torch.Tensor,
    vq_emb: torch.nn.Embedding,
    eval_cfg_dir: Path,
    sample_idx: int,
    tag: str,
) -> Tuple[IndexSetStats, Dict[str, Any], float]:
    Pruner = get_pruner_class(name)
    extra = load_pruner_extra_kwargs(eval_cfg_dir, name)
    pruner = Pruner(keep_ratio=keep_ratio, seed=seed, **extra)
    t0 = time.perf_counter()
    pruned, meta = pruner.prune(
        token_ids,
        voxel_grid,
        vq_embeddings=vq_emb,
        _log_sample_idx=sample_idx,
        _log_tag=tag,
    )
    elapsed = time.perf_counter() - t0
    indices = meta.get("indices")
    if indices is None:
        indices = list(range(int(pruned.numel())))
    return IndexSetStats.from_indices(indices), meta, elapsed


def _aggregate_mean(rows: List[Dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if key in r and not math.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="对比 loco3d / otprune / reconot 剪枝（详细指标 + log）")
    p.add_argument("--data-csv", type=str, default="../data/metadata.csv")
    p.add_argument("--glb-dir", type=str, default="../data")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--keep-ratios", type=str, default="0.75,0.5,0.25,0.1")
    p.add_argument("--methods", type=str, default="loco3d,otprune,reconot,random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-trials", type=int, default=3, help="random 剪枝重复次数（看 loco3d 是否接近 random）")
    p.add_argument("--device", type=str, default="cuda:0", help="VLM 设备（仅 --with-vlm）")
    p.add_argument("--vqvae-device", type=str, default="cuda:0")
    p.add_argument("--eval-config-dir", type=str, default="configs/eval")
    p.add_argument("--mesh-cache-dir", type=str, default="")
    p.add_argument("--mesh-cache-readonly", action="store_true")
    p.add_argument("--log-file", type=str, default="", help="人类可读 log；默认 ../output/logs/compare_loco3d_otprune_<ts>.log")
    p.add_argument("--jsonl-file", type=str, default="", help="机器可读 jsonl；默认同目录 .jsonl")
    p.add_argument("--with-vlm", action="store_true", help="加载 VLM 并生成 caption（慢、占显存）")
    p.add_argument("--model-id", type=str, default="yejunliang23/ShapeLLM-7B-omni")
    p.add_argument("--vlm-torch-dtype", type=str, default="bfloat16", choices=("auto", "float16", "bfloat16"))
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--caption-prompt", type=str, default=EvalConfig.caption_prompt)
    args = p.parse_args()

    cfg_stub = EvalConfig(
        data_csv=args.data_csv,
        glb_dir=args.glb_dir,
        eval_config_dir=args.eval_config_dir,
        mesh_cache_dir=args.mesh_cache_dir,
        mesh_cache_readonly=args.mesh_cache_readonly,
    )
    cfg_stub = resolve_repo_paths(cfg_stub, _REPO_ROOT)
    keep_ratios = _parse_float_list(args.keep_ratios)
    methods = _parse_str_list(args.methods)
    _require_pruners(methods)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else _REPO_ROOT.parent / "output" / "logs" / f"compare_loco3d_otprune_{ts}.log"
    jsonl_path = (
        Path(args.jsonl_file)
        if args.jsonl_file
        else log_path.with_suffix(".jsonl")
    )

    if not torch.cuda.is_available() and "cuda" in (args.device + args.vqvae_device):
        print("Warning: CUDA 不可用，使用 CPU。", file=sys.stderr)

    vqvae_dev = resolve_torch_device(args.vqvae_device)
    vlm_dev = resolve_torch_device(args.device)
    init_cuda_for_eval(vqvae_dev, vlm_dev if args.with_vlm else vqvae_dev)

    log = _TeeLogger(log_path)
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    def emit_json(record: Dict[str, Any]) -> None:
        jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_f.flush()

    log.line(f"# compare_loco3d_otprune  started={_utc_now()}")
    log.line(f"# log_file={log_path.resolve()}")
    log.line(f"# jsonl_file={jsonl_path.resolve()}")
    log.line(f"# methods={methods} keep_ratios={keep_ratios} num_samples={args.num_samples}")
    log.line(f"# seed={args.seed} random_trials={args.random_trials} with_vlm={args.with_vlm}")
    log.line("")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.line("Loading VQVAE...")
    vqvae = load_vqvae(vqvae_dev)
    vq_emb = getattr(vqvae.vq, "embeddings", None)
    if vq_emb is None:
        log.line("ERROR: VQVAE 无 vq.embeddings")
        log.close()
        jsonl_f.close()
        return 1

    model = processor = tokenizer = None
    if args.with_vlm:
        from eval.generator import generate_caption

        log.line(f"Loading VLM on {vlm_dev}...")
        model, processor, tokenizer = load_llm(
            args.model_id,
            vlm_dev,
            load_in_4bit=False,
            vlm_torch_dtype=args.vlm_torch_dtype,
        )
        _generate_caption = generate_caption
    else:
        _generate_caption = None

    samples = list(
        iter_dataset(
            cfg_stub.data_csv,
            cfg_stub.glb_dir,
            num_samples=args.num_samples,
            skip_missing_glb=True,
        )
    )
    if not samples:
        log.line("ERROR: 无可用样本，检查 --data-csv / --glb-dir")
        log.close()
        jsonl_f.close()
        return 1

    eval_cfg_dir = Path(cfg_stub.eval_config_dir)
    agg_pair: Dict[Tuple[str, str, float], List[Dict[str, float]]] = defaultdict(list)
    agg_method: Dict[Tuple[str, float], List[Dict[str, float]]] = defaultdict(list)

    for si, sample in enumerate(samples):
        log.line("=" * 88)
        log.line(f"SAMPLE {si}: file_identifier={sample.file_identifier}")
        log.line(f"  glb={sample.glb_path}")
        log.line(f"  ref_captions_n={len(sample.captions)}")
        log.line("=" * 88)

        try:
            token_ids, voxel_grid = mesh_to_tokens(
                sample.glb_path,
                vqvae,
                vqvae_dev,
                file_identifier=sample.file_identifier,
                mesh_cache_dir=cfg_stub.mesh_cache_dir,
                mesh_cache_readonly=cfg_stub.mesh_cache_readonly,
                vlm_device=vlm_dev if args.with_vlm else None,
            )
        except Exception:
            log.line("  mesh_to_tokens FAILED:")
            log.line(traceback.format_exc())
            emit_json(
                {
                    "event": "sample_error",
                    "sample_idx": si,
                    "file_identifier": sample.file_identifier,
                    "error": "mesh_to_tokens_failed",
                }
            )
            continue

        t_cpu = token_ids.detach().long().view(-1)
        uniq_tok_full = int(t_cpu.unique().numel())
        log.line(f"  tokens: unique={uniq_tok_full}/1024 voxel_occupancy_sum={int(voxel_grid.sum())}")

        # embedding 邻域误差（otprune 相关）+ loco3d 邻接熵（v20 实际打分）
        device = vq_emb.weight.device
        t_dev = t_cpu.to(device)
        emb_all = gather_embeddings(vq_emb, t_dev)
        is_empty = torch.zeros(MESH_SEQ_LEN, dtype=torch.bool, device=device)
        if voxel_grid is not None:
            _, is_empty, _, _ = latent_surface_mask(voxel_grid.cpu())
            is_empty = is_empty.to(device)
        err_t = _six_neighbor_embedding_error_flat(emb_all, is_empty)
        err_np = err_t.detach().cpu().numpy()
        
        # ---- 新增：6‑邻居重要性量度 ----
        neighbor_importance = _six_neighbor_embedding_importance_measures(emb_all, is_empty)
        local_dissim_np = neighbor_importance["local_dissim"]
        neighbor_var_np = neighbor_importance["neighbor_var"]
        
        entropy_np = _loco3d_neighbor_entropy_np(token_ids, voxel_grid)

        for kr in keep_ratios:
            k_target = target_keep_count(kr)
            log.line("")
            log.line(f"--- keep_ratio={kr:.4g}  k_target={k_target} ---")

            runs: Dict[str, List[Tuple[IndexSetStats, Dict[str, Any], float]]] = {}

            for name in methods:
                if name == "random":
                    trials = []
                    for rt in range(args.random_trials):
                        stats, meta, elapsed = _run_prune(
                            name,
                            kr,
                            args.seed + rt,
                            token_ids,
                            voxel_grid,
                            vq_emb,
                            eval_cfg_dir,
                            si,
                            sample.file_identifier,
                        )
                        trials.append((stats, meta, elapsed))
                    runs[name] = trials
                    # 主 random 用 seed 作对比代表
                    stats0, meta0, elapsed0 = trials[0]
                else:
                    stats0, meta0, elapsed0 = _run_prune(
                        name,
                        kr,
                        args.seed,
                        token_ids,
                        voxel_grid,
                        vq_emb,
                        eval_cfg_dir,
                        si,
                        sample.file_identifier,
                    )
                    runs[name] = [(stats0, meta0, elapsed0)]

                stats0, meta0, elapsed0 = runs[name][0]
                spatial = _spatial_metrics(stats0.indices, voxel_grid)
                extra_row: Dict[str, float] = {}
                if name in ("otprune", "reconot"):
                    extra_row = _embedding_pruner_extra_metrics(
                        name,
                        stats0.indices,
                        token_ids,
                        voxel_grid,
                        vq_emb,
                        log,
                    )
                emb_m = _embedding_metrics(stats0.indices, vq_emb, token_ids, voxel_grid)
                method_row = {
                    "prune_time_sec": elapsed0,
                    "k_actual": float(stats0.k),
                    **spatial,
                    **emb_m,
                    **extra_row,
                }
                agg_method[(name, kr)].append(method_row)

                log.line(f"  [{name}] k={stats0.k} prune_time={elapsed0:.4f}s")
                for mk, mv in sorted(spatial.items()):
                    log.line(f"      spatial.{mk}={mv:.6g}")
                for mk, mv in sorted(emb_m.items()):
                    if isinstance(mv, float) and not math.isnan(mv):
                        log.line(f"      embed.{mk}={mv:.6g}")

                if name == "loco3d" and isinstance(meta0.get("diagnostics"), dict):
                    diag = meta0["diagnostics"]
                    log.line(f"      loco3d.diagnostics={json.dumps(diag, ensure_ascii=False)[:2000]}")
                if name == "reconot" and isinstance(meta0.get("diagnostics"), dict):
                    diag = meta0["diagnostics"]
                    log.line(f"      reconot.diagnostics={json.dumps(diag, ensure_ascii=False)[:2000]}")

                if name == "random" and len(runs[name]) > 1:
                    j_list = []
                    for t_i in range(1, len(runs[name])):
                        ja = runs[name][0][0]
                        jb = runs[name][t_i][0]
                        j_list.append(_set_overlap_metrics(ja, jb)["jaccard"])
                    log.line(
                        f"      random_trials_jaccard_mean={float(np.mean(j_list)):.6g} "
                        f"(trials={args.random_trials}, 应与 k/N≈{k_target/MESH_SEQ_LEN:.4g} 同量级若近似独立)"
                    )

                rec = {
                    "event": "method_stats",
                    "sample_idx": si,
                    "file_identifier": sample.file_identifier,
                    "keep_ratio": kr,
                    "method": name,
                    "k_target": k_target,
                    **method_row,
                    "pruner_metadata_keys": list(meta0.keys()),
                }
                if name == "loco3d":
                    rec["loco3d_diagnostics"] = meta0.get("diagnostics")
                if name == "reconot":
                    rec["reconot_diagnostics"] = meta0.get("diagnostics")
                    rec["reconot_selector"] = meta0.get("selector")
                emit_json(rec)

                if args.with_vlm and _generate_caption is not None and model is not None:
                    mesh_str = tokens_to_mesh_string(
                        t_cpu[torch.tensor(stats0.indices, dtype=torch.long)]
                    )
                    try:
                        caption, elapsed, n_in, n_out = _generate_caption(
                            model,
                            processor,
                            tokenizer,
                            mesh_str,
                            args.caption_prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.7,
                            top_p=0.7,
                            top_k=8192,
                            device=vlm_dev,
                        )
                        scores = compute_text_metrics(caption, sample.captions)
                        log.line(f"      vlm.caption={caption[:120]!r}...")
                        log.line(
                            "      "
                            f"vlm.bleu_4={scores.get('bleu_4', 0):.4f} "
                            f"rouge_l={scores.get('rouge_l', 0):.4f} "
                            f"sentence_bert={scores.get('sentence_bert') or 0:.4f} "
                            f"simcse={scores.get('simcse') or 0:.4f}"
                        )
                        emit_json(
                            {
                                "event": "vlm_caption",
                                "sample_idx": si,
                                "file_identifier": sample.file_identifier,
                                "keep_ratio": kr,
                                "method": name,
                                "caption": caption,
                                "generation_time_sec": elapsed,
                                **scores,
                            }
                        )
                    except Exception:
                        log.line(f"      vlm FAILED: {traceback.format_exc()}")

            # 主对比：各 method 取第一次 run
            primary: Dict[str, IndexSetStats] = {
                m: runs[m][0][0] for m in methods if m in runs
            }

            if "loco3d" in primary and "otprune" in primary:
                rank_emb = _rank_agreement(
                    primary["loco3d"].indices,
                    primary["otprune"].indices,
                    err_np,
                    score_label="embed_neighbor_err",
                )
                rank_ent = _rank_agreement(
                    primary["loco3d"].indices,
                    primary["otprune"].indices,
                    entropy_np,
                    score_label="loco3d_neighbor_entropy",
                )
                log.line(f"  [rank vs embed_neighbor_err] {rank_emb}")
                log.line(f"  [rank vs loco3d_neighbor_entropy] {rank_ent}")
                # 新增：用 otprune 的选择直接评估新量度的重要性相关性
                rank_dissim = _rank_agreement(
                    primary["otprune"].indices,
                    primary["otprune"].indices,
                    local_dissim_np,
                    score_label="local_dissim",
                )
                rank_var = _rank_agreement(
                    primary["otprune"].indices,
                    primary["otprune"].indices,
                    neighbor_var_np,
                    score_label="neighbor_var",
                )
                # 只关注与 sel_b (即 otprune 选择) 的 spearman 系数
                log.line(f"  [rank vs local_dissim] spearman_with_otprune_sel={rank_dissim['spearman_local_dissim_vs_sel_b']:.4f}")
                log.line(f"  [rank vs neighbor_var]   spearman_with_otprune_sel={rank_var['spearman_neighbor_var_vs_sel_b']:.4f}")
                emit_json(
                    {
                        "event": "rank_agreement",
                        "sample_idx": si,
                        "file_identifier": sample.file_identifier,
                        "keep_ratio": kr,
                        **rank_emb,
                        **rank_ent,
                    }
                )

            if "otprune" in primary and "reconot" in primary:
                rank_recon = _rank_agreement(
                    primary["otprune"].indices,
                    primary["reconot"].indices,
                    err_np,
                    score_label="embed_neighbor_err",
                )
                log.line(f"  [rank otprune vs reconot vs embed_neighbor_err] {rank_recon}")
                rank_recon_dissim = _rank_agreement(
                    primary["reconot"].indices,
                    primary["reconot"].indices,
                    local_dissim_np,
                    score_label="local_dissim",
                )
                log.line(
                    f"  [rank vs local_dissim] spearman_with_reconot_sel="
                    f"{rank_recon_dissim['spearman_local_dissim_vs_sel_b']:.4f}"
                )
                emit_json(
                    {
                        "event": "rank_agreement_otprune_reconot",
                        "sample_idx": si,
                        "file_identifier": sample.file_identifier,
                        "keep_ratio": kr,
                        **rank_recon,
                    }
                )

            for ma, mb in PAIR_FOCUS:
                if ma not in primary or mb not in primary:
                    continue
                ov = _set_overlap_metrics(primary[ma], primary[mb])
                tok_ov = _token_id_overlap(primary[ma].indices, primary[mb].indices, token_ids)
                row = {**ov, **tok_ov}
                agg_pair[(ma, mb, kr)].append(row)

                log.line(f"  [pair {ma} vs {mb}]")
                for k, v in sorted(row.items()):
                    log.line(f"      {k}={v:.6g}")
                if ma == "loco3d" and mb == "random":
                    log.line(
                        f"      NOTE: loco3d-vs-random jaccard≈rand_expected "
                        f"{ov['rand_indep_expected_jaccard']:.4g} "
                        f"delta={ov['jaccard_minus_rand_expected']:.4g} "
                        f"(接近 0 表示与 random 难以区分)"
                    )
                if ma == "otprune" and mb == "random":
                    log.line(
                        f"      NOTE: otprune 应显著高于 random 期望 Jaccard "
                        f"(delta={ov['jaccard_minus_rand_expected']:.4g})"
                    )
                if ma == "reconot" and mb == "random":
                    log.line(
                        f"      NOTE: reconot 应显著高于 random 期望 Jaccard "
                        f"(delta={ov['jaccard_minus_rand_expected']:.4g})"
                    )
                if ma == "otprune" and mb == "reconot":
                    log.line(
                        f"      NOTE: otprune-vs-reconot jaccard={ov['jaccard']:.4g} "
                        f"(reconot 在 otprune 多样性核上叠加重构误差 q 权重)"
                    )

                emit_json(
                    {
                        "event": "pairwise",
                        "sample_idx": si,
                        "file_identifier": sample.file_identifier,
                        "keep_ratio": kr,
                        "method_a": ma,
                        "method_b": mb,
                        **row,
                    }
                )

            # loco3d vs otprune：仅在 ot 保留、loco 丢弃 的 token 上的误差对比
            if "loco3d" in primary and "otprune" in primary:
                m_l = primary["loco3d"].mask
                m_o = primary["otprune"].mask
                only_ot = np.logical_and(m_o, ~m_l)
                only_loco = np.logical_and(m_l, ~m_o)
                both = np.logical_and(m_l, m_o)
                log.line("  [exclusive regions: embed_neighbor_err / loco3d_entropy]")
                for label, mask in (
                    ("both_kept", both),
                    ("only_otprune", only_ot),
                    ("only_loco3d", only_loco),
                ):
                    if mask.any():
                        log.line(
                            f"      {label}: count={int(mask.sum())} "
                            f"mean_embed_err={float(err_np[mask].mean()):.6g} "
                            f"mean_entropy={float(entropy_np[mask].mean()):.6g}"
                        )
                emit_json(
                    {
                        "event": "exclusive_regions",
                        "sample_idx": si,
                        "file_identifier": sample.file_identifier,
                        "keep_ratio": kr,
                        "both_kept_mean_embed_err": float(err_np[both].mean()) if both.any() else None,
                        "only_otprune_mean_embed_err": float(err_np[only_ot].mean()) if only_ot.any() else None,
                        "only_loco3d_mean_embed_err": float(err_np[only_loco].mean()) if only_loco.any() else None,
                        "both_kept_mean_entropy": float(entropy_np[both].mean()) if both.any() else None,
                        "only_otprune_mean_entropy": float(entropy_np[only_ot].mean()) if only_ot.any() else None,
                        "only_loco3d_mean_entropy": float(entropy_np[only_loco].mean()) if only_loco.any() else None,
                    }
                )

            # otprune vs reconot：独占区域对比（两者均为 embedding 选择）
            if "otprune" in primary and "reconot" in primary:
                m_o = primary["otprune"].mask
                m_r = primary["reconot"].mask
                only_ot = np.logical_and(m_o, ~m_r)
                only_recon = np.logical_and(m_r, ~m_o)
                both = np.logical_and(m_o, m_r)
                log.line("  [exclusive regions otprune vs reconot: embed_neighbor_err]")
                for label, mask in (
                    ("both_kept", both),
                    ("only_otprune", only_ot),
                    ("only_reconot", only_recon),
                ):
                    if mask.any():
                        log.line(
                            f"      {label}: count={int(mask.sum())} "
                            f"mean_embed_err={float(err_np[mask].mean()):.6g}"
                        )
                emit_json(
                    {
                        "event": "exclusive_regions_otprune_reconot",
                        "sample_idx": si,
                        "file_identifier": sample.file_identifier,
                        "keep_ratio": kr,
                        "both_kept_mean_embed_err": float(err_np[both].mean()) if both.any() else None,
                        "only_otprune_mean_embed_err": float(err_np[only_ot].mean()) if only_ot.any() else None,
                        "only_reconot_mean_embed_err": float(err_np[only_recon].mean()) if only_recon.any() else None,
                    }
                )

    # 汇总
    log.line("")
    log.line("=" * 88)
    log.line("AGGREGATE (mean over samples)")
    log.line("=" * 88)
    for (ma, mb, kr), rows in sorted(agg_pair.items()):
        log.line(f"pair {ma} vs {mb} @ kr={kr}:")
        for key in (
            "jaccard",
            "jaccard_minus_rand_expected",
            "dice",
            "overlap_coef",
            "token_id_jaccard",
            "symmetric_diff",
        ):
            log.line(f"    mean_{key}={_aggregate_mean(rows, key):.6g}")
        emit_json(
            {
                "event": "aggregate_pair",
                "method_a": ma,
                "method_b": mb,
                "keep_ratio": kr,
                "n_samples": len(rows),
                **{f"mean_{k}": _aggregate_mean(rows, k) for k in rows[0].keys()},
            }
        )

    for (name, kr), rows in sorted(agg_method.items()):
        log.line(f"method {name} @ kr={kr}:")
        for key in (
            "mean_neighbor_err_kept",
            "oracle_high_neighbor_err_recall",
            "mean_one_minus_cos_disc_to_kept",
            "kept_surface_fraction",
            "surface_recall_vs_all_surface",
            "grid_boundary_fraction",
        ):
            if rows and key in rows[0]:
                log.line(f"    mean_{key}={_aggregate_mean(rows, key):.6g}")

    log.line("")
    log.line(f"# finished={_utc_now()}")
    log.close()
    jsonl_f.close()
    print(f"\nDone. Log: {log_path}\nJSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
