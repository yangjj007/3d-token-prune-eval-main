# -*- coding: utf-8 -*-
"""
方案三 v21：Spatial3D-LOCO 质量加权 DPP 修剪器（Quality-weighted DPP）

- 多样性核 K = I + γ·GGᵀ（与 otprune 同构，G 为 L2 归一化 VQ embedding）。
- 质量先验 q_i：占据(surface>filled>empty) + 高邻接熵(边界) + 稀有 token-id。
- 选择核 L_ij = q_i · q_j · K_ij，greedy MAP (DPP) 选 k 个。
- q≡1 时退化为 otprune；无 embedding 时用坐标 FPS + q 引导。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings, target_keep_count
from eval.baseline.otprune import greedy_select
from eval.pruners import BasePruner, register_pruner
from eval.proposed._diag import boundary_interior_counts, tensor_score_stats
from eval.proposed._logging import get_pruner_logger, summarize_vec
from eval.proposed._spatial import (
    GRID_X,
    GRID_Y,
    GRID_Z,
    all_coords_tensor,
    flat_index_to_coord,
    latent_surface_mask,
)

# 六个方向的邻居偏移
_NEIGHBOR_OFFSETS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]

# 默认超参（可被 configs/eval/loco3d.json 覆盖）
_DEFAULT_GAMMA = 0.01
_DEFAULT_EPSILON = 1e-10
_DEFAULT_EMPTY_QUALITY_SCALE = 0.08
_DEFAULT_SURFACE_QUALITY = 1.0
_DEFAULT_FILLED_QUALITY = 0.55
_DEFAULT_ENTROPY_WEIGHT = 0.35
_DEFAULT_RARITY_WEIGHT = 0.25
_DEFAULT_USE_EMBEDDING = True
_DEFAULT_MIN_QUALITY = 1e-4
_DEFAULT_QUALITY_MODE = "spatial"
_DEFAULT_CANDIDATE_MODE = "all"


def _compute_neighbor_entropy(
    token_ids: torch.Tensor,
    is_surface: torch.Tensor | None = None,
    is_empty: torch.Tensor | None = None,
) -> torch.Tensor:
    """为每个位置计算 6-邻域 token id 分布熵 [1024] float32 CPU。

    ``is_surface`` / ``is_empty`` 保留以兼容 compare 脚本，不参与计算。
    """
    tokens = token_ids.detach().cpu().numpy().astype(np.int64)
    coord_to_idx: Dict[Tuple[int, int, int], int] = {}
    for i in range(MESH_SEQ_LEN):
        coord_to_idx[flat_index_to_coord(i)] = i

    neighbor_counts = [defaultdict(int) for _ in range(MESH_SEQ_LEN)]
    total_degree = np.zeros(MESH_SEQ_LEN, dtype=np.int32)

    for i in range(MESH_SEQ_LEN):
        x, y, z = flat_index_to_coord(i)
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            nx, ny, nz = x + dx, y + dy, z + dz
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

    return torch.from_numpy(entropy)


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    """Min-max 归一化到 [0,1]；常数向量返回 0.5。"""
    x = x.float()
    lo = float(x.min().item())
    hi = float(x.max().item())
    if hi - lo < 1e-12:
        return torch.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def _compute_token_rarity(token_ids: torch.Tensor) -> torch.Tensor:
    """逆频次稀有度 [1024]，越高表示该 token-id 在本样本越稀有。"""
    t = token_ids.detach().long().view(-1).cpu()
    freq_map = Counter(int(x) for x in t.tolist())
    freq = torch.tensor([float(freq_map[int(x)]) for x in t], dtype=torch.float32)
    return 1.0 / freq.clamp_min(1.0)


def _compute_quality_scores(
    token_ids: torch.Tensor,
    is_surface: torch.Tensor,
    is_empty: torch.Tensor,
    is_filled: torch.Tensor,
    neighbor_entropy: torch.Tensor,
    *,
    empty_quality_scale: float,
    surface_quality: float,
    filled_quality: float,
    entropy_weight: float,
    rarity_weight: float,
    min_quality: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    q_i = base_occ + entropy_weight * entropy_norm + rarity_weight * rarity_norm，clamp 到正数。
    """
    device = neighbor_entropy.device
    base_occ = torch.full((MESH_SEQ_LEN,), float(empty_quality_scale), dtype=torch.float32, device=device)
    base_occ[is_surface] = float(surface_quality)
    base_occ[is_filled] = float(filled_quality)

    ent_norm = _normalize01(neighbor_entropy)
    rarity = _compute_token_rarity(token_ids).to(device)
    rarity_norm = _normalize01(rarity)

    q = base_occ + float(entropy_weight) * ent_norm + float(rarity_weight) * rarity_norm
    q = q.clamp(min=float(min_quality))

    occ_stats = {
        "surface_mean_q": float(q[is_surface].mean().item()) if is_surface.any() else 0.0,
        "filled_mean_q": float(q[is_filled].mean().item()) if is_filled.any() else 0.0,
        "empty_mean_q": float(q[is_empty].mean().item()) if is_empty.any() else 0.0,
    }
    q_stats = tensor_score_stats(q)
    ent_stats = tensor_score_stats(neighbor_entropy)
    rarity_stats = tensor_score_stats(rarity_norm)

    detail = {
        "quality": q_stats,
        "neighbor_entropy": ent_stats,
        "rarity_norm": rarity_stats,
        "occupancy_quality": occ_stats,
        "entropy_weight": float(entropy_weight),
        "rarity_weight": float(rarity_weight),
        "empty_quality_scale": float(empty_quality_scale),
    }
    return q, detail


def _build_diversity_kernel(feats: Tensor, gamma: float) -> Tensor:
    """K = I + γ·(G G^T)²，G 为行 L2 归一化 embedding。"""
    device = feats.device
    dtype = feats.dtype
    n = feats.size(0)
    g = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gram = g @ g.t()
    eye = torch.eye(n, device=device, dtype=dtype)
    return eye + float(gamma) * (gram @ gram.t())


def _quality_weighted_dpp_select(
    kernel_matrix: Tensor,
    quality: Tensor,
    max_length: int,
    epsilon: float,
) -> List[int]:
    """L = diag(q) @ K @ diag(q)，greedy MAP；不足则按 q 降序补足。"""
    q = quality.to(device=kernel_matrix.device, dtype=kernel_matrix.dtype).clamp_min(1e-12)
    l_kernel = q.unsqueeze(1) * kernel_matrix * q.unsqueeze(0)

    selected_items = greedy_select(kernel_matrix=l_kernel, max_length=max_length, epsilon=epsilon)
    selected_set = set(selected_items)
    km = l_kernel.clone()

    while len(selected_set) < max_length:
        for idx in selected_set:
            km[idx, :] = 0
            km[:, idx] = 0
        remain = max_length - len(selected_set)
        if remain <= 0:
            break
        new_items = greedy_select(kernel_matrix=km, max_length=remain, epsilon=epsilon)
        if not new_items:
            break
        selected_set.update(new_items)

    if len(selected_set) < max_length:
        remaining = [i for i in range(MESH_SEQ_LEN) if i not in selected_set]
        if remaining:
            rem_q = quality[remaining]
            order = torch.argsort(rem_q, descending=True)
            for j in order.tolist():
                selected_set.add(remaining[j])
                if len(selected_set) >= max_length:
                    break

    return sorted(selected_set)[:max_length]


def _quality_weighted_dpp_select_mapped(
    feats: Tensor,
    quality: Tensor,
    candidate_indices: torch.Tensor,
    max_length: int,
    gamma: float,
    epsilon: float,
) -> List[int]:
    """Run quality-weighted DPP on candidate rows, then map back to flat indices."""
    candidate_indices = candidate_indices.detach().long().view(-1).cpu()
    if candidate_indices.numel() == 0 or max_length <= 0:
        return []
    candidate_indices = candidate_indices.unique(sorted=True)
    k = max(1, min(int(max_length), int(candidate_indices.numel())))
    cand_dev = candidate_indices.to(feats.device)
    sub_feats = feats.index_select(0, cand_dev)
    sub_quality = quality.index_select(0, candidate_indices).to(feats.device)
    sub_kernel = _build_diversity_kernel(sub_feats, gamma)
    rel = _quality_weighted_dpp_select(sub_kernel, sub_quality, k, epsilon)
    mapped = candidate_indices[torch.tensor(rel, dtype=torch.long)].tolist()
    return [int(x) for x in mapped]


def _select_embedding_dpp_with_candidates(
    feats: Tensor,
    quality: Tensor,
    is_empty: torch.Tensor,
    max_length: int,
    *,
    gamma: float,
    epsilon: float,
    candidate_mode: str,
) -> List[int]:
    """Select from all tokens or prefer non-empty latent cells before filling."""
    if candidate_mode not in {"nonempty", "nonempty_only"}:
        k_mat = _build_diversity_kernel(feats, gamma)
        return _quality_weighted_dpp_select(k_mat, quality.to(feats.device), max_length, epsilon)

    nonempty = torch.nonzero(~is_empty.cpu(), as_tuple=False).view(-1)
    if nonempty.numel() >= max_length:
        return sorted(
            _quality_weighted_dpp_select_mapped(
                feats, quality.cpu(), nonempty, max_length, gamma, epsilon
            )
        )

    selected = set(int(i) for i in nonempty.tolist())
    remaining = torch.tensor(
        [i for i in range(MESH_SEQ_LEN) if i not in selected],
        dtype=torch.long,
    )
    need = max_length - len(selected)
    if need > 0 and remaining.numel() > 0:
        fill = _quality_weighted_dpp_select_mapped(
            feats, quality.cpu(), remaining, need, gamma, epsilon
        )
        selected.update(fill)
    return sorted(selected)[:max_length]


def _fps_select_with_quality(
    coords: Tensor,
    quality: Tensor,
    max_length: int,
) -> List[int]:
    """坐标最远点采样，首点取 q 最大；后续选 min_dist * q 最大。"""
    n = coords.size(0)
    max_length = max(1, min(max_length, n))
    q = quality.float().cpu()
    c = coords.float().cpu()

    first = int(torch.argmax(q).item())
    selected = [first]
    min_dist = torch.norm(c - c[first : first + 1], dim=1)

    while len(selected) < max_length:
        score = min_dist.clone()
        for idx in selected:
            score = torch.minimum(score, torch.norm(c - c[idx : idx + 1], dim=1))
        score = score * q
        score[selected] = -1.0
        nxt = int(torch.argmax(score).item())
        if score[nxt] < 0:
            break
        selected.append(nxt)
        min_dist = torch.minimum(min_dist, torch.norm(c - c[nxt : nxt + 1], dim=1))

    if len(selected) < max_length:
        remaining = [i for i in range(n) if i not in set(selected)]
        order = torch.argsort(q[remaining], descending=True)
        for j in order.tolist():
            selected.append(remaining[j])
            if len(selected) >= max_length:
                break

    return sorted(selected)[:max_length]

@register_pruner("loco3d")
class Loco3DPruner(BasePruner):
    """v21: 质量加权 DPP（几何先验 + embedding 多样性）。"""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t = token_ids.detach().long().view(-1)
        assert t.numel() == MESH_SEQ_LEN
        k_target = target_keep_count(self.keep_ratio)

        extra = self.extra
        gamma = float(extra.get("gamma", _DEFAULT_GAMMA))
        epsilon = float(extra.get("epsilon", _DEFAULT_EPSILON))
        empty_quality_scale = float(extra.get("empty_quality_scale", _DEFAULT_EMPTY_QUALITY_SCALE))
        surface_quality = float(extra.get("surface_quality", _DEFAULT_SURFACE_QUALITY))
        filled_quality = float(extra.get("filled_quality", _DEFAULT_FILLED_QUALITY))
        entropy_weight = float(extra.get("entropy_weight", _DEFAULT_ENTROPY_WEIGHT))
        rarity_weight = float(extra.get("rarity_weight", _DEFAULT_RARITY_WEIGHT))
        use_embedding = bool(extra.get("use_embedding", _DEFAULT_USE_EMBEDDING))
        min_quality = float(extra.get("min_quality", _DEFAULT_MIN_QUALITY))
        quality_mode = str(extra.get("quality_mode", _DEFAULT_QUALITY_MODE)).lower()
        candidate_mode = str(extra.get("candidate_mode", _DEFAULT_CANDIDATE_MODE)).lower()
        if quality_mode not in {"spatial", "flat"}:
            quality_mode = _DEFAULT_QUALITY_MODE
        if candidate_mode not in {"all", "nonempty", "nonempty_only"}:
            candidate_mode = _DEFAULT_CANDIDATE_MODE

        device = torch.device("cpu")
        if voxel_grid is not None:
            is_surface, is_empty, is_filled, _ = latent_surface_mask(voxel_grid)
            is_surface = is_surface.to(device)
            is_empty = is_empty.to(device)
            is_filled = is_filled.to(device)
        else:
            is_surface = torch.ones(MESH_SEQ_LEN, dtype=torch.bool, device=device)
            is_empty = torch.zeros(MESH_SEQ_LEN, dtype=torch.bool, device=device)
            is_filled = torch.zeros(MESH_SEQ_LEN, dtype=torch.bool, device=device)

        log_tag = str(kwargs.get("_log_tag", ""))
        log_kr = kwargs.get("_log_keep_ratio", self.keep_ratio)
        logger = get_pruner_logger("loco3d")
        logger.info(f"prune_start kr={float(log_kr):.4g} tag={log_tag} k_target={k_target}")

        neighbor_entropy = _compute_neighbor_entropy(t)
        ent_stats_all = summarize_vec(neighbor_entropy)

        q, quality_detail = _compute_quality_scores(
            t,
            is_surface,
            is_empty,
            is_filled,
            neighbor_entropy,
            empty_quality_scale=empty_quality_scale,
            surface_quality=surface_quality,
            filled_quality=filled_quality,
            entropy_weight=entropy_weight,
            rarity_weight=rarity_weight,
            min_quality=min_quality,
        )
        if quality_mode == "flat":
            q = torch.ones_like(q).clamp(min=float(min_quality))
            quality_detail = {
                "quality_mode": "flat",
                "note": "q is constant; selection reduces to OTPrune-style embedding DPP.",
            }
        else:
            quality_detail["quality_mode"] = "spatial"

        vq_emb = kwargs.get("vq_embeddings")
        selection_mode = "quality_weighted_dpp"
        kernel_diag_mean = float("nan")

        if use_embedding and vq_emb is not None:
            feats = gather_embeddings(vq_emb, t)
            final_indices = _select_embedding_dpp_with_candidates(
                feats,
                q,
                is_empty,
                k_target,
                gamma=gamma,
                epsilon=epsilon,
                candidate_mode=candidate_mode,
            )
            selection_mode = (
                f"{quality_mode}_quality_{candidate_mode}_candidate_dpp"
            )
            k_mat_diag = _build_diversity_kernel(feats[:1], gamma)
            kernel_diag_mean = float(torch.diag(k_mat_diag).mean().item())
        else:
            selection_mode = "fps_quality_fallback"
            coords = all_coords_tensor(device=torch.device("cpu")).float()
            final_indices = _fps_select_with_quality(coords, q, k_target)

        idx_t = torch.tensor(final_indices, dtype=torch.long)
        pruned = t.index_select(0, idx_t)

        nonempty_count = int((~is_empty).sum().item())
        b_kept, i_kept = boundary_interior_counts(final_indices, surface_mask=is_surface)
        b_shell, i_shell = boundary_interior_counts(final_indices)

        kept_s = sum(1 for i in final_indices if bool(is_surface[i].item()))
        kept_fill = sum(
            1
            for i in final_indices
            if bool((~is_surface[i]).item() and (~is_empty[i]).item())
        )
        kept_e = sum(1 for i in final_indices if bool(is_empty[i].item()))

        voxel_surface_share = float(is_surface.float().mean().item())
        token_top3_share = self._token_top3_share(t)

        q_kept = q[final_indices]
        ent_kept = neighbor_entropy[final_indices]
        q_kept_stats = summarize_vec(q_kept)
        ent_kept_stats = summarize_vec(ent_kept)

        logger.info(
            f"prune_done kr={float(log_kr):.4g} tag={log_tag} mode={selection_mode} "
            f"voxel_surf_share={voxel_surface_share:.4g} token_top3_share={token_top3_share:.4g} "
            f"nonempty_total={nonempty_count} "
            f"kept(surf/fill/empty)={kept_s}/{kept_fill}/{kept_e} "
            f"q_kept_mean={q_kept_stats['mean']:.4g} ent_kept_mean={ent_kept_stats['mean']:.4g} "
            f"kept(surf/other)={b_kept}/{i_kept} shell_kept(B/I)={b_shell}/{i_shell}"
        )

        meta: Dict[str, Any] = {
            "method": "loco3d",
            "version": "v21_quality_weighted_dpp",
            "k": int(pruned.numel()),
            "indices": final_indices,
            "selection_mode": selection_mode,
            "gamma": gamma,
            "diagnostics": {
                "voxel_surface_share": voxel_surface_share,
                "token_id_top3_share": token_top3_share,
                "kept_surface": kept_s,
                "kept_fill": kept_fill,
                "kept_empty": kept_e,
                "entropy_stats": {
                    "mean": float(ent_stats_all["mean"]),
                    "std": float(ent_stats_all["std"]),
                    "median": float(ent_stats_all["p50"]),
                },
                "quality_kept_stats": {
                    "mean": float(q_kept_stats["mean"]),
                    "std": float(q_kept_stats["std"]),
                    "median": float(q_kept_stats["p50"]),
                },
                "entropy_kept_stats": {
                    "mean": float(ent_kept_stats["mean"]),
                    "std": float(ent_kept_stats["std"]),
                    "median": float(ent_kept_stats["p50"]),
                },
                "quality_detail": quality_detail,
                "kernel_diag_mean": kernel_diag_mean,
                "use_embedding": use_embedding,
                "quality_mode": quality_mode,
                "candidate_mode": candidate_mode,
            },
        }
        return pruned, meta

    @staticmethod
    def _token_top3_share(t: torch.Tensor) -> float:
        c = Counter(int(x) for x in t.view(-1).tolist())
        if not c:
            return 0.0
        top3 = sum(n for _, n in c.most_common(3))
        return float(top3) / float(max(t.numel(), 1))


@register_pruner("loco3d_dpp")
class Loco3DFlatDPPPruner(Loco3DPruner):
    """Ablation: q is constant, so loco3d uses the OTPrune-style DPP backbone."""

    def __init__(self, keep_ratio: float = 1.0, seed: int = 42, **kwargs: Any):
        kwargs.setdefault("quality_mode", "flat")
        kwargs.setdefault("candidate_mode", "all")
        super().__init__(keep_ratio=keep_ratio, seed=seed, **kwargs)


@register_pruner("loco3d_nonempty_dpp")
class Loco3DNonEmptyDPPPruner(Loco3DPruner):
    """Ablation: OTPrune-style DPP restricted to non-empty latent cells when possible."""

    def __init__(self, keep_ratio: float = 1.0, seed: int = 42, **kwargs: Any):
        kwargs.setdefault("quality_mode", "flat")
        kwargs.setdefault("candidate_mode", "nonempty")
        super().__init__(keep_ratio=keep_ratio, seed=seed, **kwargs)
