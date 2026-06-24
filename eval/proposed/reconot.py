# -*- coding: utf-8 -*-
"""
ReconOT：重构误差引导的 OT/DPP 纯 VQ-Embedding 剪枝器。

v4：单一 `_select_to_k` 流水线 — rank 引导（DPP 小池 / OT Sinkhorn）+
自适应 per-id cap + 在线多样性，恒凑满 k_target。
v4_perf：向量化 cap 屏蔽、跨 keep_ratio 中间结果缓存、可选 fast 诊断。
v4_perf2：低秩 Gram 核、扩展 mesh 缓存、select/DPP 热路径减量（与 v4_perf 逐位一致）。
v4_perf3：cap 二分跳跃、sim 预计算、FPS 平方距、可选 prune_device GPU（与 v4_perf2 逐位一致）。
v4_perf4：DPP 复用 f_norm、L-kernel outer 融合（与 v4_perf3 逐位一致）。
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings, require_vq_embeddings, target_keep_count
from eval.baseline.otprune import greedy_select as _greedy_select_dpp
from eval.pruners import BasePruner, register_pruner
from eval.proposed._logging import get_pruner_logger, summarize_vec

# 默认超参（可被 configs/eval/reconot.json 覆盖）
_DEFAULT_GAMMA = 0.01
_DEFAULT_EPSILON = 1e-10
_DEFAULT_N_BASIS = 64
_DEFAULT_RIDGE_LAMBDA = 1e-3
_DEFAULT_RECON_WEIGHT = 0.5
_DEFAULT_RARITY_WEIGHT = 0.25
_DEFAULT_SELECTOR = "dpp"
_DEFAULT_SINKHORN_EPS = 0.05
_DEFAULT_SINKHORN_ITERS = 80
_DEFAULT_CODEBOOK_ACCELERATE = True
_DEFAULT_OT_PER_ID_CAP = 1
_DEFAULT_FAST_DIAGNOSTICS = True
_DEFAULT_PRUNE_DEVICE = "cpu"

_MESH_SCORE_CACHE: Dict[str, Dict[str, Any]] = {}
_MESH_CACHE_ORDER: List[str] = []
_MESH_CACHE_MAX = 8


def _mesh_cache_key(tag: str, n_basis: int, ridge_lambda: float) -> str:
    if not tag:
        return ""
    return f"{tag}|{n_basis}|{ridge_lambda:g}"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    return _MESH_SCORE_CACHE.get(key)


def _cache_put(key: str, entry: Dict[str, Any]) -> None:
    if not key:
        return
    if key in _MESH_SCORE_CACHE:
        _MESH_CACHE_ORDER.remove(key)
    elif len(_MESH_CACHE_ORDER) >= _MESH_CACHE_MAX:
        old = _MESH_CACHE_ORDER.pop(0)
        _MESH_SCORE_CACHE.pop(old, None)
    _MESH_SCORE_CACHE[key] = entry
    _MESH_CACHE_ORDER.append(key)


def clear_mesh_score_cache() -> None:
    """清空跨 keep_ratio mesh 缓存（parity / 单测用）。"""
    _MESH_SCORE_CACHE.clear()
    _MESH_CACHE_ORDER.clear()


def _resolve_prune_device(extra: Dict[str, Any]) -> torch.device:
    raw = str(extra.get("prune_device", _DEFAULT_PRUNE_DEVICE)).strip().lower()
    if raw in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(raw)


def _kr_adaptive_weights(
    keep_ratio: float,
    recon_weight: float,
    rarity_weight: float,
    gamma: float,
) -> Tuple[float, float, float]:
    """按 keep_ratio 微调 effective 权重（不暴露为独立超参）。"""
    kr = float(keep_ratio)
    recon_eff = recon_weight * (0.6 + 0.8 * kr)
    rarity_eff = rarity_weight * (0.4 + 1.2 * kr)
    gamma_eff = gamma * (1.5 - kr)
    return recon_eff, rarity_eff, gamma_eff


def _normalize01(x: Tensor) -> Tensor:
    """Min-max 归一化到 [0,1]；常数向量返回 0.0（不引入额外重要性）。"""
    x = x.float()
    lo = float(x.min().item())
    hi = float(x.max().item())
    if hi - lo < 1e-12:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def _embedding_fps(feats: Tensor, n_basis: int) -> List[int]:
    """embedding 空间最远点采样，选 n_basis 个语义锚点（起点固定 0，保证确定性）。"""
    n = feats.size(0)
    n_basis = max(1, min(n_basis, n))
    f = feats.float()
    selected_mask = torch.zeros(n, dtype=torch.bool, device=feats.device)
    first = 0
    selected_mask[first] = True
    selected = [first]
    min_sq = torch.sum((f - f[first : first + 1]) ** 2, dim=1)
    min_sq = min_sq.masked_fill(selected_mask, -1.0)
    while len(selected) < n_basis:
        nxt = int(torch.argmax(min_sq).item())
        if selected_mask[nxt]:
            break
        selected.append(nxt)
        selected_mask[nxt] = True
        min_sq = torch.minimum(min_sq, torch.sum((f - f[nxt : nxt + 1]) ** 2, dim=1))
        min_sq = min_sq.masked_fill(selected_mask, -1.0)
    return selected


def _linear_reconstruct_with_basis(
    tokens: Tensor,
    basis: Tensor,
    ridge_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """全局线性重构：tokens [B,N,D], basis [B,K,D] -> recon, errors [B,N]。"""
    tokens_f = tokens.float()
    basis_f = basis.float()
    g = torch.matmul(basis_f, basis_f.transpose(-1, -2))
    g.diagonal(dim1=-2, dim2=-1).add_(float(ridge_lambda))
    b_rhs = torch.matmul(basis_f, tokens_f.transpose(-1, -2))
    w = torch.linalg.solve(g, b_rhs).transpose(-1, -2)
    recon = torch.matmul(w, basis_f)
    errors = torch.norm(tokens_f - recon, dim=-1)
    return recon, errors


def _recon_error_basis_fps(
    feats: Tensor,
    n_basis: int,
    ridge_lambda: float,
) -> Tuple[Tensor, List[int]]:
    """FPS basis + 全局线性重构残差 [N]。"""
    n, d = feats.size(0), feats.size(1)
    n_basis = max(1, min(n_basis, n))
    basis_idx = _embedding_fps(feats, n_basis)
    x = feats.unsqueeze(0)
    idx_t = torch.tensor(basis_idx, dtype=torch.long, device=feats.device).view(1, -1)
    idx_exp = idx_t.unsqueeze(-1).expand(-1, -1, d)
    basis = x.gather(1, idx_exp)
    _, errors = _linear_reconstruct_with_basis(x, basis, ridge_lambda)
    return errors[0], basis_idx


def _codebook_rarity(token_ids: Tensor) -> Tensor:
    """逆频次稀有度 [N]：rarity_i = 1/freq(id_i)。"""
    t = token_ids.detach().long().view(-1)
    max_id = int(t.max().item()) + 1
    counts = torch.bincount(t, minlength=max_id).float()
    return (1.0 / counts[t].clamp_min(1.0))


def _build_quality(
    recon_error: Tensor,
    rarity: Tensor,
    *,
    recon_weight: float,
    rarity_weight: float,
    fast_diagnostics: bool = True,
    err_norm: Optional[Tensor] = None,
    rar_norm: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """q_i = 1 + recon_weight·norm(error) + rarity_weight·norm(rarity)。"""
    if err_norm is None:
        err_norm = _normalize01(recon_error)
    if rar_norm is None:
        rar_norm = _normalize01(rarity)
    q = 1.0 + float(recon_weight) * err_norm + float(rarity_weight) * rar_norm
    q = q.clamp_min(1e-6)
    if fast_diagnostics:
        detail = {
            "recon_weight": float(recon_weight),
            "rarity_weight": float(rarity_weight),
            "quality_mean": float(q.mean().item()),
        }
    else:
        detail = {
            "recon_error": summarize_vec(recon_error),
            "recon_error_norm": summarize_vec(err_norm),
            "rarity_norm": summarize_vec(rar_norm),
            "quality": summarize_vec(q),
            "recon_weight": float(recon_weight),
            "rarity_weight": float(rarity_weight),
        }
    return q, detail


def _diversity_kernel_from_norm(g: Tensor, gamma: float) -> Tensor:
    """K = I + γ·(G Gᵀ)²；g 已 L2 归一化。"""
    h = g.t() @ g
    gram_sq = (g @ h) @ g.t()
    eye = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
    return eye + float(gamma) * gram_sq


def _diversity_kernel(feats: Tensor, gamma: float) -> Tensor:
    """K = I + γ·(G Gᵀ)²；经 g(gᵀg)gᵀ 低秩等价实现。"""
    g = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return _diversity_kernel_from_norm(g, gamma)


def _greedy_fill(
    kernel_matrix: Tensor,
    max_length: int,
    epsilon: float,
    *,
    stats: Optional[Dict[str, int]] = None,
) -> List[int]:
    """在给定核上 greedy MAP 选 max_length 个（工作矩阵 in-place 置零已选行列）。"""
    n = kernel_matrix.size(0)
    max_length = min(max_length, n)
    km = kernel_matrix.clone()
    fill_rounds = 0
    selected_items = _greedy_select_dpp(kernel_matrix=km, max_length=max_length, epsilon=epsilon)
    selected_set = set(selected_items)
    while len(selected_set) < max_length:
        fill_rounds += 1
        prev_len = len(selected_set)
        for idx in selected_set:
            km[idx, :] = 0
            km[:, idx] = 0
        remain = max_length - len(selected_set)
        if remain <= 0:
            break
        new_items = _greedy_select_dpp(kernel_matrix=km, max_length=remain, epsilon=epsilon)
        if not new_items:
            break
        selected_set.update(new_items)
        if len(selected_set) == prev_len:
            break
    if stats is not None:
        stats["greedy_fill_rounds"] = stats.get("greedy_fill_rounds", 0) + fill_rounds
    return sorted(selected_set)[:max_length]


def _select_dpp_on_subset(
    feats: Tensor,
    quality: Tensor,
    k_pick: int,
    *,
    gamma: float,
    epsilon: float,
    f_norm: Optional[Tensor] = None,
) -> List[int]:
    """子集上 q 加权 DPP greedy MAP。"""
    n = feats.size(0)
    k_pick = min(k_pick, n)
    if f_norm is not None and f_norm.size(0) == n:
        k_mat = _diversity_kernel_from_norm(f_norm, gamma)
    else:
        k_mat = _diversity_kernel(feats, gamma)
    q = quality.clamp_min(1e-12)
    l_kernel = k_mat * torch.outer(q, q)
    selected_set = set(_greedy_fill(l_kernel, k_pick, epsilon))
    if len(selected_set) < k_pick:
        sel_mask = torch.ones(n, dtype=torch.bool, device=feats.device)
        if selected_set:
            sel_mask[list(selected_set)] = False
        remaining = torch.masked_select(
            torch.arange(n, device=feats.device), sel_mask
        )
        order = torch.argsort(quality[remaining], descending=True)
        for j in order.tolist():
            selected_set.add(int(remaining[j].item()))
            if len(selected_set) >= k_pick:
                break
    return sorted(selected_set)[:k_pick]


def _id_freq_counter(ids: Tensor) -> Counter:
    """从 token id 序列构建频次 Counter（无 tolist）。"""
    unique_ids, counts = ids.unique(return_counts=True)
    return Counter(
        {int(u.item()): int(c.item()) for u, c in zip(unique_ids, counts)}
    )


def _min_cap_for_k(k_eff: int, id_freq: Counter) -> int:
    """最小 cap 使 sum(min(freq[id], cap)) >= k_eff。"""
    if k_eff <= 0 or not id_freq:
        return 1
    freqs = list(id_freq.values())
    max_f = max(freqs)

    def selectable(c: int) -> int:
        return sum(min(f, c) for f in freqs)

    lo, hi = 1, max_f
    while lo < hi:
        mid = (lo + hi) // 2
        if selectable(mid) >= k_eff:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _dpp_rank_scores(
    feats: Tensor,
    q: Tensor,
    k_eff: int,
    *,
    gamma: float,
    epsilon: float,
    f_norm: Optional[Tensor] = None,
) -> Tensor:
    """小池 DPP → 全序列 rank_scores（DPP 选中位置获得额外加分）。"""
    n = feats.size(0)
    device = feats.device
    m = min(n, max(2 * k_eff, k_eff + 64, 384))
    cand = torch.topk(q, m).indices
    sub_feats = feats[cand]
    sub_q = q[cand]
    sub_f_norm = f_norm[cand] if f_norm is not None else None
    sub_k = min(int(cand.numel()), k_eff)
    sub_sel = _select_dpp_on_subset(
        sub_feats,
        sub_q,
        sub_k,
        gamma=gamma,
        epsilon=epsilon,
        f_norm=sub_f_norm,
    )
    global_sel = cand[sub_sel]

    rank_scores = torch.zeros(n, dtype=torch.float32, device=device)
    q_order = torch.argsort(q, descending=True)
    ranks = torch.arange(n, dtype=torch.float32, device=device)
    rank_scores[q_order] = (n - ranks) / max(n, 1)
    rank_scores[global_sel] = rank_scores[global_sel] + float(n)
    return rank_scores


def _sinkhorn_ot(
    cost: Tensor,
    mu: Tensor,
    nu: Tensor,
    *,
    eps: float,
    iters: int,
) -> Tensor:
    """熵正则 Sinkhorn，返回传输计划 π [n, m]。"""
    kmat = torch.exp(-cost / max(float(eps), 1e-8))
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(int(iters)):
        u = mu / (kmat @ v).clamp_min(1e-12)
        v = nu / (kmat.t() @ u).clamp_min(1e-12)
    return u.unsqueeze(1) * kmat * v.unsqueeze(0)


def _map_super_scores_to_positions(
    super_score: Tensor,
    inverse: Tensor,
    q: Tensor,
) -> Tensor:
    """super-token 分数映射到每 id 的 q 最大代表位置。"""
    n = inverse.numel()
    device = super_score.device
    dtype = super_score.dtype
    u = int(inverse.max().item()) + 1
    max_q = torch.full((u,), float("-inf"), device=q.device, dtype=q.dtype)
    max_q.scatter_reduce_(0, inverse, q, reduce="amax", include_self=False)
    is_rep = q >= (max_q[inverse] - 1e-7)
    pos = torch.arange(n, device=device, dtype=torch.long)
    cand_pos = torch.where(is_rep, pos, torch.full_like(pos, n))
    rep_idx = torch.full((u,), n, dtype=torch.long, device=device)
    rep_idx.scatter_reduce_(0, inverse, cand_pos, reduce="amin", include_self=False)
    valid = rep_idx < n
    row_score = torch.full((n,), float("-inf"), dtype=dtype, device=device)
    row_score[rep_idx[valid]] = super_score[valid]
    return row_score


def _ot_rank_scores(
    feats: Tensor,
    token_ids: Tensor,
    quality: Tensor,
    k_target: int,
    *,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
    codebook_accelerate: bool,
) -> Tensor:
    """Sinkhorn OT → rank_scores（薄封装，供 _select_to_k 使用）。"""
    n = feats.size(0)
    f = feats.float()
    q = quality.float().clamp_min(1e-12)
    ids = token_ids.detach().long().view(-1).to(f.device)

    if codebook_accelerate:
        unique_ids, inverse = torch.unique(ids, return_inverse=True)
        u = unique_ids.numel()
        super_feats = torch.zeros(u, f.size(1), dtype=f.dtype, device=f.device)
        super_feats.index_copy_(0, inverse, f)
        super_q = torch.zeros(u, dtype=f.dtype, device=f.device)
        super_q.index_add_(0, inverse, q)
        mu = super_q / super_q.sum().clamp_min(1e-12)
        m = min(k_target, u)
        nu = torch.full((m,), 1.0 / m, dtype=f.dtype)
        anchor = torch.topk(mu, m).indices
        cost = torch.cdist(super_feats, super_feats[anchor]) ** 2
        pi = _sinkhorn_ot(cost, mu, nu, eps=sinkhorn_eps, iters=sinkhorn_iters)
        super_score = pi.sum(dim=1)
        row_score = _map_super_scores_to_positions(super_score, inverse, q)
    else:
        mu = q / q.sum().clamp_min(1e-12)
        m = min(k_target, n)
        nu = torch.full((m,), 1.0 / m, dtype=f.dtype)
        anchor = torch.topk(mu, m).indices
        cost = torch.cdist(f, f[anchor]) ** 2
        pi = _sinkhorn_ot(cost, mu, nu, eps=sinkhorn_eps, iters=sinkhorn_iters)
        row_score = pi.sum(dim=1)

    rank_scores = row_score.clone()
    finite = torch.isfinite(rank_scores)
    if finite.any():
        lo = rank_scores[finite].min()
        rank_scores = rank_scores.masked_fill(~finite, lo - 1.0)
    else:
        rank_scores = q.clone()
    return rank_scores


def _priority_at_cap(
    wq: Tensor,
    max_cos: Tensor,
    selected_mask: Tensor,
    rs_bad: Tensor,
    inverse: Tensor,
    id_sel_count: Tensor,
    cap_eff: int,
    epsilon: float,
) -> Tensor:
    priority = wq * (1.0 - max_cos).clamp_min(epsilon)
    priority = priority.masked_fill(selected_mask, float("-inf"))
    priority = priority.masked_fill(rs_bad, float("-inf"))
    counts_pos = id_sel_count[inverse]
    priority = priority.masked_fill(counts_pos >= cap_eff, float("-inf"))
    return priority


def _next_feasible_cap(
    cap_eff: int,
    cap_ceiling: int,
    wq: Tensor,
    max_cos: Tensor,
    selected_mask: Tensor,
    rs_bad: Tensor,
    inverse: Tensor,
    id_sel_count: Tensor,
    epsilon: float,
) -> int:
    """大于 cap_eff 的最小 cap，使至少一个未选位置 priority 有限（等价于逐步 +1）。"""
    lo, hi = cap_eff + 1, cap_ceiling
    if lo > hi:
        return cap_eff
    if not torch.isfinite(
        _priority_at_cap(
            wq, max_cos, selected_mask, rs_bad, inverse, id_sel_count, hi, epsilon
        )
    ).any():
        return cap_ceiling
    while lo < hi:
        mid = (lo + hi) // 2
        if torch.isfinite(
            _priority_at_cap(
                wq, max_cos, selected_mask, rs_bad, inverse, id_sel_count, mid, epsilon
            )
        ).any():
            hi = mid
        else:
            lo = mid + 1
    return lo


def _select_to_k(
    feats: Tensor,
    token_ids: Tensor,
    q: Tensor,
    k_target: int,
    *,
    rank_scores: Tensor,
    per_id_cap: int,
    epsilon: float = 1e-12,
    f_norm: Optional[Tensor] = None,
    inverse: Optional[Tensor] = None,
    id_freq: Optional[Counter] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    rank 引导 + 在线多样性 + 自适应 per-id cap，恒输出 k_eff 个位置索引。
    cap 屏蔽经 inverse gather 向量化（语义与逐 id masked_fill 等价）。
    """
    n = feats.size(0)
    k_eff = min(k_target, n)
    ids = token_ids.detach().long().view(-1)
    device = feats.device
    if inverse is not None:
        inverse = inverse.to(device)

    if id_freq is None:
        id_freq = _id_freq_counter(ids)
    cap_needed = _min_cap_for_k(k_eff, id_freq)
    cap_ceiling = max(per_id_cap, cap_needed)

    if f_norm is None:
        f_norm = feats.float()
        f_norm = f_norm / f_norm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        f_norm = f_norm.float()

    if inverse is None:
        _, inverse = ids.unique(return_inverse=True)
    n_unique = int(inverse.max().item()) + 1
    id_sel_count = torch.zeros(n_unique, dtype=torch.long, device=device)

    rs = rank_scores.float()
    qf = q.float().clamp_min(epsilon)
    wq = rs * qf
    rs_bad = ~torch.isfinite(rs)
    max_cos = torch.zeros(n, dtype=torch.float32, device=device)
    selected: List[int] = []
    selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
    cap_eff = per_id_cap
    cap_start = per_id_cap
    sim = f_norm @ f_norm.t()
    cap_jump_count = 0
    select_steps = 0

    while len(selected) < k_eff:
        select_steps += 1
        priority = _priority_at_cap(
            wq, max_cos, selected_mask, rs_bad, inverse, id_sel_count, cap_eff, epsilon
        )

        best = int(torch.argmax(priority).item())
        if not torch.isfinite(priority[best]):
            if cap_eff < cap_ceiling:
                cap_jump_count += 1
                cap_eff = _next_feasible_cap(
                    cap_eff,
                    cap_ceiling,
                    wq,
                    max_cos,
                    selected_mask,
                    rs_bad,
                    inverse,
                    id_sel_count,
                    epsilon,
                )
                continue
            remaining = (~selected_mask).nonzero(as_tuple=False).view(-1)
            order = torch.argsort(qf[remaining], descending=True)
            for j in order.tolist():
                idx = int(remaining[j].item())
                selected.append(idx)
                selected_mask[idx] = True
                if len(selected) >= k_eff:
                    break
            break

        selected.append(best)
        selected_mask[best] = True
        id_sel_count[inverse[best]] += 1
        max_cos = torch.maximum(max_cos, sim[:, best])

    if len(selected) < k_eff:
        remaining = (~selected_mask).nonzero(as_tuple=False).view(-1)
        order = torch.argsort(qf[remaining], descending=True)
        for j in order.tolist():
            idx = int(remaining[j].item())
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= k_eff:
                break

    selected = sorted(selected)[:k_eff]
    assert len(selected) == k_eff, f"select_to_k: got {len(selected)} != {k_eff}"

    diag: Dict[str, Any] = {
        "k_target": int(k_target),
        "k_out": len(selected),
        "cap_needed": int(cap_needed),
        "cap_eff_final": int(cap_eff),
        "cap_relaxed": cap_eff > cap_start,
        "unique_ids_in_mesh": len(id_freq),
        "cap_jump_count": int(cap_jump_count),
        "select_steps": int(select_steps),
    }
    return selected, diag


def _mean_pairwise_cosine(feats: Tensor) -> float:
    if feats.size(0) < 2:
        return 0.0
    g = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = g @ g.t()
    n = g.size(0)
    off = (sim.sum() - torch.diag(sim).sum()) / (n * (n - 1))
    return float(off.item())


def _kl_all_vs_kept(token_ids_all: Tensor, token_ids_kept: Tensor) -> float:
    """KL(p_all || p_kept)。"""
    a = token_ids_all.detach().long().view(-1)
    b = token_ids_kept.detach().long().view(-1)
    ua, ca = a.unique(return_counts=True)
    ub, cb = b.unique(return_counts=True)
    cb_map = {int(u.item()): int(c.item()) for u, c in zip(ub, cb)}
    na, nb = float(a.numel()), float(b.numel())
    if na == 0 or nb == 0:
        return 0.0
    kl = 0.0
    for u, c in zip(ua, ca):
        pa = float(c.item()) / na
        pb = max(cb_map.get(int(u.item()), 0) / nb, 1e-9)
        kl += pa * (torch.log(torch.tensor(pa)) - torch.log(torch.tensor(pb))).item()
    return float(kl)


def _compute_mesh_scores(
    embed: Any,
    t: Tensor,
    *,
    recon_weight: float,
    n_basis: int,
    ridge_lambda: float,
    prune_device: torch.device | str | None = "cpu",
) -> Dict[str, Any]:
    """kr 无关：embed / recon / rarity / 静态张量（供跨 keep_ratio 缓存）。"""
    feats = gather_embeddings(embed, t, device=prune_device)
    dev = feats.device
    basis_idx: List[int] = []
    if recon_weight > 0.0:
        recon_error, basis_idx = _recon_error_basis_fps(feats, n_basis, ridge_lambda)
    else:
        recon_error = torch.zeros(feats.size(0), dtype=torch.float32, device=dev)
        basis_idx = _embedding_fps(feats, n_basis)
    rarity = _codebook_rarity(t).to(dev)
    err_norm = _normalize01(recon_error)
    rar_norm = _normalize01(rarity)
    f = feats.float()
    f_norm = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ids = t.detach().long().view(-1)
    _, inverse = ids.unique(return_inverse=True)
    id_freq = _id_freq_counter(ids)
    return {
        "feats": feats,
        "recon_error": recon_error,
        "rarity": rarity,
        "basis_idx": basis_idx,
        "err_norm": err_norm,
        "rar_norm": rar_norm,
        "f_norm": f_norm,
        "inverse": inverse,
        "id_freq": id_freq,
    }


@register_pruner("reconot")
class ReconOTPruner(BasePruner):
    """重构误差引导的 OT/DPP 纯 VQ-embedding 剪枝器（仅选择，不合并）。"""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        with torch.inference_mode():
            return self._prune_impl(token_ids, voxel_grid, **kwargs)

    def _prune_impl(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embed = require_vq_embeddings(kwargs.get("vq_embeddings"))
        t = token_ids.detach().long().view(-1)
        assert t.numel() == MESH_SEQ_LEN
        k_target = target_keep_count(self.keep_ratio)

        extra = self.extra
        gamma = float(extra.get("gamma", _DEFAULT_GAMMA))
        epsilon = float(extra.get("epsilon", _DEFAULT_EPSILON))
        n_basis = int(extra.get("n_basis", _DEFAULT_N_BASIS))
        ridge_lambda = float(extra.get("ridge_lambda", _DEFAULT_RIDGE_LAMBDA))
        recon_weight = float(extra.get("recon_weight", _DEFAULT_RECON_WEIGHT))
        rarity_weight = float(extra.get("rarity_weight", _DEFAULT_RARITY_WEIGHT))
        selector = str(extra.get("selector", _DEFAULT_SELECTOR)).lower()
        if selector not in ("dpp", "ot"):
            selector = "dpp"
        sinkhorn_eps = float(extra.get("sinkhorn_eps", _DEFAULT_SINKHORN_EPS))
        sinkhorn_iters = int(extra.get("sinkhorn_iters", _DEFAULT_SINKHORN_ITERS))
        codebook_accelerate = bool(extra.get("codebook_accelerate", _DEFAULT_CODEBOOK_ACCELERATE))
        per_id_cap = int(extra.get("per_id_cap", extra.get("ot_per_id_cap", _DEFAULT_OT_PER_ID_CAP)))
        per_id_cap = max(1, per_id_cap)
        fast_diagnostics = bool(extra.get("fast_diagnostics", _DEFAULT_FAST_DIAGNOSTICS))
        profile_prune = bool(extra.get("profile_prune", False))
        prune_device = _resolve_prune_device(extra)

        recon_weight, rarity_weight, gamma = _kr_adaptive_weights(
            self.keep_ratio, recon_weight, rarity_weight, gamma
        )

        log_tag = str(kwargs.get("_log_tag", ""))
        log_kr = kwargs.get("_log_keep_ratio", self.keep_ratio)
        logger = get_pruner_logger("reconot")
        logger.info(
            f"prune_start kr={float(log_kr):.4g} tag={log_tag} k_target={k_target} "
            f"selector={selector} fast_diagnostics={fast_diagnostics} per_id_cap={per_id_cap}"
        )

        prof: Dict[str, float] = {}
        t0 = time.perf_counter()

        cache_key = _mesh_cache_key(log_tag, n_basis, ridge_lambda)
        cached = _cache_get(cache_key)
        if cached is not None:
            feats = cached["feats"]
            recon_error = cached["recon_error"]
            rarity = cached["rarity"]
            basis_idx = cached["basis_idx"]
            err_norm = cached["err_norm"]
            rar_norm = cached["rar_norm"]
            f_norm = cached["f_norm"]
            inverse = cached["inverse"]
            id_freq = cached["id_freq"]
            prof["embed_gather"] = 0.0
            prof["recon_rarity"] = 0.0
        else:
            try:
                scores = _compute_mesh_scores(
                    embed,
                    t,
                    recon_weight=recon_weight,
                    n_basis=n_basis,
                    ridge_lambda=ridge_lambda,
                    prune_device=prune_device,
                )
            except RuntimeError as exc:
                if prune_device.type == "cuda" and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    prune_device = torch.device("cpu")
                    scores = _compute_mesh_scores(
                        embed,
                        t,
                        recon_weight=recon_weight,
                        n_basis=n_basis,
                        ridge_lambda=ridge_lambda,
                        prune_device=prune_device,
                    )
                else:
                    raise
            feats = scores["feats"]
            recon_error = scores["recon_error"]
            rarity = scores["rarity"]
            basis_idx = scores["basis_idx"]
            err_norm = scores["err_norm"]
            rar_norm = scores["rar_norm"]
            f_norm = scores["f_norm"]
            inverse = scores["inverse"]
            id_freq = scores["id_freq"]
            t1 = time.perf_counter()
            prof["embed_gather"] = t1 - t0
            prof["recon_rarity"] = 0.0
            _cache_put(cache_key, scores)

        t2 = time.perf_counter()
        q, quality_detail = _build_quality(
            recon_error,
            rarity,
            recon_weight=recon_weight,
            rarity_weight=rarity_weight,
            fast_diagnostics=fast_diagnostics,
            err_norm=err_norm,
            rar_norm=rar_norm,
        )

        n = feats.size(0)
        k_eff = min(k_target, n)
        if selector == "ot":
            rank_scores = _ot_rank_scores(
                feats,
                t,
                q,
                k_target,
                sinkhorn_eps=sinkhorn_eps,
                sinkhorn_iters=sinkhorn_iters,
                codebook_accelerate=codebook_accelerate,
            )
        else:
            rank_scores = _dpp_rank_scores(
                feats, q, k_eff, gamma=gamma, epsilon=epsilon, f_norm=f_norm
            )
        prof["rank_scores"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        final_indices, select_diag = _select_to_k(
            feats,
            t,
            q,
            k_target,
            rank_scores=rank_scores,
            per_id_cap=per_id_cap,
            epsilon=epsilon,
            f_norm=f_norm,
            inverse=inverse,
            id_freq=id_freq,
        )
        prof["select_to_k"] = time.perf_counter() - t3

        idx_t = torch.tensor(final_indices, dtype=torch.long, device=t.device)
        pruned = t.index_select(0, idx_t)
        idx_on_feats = idx_t.to(feats.device)

        t4 = time.perf_counter()
        unique_token_count = int(torch.unique(pruned).numel())
        q_kept = q.index_select(0, idx_on_feats)

        if fast_diagnostics:
            mean_pairwise_cosine_kept = -1.0
            kl_all_vs_kept = -1.0
            q_kept_stats = {
                "mean": float(q_kept.mean().item()),
                "std": 0.0,
                "p50": float(q_kept.mean().item()),
            }
        else:
            feats_kept = feats.index_select(0, idx_on_feats)
            mean_pairwise_cosine_kept = _mean_pairwise_cosine(feats_kept)
            kl_all_vs_kept = _kl_all_vs_kept(t, pruned)
            q_kept_stats = summarize_vec(q_kept)
        prof["diagnostics"] = time.perf_counter() - t4

        cap_relaxed = bool(select_diag["cap_relaxed"])
        pair_cos_log = (
            f"{mean_pairwise_cosine_kept:.4g}"
            if mean_pairwise_cosine_kept >= 0
            else "skipped"
        )
        kl_log = (
            f"{kl_all_vs_kept:.4g}" if kl_all_vs_kept >= 0 else "skipped"
        )
        logger.info(
            f"prune_done kr={float(log_kr):.4g} tag={log_tag} selector={selector} "
            f"k={pruned.numel()} k_target={k_target} unique_kept={unique_token_count} "
            f"q_kept_mean={q_kept_stats['mean']:.4g} "
            f"pair_cos_kept={pair_cos_log} kl_all_vs_kept={kl_log} "
            f"cap_relaxed={cap_relaxed} cap_needed={select_diag['cap_needed']} "
            f"cap_eff_final={select_diag['cap_eff_final']} per_id_cap={per_id_cap}"
        )
        if profile_prune:
            prof["total"] = time.perf_counter() - t0
            logger.info(
                "prune_profile kr=%.4g tag=%s "
                "embed_gather=%.3fs recon_rarity=%.3fs rank_scores=%.3fs "
                "select_to_k=%.3fs diagnostics=%.3fs total=%.3fs cache_hit=%s",
                float(log_kr),
                log_tag,
                prof.get("embed_gather", 0.0),
                prof.get("recon_rarity", 0.0),
                prof.get("rank_scores", 0.0),
                prof.get("select_to_k", 0.0),
                prof.get("diagnostics", 0.0),
                prof.get("total", 0.0),
                cached is not None,
            )

        meta: Dict[str, Any] = {
            "method": "reconot",
            "version": "v4_unified_select_to_k_perf4",
            "k": int(pruned.numel()),
            "indices": final_indices,
            "selector": selector,
            "gamma": gamma,
            "merge_mode": "select",
            "diagnostics": {
                "unique_token_count": unique_token_count,
                "n_basis_used": len(basis_idx),
                "recon_method": "basis_fps_global",
                "per_id_cap": per_id_cap,
                "cap_relaxed": cap_relaxed,
                "cap_needed": select_diag["cap_needed"],
                "cap_eff_final": select_diag["cap_eff_final"],
                "cap_jump_count": select_diag.get("cap_jump_count", 0),
                "select_steps": select_diag.get("select_steps", 0),
                "k_target": k_target,
                "k_out": select_diag["k_out"],
                "mean_pairwise_cosine_kept": mean_pairwise_cosine_kept,
                "kl_divergence_all_vs_kept": kl_all_vs_kept,
                "quality_kept_stats": {
                    "mean": float(q_kept_stats["mean"]),
                    "std": float(q_kept_stats.get("std", 0.0)),
                    "median": float(q_kept_stats.get("p50", q_kept_stats["mean"])),
                },
                "quality_detail": quality_detail,
                "selector": selector,
                "recon_weight": recon_weight,
                "rarity_weight": rarity_weight,
                "fast_diagnostics": fast_diagnostics,
                "prune_device": str(prune_device),
            },
        }
        if profile_prune:
            meta["diagnostics"]["profile"] = prof
        return pruned, meta
