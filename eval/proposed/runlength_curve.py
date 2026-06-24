# -*- coding: utf-8 -*-
"""
方案九：Spatial curve run-length mesh token compression（空间填充曲线游程压缩）

V9：体素真表面、自适应 ``epsilon``、``d_mix`` 的 rank 打分 + 表面标量 bonus；base 按 run 分数取锚点（不再仅表面锚点）。
'''V7_BEGIN
(verbatim V7 RunLengthCurvePruner omitted — restore from git/docs if needed)
V7_END'''
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings, require_vq_embeddings, target_keep_count
from eval.pruners import BasePruner, register_pruner
from eval.proposed._diag import boundary_interior_counts, tensor_score_stats
from eval.proposed._logging import get_pruner_logger, should_deep_dump, summarize_vec, write_deep_dump
from eval.proposed._spatial import (
    GRID_X,
    GRID_Y,
    GRID_Z,
    _is_grid_boundary,
    curve_sort_order,
    flat_index_to_coord,
    latent_surface_mask,
    per_token_intra_l1_mean_edge_norm,
)


def _build_runs_curve_order(
    curve_flat_indices: torch.Tensor,
    token_ids_1d: torch.Tensor,
    emb_curve: torch.Tensor,
    epsilon: float,
    max_run_len: int,
) -> List[Tuple[int, int, int]]:
    """Returns list of (anchor_flat_idx, run_start_pos_along_curve, run_length)."""
    n = curve_flat_indices.numel()
    if n == 0:
        return []

    runs: List[Tuple[int, int, int]] = []
    eps = float(epsilon)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            if max_run_len > 0 and (j - i) >= max_run_len:
                break
            if eps <= 0.0:
                same = int(token_ids_1d[curve_flat_indices[j]].item()) == int(
                    token_ids_1d[curve_flat_indices[j - 1]].item()
                )
            else:
                d = torch.norm(emb_curve[j] - emb_curve[j - 1], p=2).item()
                same = d < eps
            if not same:
                break
            j += 1
        anchor_flat = int(curve_flat_indices[i].item())
        runs.append((anchor_flat, i, j - i))
        i = j

    return runs


def _run_coverage_cdf(run_lens: List[int]) -> List[float]:
    if not run_lens:
        return [0.0, 0.0, 0.0]
    s = sorted(run_lens, reverse=True)
    cum = np.cumsum(np.array(s, dtype=np.float64)) / 1024.0

    def at_pct(p: float) -> float:
        if cum.size == 0:
            return 0.0
        idx = int(np.ceil(p / 100.0 * float(cum.size))) - 1
        idx = max(0, min(idx, cum.size - 1))
        return float(cum[idx])

    return [at_pct(10.0), at_pct(50.0), at_pct(90.0)]


@register_pruner("runlength_curve")
class RunLengthCurvePruner(BasePruner):
    """Run-length on space-filling curve + V9 rank scores and run-ordered base anchors."""

    def prune(
        self,
        token_ids: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embed = require_vq_embeddings(kwargs.get("vq_embeddings"))
        t = token_ids.detach().long().view(-1)
        assert t.numel() == MESH_SEQ_LEN
        k_target = target_keep_count(self.keep_ratio)

        curve_type = str(self.extra.get("curve_type", "z_order"))
        if curve_type not in ("z_order", "hilbert"):
            curve_type = "z_order"
        epsilon = float(self.extra.get("epsilon", 0.0))
        epsilon_pct = float(self.extra.get("epsilon_pct", 1.0))
        adaptive_epsilon = bool(self.extra.get("adaptive_epsilon", False))
        target_run_ratio = float(self.extra.get("target_run_ratio", 1.0))
        max_run_len = int(self.extra.get("max_run_len", 32))
        d_block_weight = float(self.extra.get("d_block_weight", 0.5))
        run_coverage_alpha = float(self.extra.get("run_coverage_alpha", 0.5))
        boundary_bonus_scale = float(self.extra.get("boundary_bonus_scale", 0.0))
        base_reserve_ratio = float(self.extra.get("base_reserve_ratio", 0.2))
        surface_source = str(self.extra.get("surface_source", "voxel"))

        device = embed.weight.device
        curve_flat = curve_sort_order(curve_type, device=device).long()

        t_dev = t.to(device)
        emb_all = gather_embeddings(embed, t_dev)
        emb_curve = emb_all[curve_flat]

        n = MESH_SEQ_LEN
        d_tmp = torch.norm(emb_curve[1:] - emb_curve[:-1], p=2, dim=-1).clamp_min(0.0) if n > 1 else torch.zeros(0, device=device)
        if adaptive_epsilon and d_tmp.numel() > 0:
            p30 = float(torch.quantile(d_tmp.float(), 0.30).item())
            eps_eff = float(epsilon_pct) * max(p30, 1e-8)
        else:
            eps_eff = float(epsilon)

        if adaptive_epsilon and d_tmp.numel() > 0 and 0.0 < target_run_ratio < 1.0:
            lo = float(d_tmp.min().item())
            hi = float(d_tmp.max().item()) + 1e-6
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                tmp_runs = _build_runs_curve_order(curve_flat, t_dev, emb_curve, mid, max_run_len)
                ratio = len(tmp_runs) / 1024.0
                if ratio > target_run_ratio:
                    lo = mid
                else:
                    hi = mid
            eps_eff = max(eps_eff, hi)

        # 将 eps_eff 夹紧到曲线邻域距离 p50 的带状区间，抑制跨样本 10×～20× 方差（V7 日志复盘 §6.7.3）。
        eps_clip_lo_mult = float(self.extra.get("eps_clip_lo_mult", 0.3))
        eps_clip_hi_mult = float(self.extra.get("eps_clip_hi_mult", 1.2))
        if adaptive_epsilon and d_tmp.numel() > 0:
            p50_d = float(torch.quantile(d_tmp.float(), 0.50).item())
            band_lo = eps_clip_lo_mult * max(p50_d, 1e-8)
            band_hi = eps_clip_hi_mult * max(p50_d, 1e-8)
            eps_eff = float(max(band_lo, min(eps_eff, band_hi)))

        runs = _build_runs_curve_order(curve_flat, t_dev, emb_curve, eps_eff, max_run_len)
        run_lens = [r[2] for r in runs]

        emb_grid = emb_all.view(GRID_X, GRID_Y, GRID_Z, -1)
        d_block_flat = per_token_intra_l1_mean_edge_norm(emb_grid)

        d_curve = torch.zeros(n, device=device, dtype=torch.float32)
        if n > 1:
            d_curve[1:] = torch.norm(emb_curve[1:] - emb_curve[:-1], p=2, dim=-1).clamp_min(0.0)
            d_curve[0] = d_curve[1]
            d_curve[-1] = d_curve[-2]

        flat_at_pos = curve_flat.long()
        d_block_at_curve = d_block_flat[flat_at_pos]

        w = float(d_block_weight)
        d_mix = (1.0 - w) * d_curve + w * d_block_at_curve
        if n > 1:
            d_mix[0] = d_mix[1]
            d_mix[-1] = d_mix[-2]

        if voxel_grid is not None and surface_source == "voxel":
            is_surface, _ie, _if, _occ = latent_surface_mask(voxel_grid.cpu())
            is_surface = is_surface.to(device=device)
        else:
            is_surface = torch.tensor(
                [_is_grid_boundary(*flat_index_to_coord(i)) for i in range(MESH_SEQ_LEN)],
                device=device,
                dtype=torch.bool,
            )

        d_mix_flat = torch.zeros(MESH_SEQ_LEN, device=device, dtype=torch.float32)
        d_mix_flat[flat_at_pos] = d_mix
        denom_flat = max(MESH_SEQ_LEN - 1, 1)
        d_rank = torch.argsort(torch.argsort(d_mix_flat)).float() / float(denom_flat)
        p50_d = float(torch.quantile(d_mix.float(), 0.5).item()) if n > 1 else 0.0
        bb_scalar = p50_d * float(boundary_bonus_scale)
        scores = d_rank + is_surface.float() * float(bb_scalar)

        avg_run = float(sum(run_lens) / max(len(run_lens), 1))
        n_runs_target = max(1, min(len(runs), int(round(float(k_target) / max(avg_run, 1.0)))))

        run_scores: List[Tuple[float, int]] = []
        for ri, (a, s, ln) in enumerate(runs):
            epos = s + ln - 1
            if ln <= 0:
                continue
            d0 = float(d_mix[int(s)].item())
            d1 = float(d_mix[int(epos)].item()) if n > 1 else d0
            rs = max(d0, d1) * (float(ln) ** float(run_coverage_alpha))
            run_scores.append((rs, ri))
        run_scores.sort(key=lambda x: (-x[0], x[1]))

        max_base = min(k_target, max(0, int(round(k_target * base_reserve_ratio))))
        base_set: set[int] = set()
        if max_base > 0 and runs:
            take_runs = min(len(run_scores), max(n_runs_target, 1))
            for j in range(take_runs):
                if len(base_set) >= max_base:
                    break
                ri = run_scores[j][1]
                a, s, ln = runs[ri]
                base_set.add(int(a))
                step = max(int(max_run_len), 1)
                pos = int(s) + step
                while pos < int(s) + int(ln) and len(base_set) < max_base:
                    base_set.add(int(curve_flat[pos].item()))
                    pos += step

        if len(base_set) > k_target:
            base_set = set(sorted(base_set, key=lambda fi: (-float(scores[fi].item()), fi))[:k_target])

        base_set_snapshot = set(base_set)
        picked: List[int] = []
        remaining = k_target - len(base_set)
        if remaining > 0:
            mask = torch.ones(MESH_SEQ_LEN, dtype=torch.bool, device=device)
            if base_set:
                mask[torch.tensor(list(base_set), device=device, dtype=torch.long)] = False
            sub_scores = scores[mask]
            sub_idx = torch.arange(MESH_SEQ_LEN, device=device, dtype=torch.long)[mask]
            k_take = min(remaining, int(sub_scores.numel()))
            _, top_loc = torch.topk(sub_scores, k_take, largest=True)
            picked = sub_idx[top_loc].cpu().tolist()
            final_indices = sorted(set(base_set) | set(picked))
        else:
            final_indices = sorted(base_set)

        if len(final_indices) > k_target:
            final_indices = sorted(final_indices, key=lambda fi: (-float(scores[fi].item()), fi))[:k_target]
            final_indices.sort()
        elif len(final_indices) < k_target:
            rest = [i for i in range(MESH_SEQ_LEN) if i not in final_indices]
            rest.sort(key=lambda fi: (-float(scores[fi].item()), fi))
            for fi in rest:
                if len(final_indices) >= k_target:
                    break
                final_indices.append(fi)
            final_indices = sorted(final_indices)

        idx_t = torch.tensor(final_indices, dtype=torch.long)
        pruned = t.index_select(0, idx_t)

        run_len_hist = dict(Counter(run_lens))
        max_run = max(run_lens) if run_lens else 0

        b_kept, i_kept = boundary_interior_counts(final_indices, surface_mask=is_surface.detach().cpu())
        b_shell, i_shell = boundary_interior_counts(final_indices)
        b_scores = [float(scores[i].item()) for i in final_indices if bool(is_surface[i].item())]
        i_scores = [float(scores[i].item()) for i in final_indices if not bool(is_surface[i].item())]
        b_mean = sum(b_scores) / len(b_scores) if b_scores else 0.0
        i_mean = sum(i_scores) / len(i_scores) if i_scores else 0.0

        d_pair_stats = summarize_vec(d_mix[1:]) if n > 1 else summarize_vec(d_mix[:0])
        run_len_p50 = float(np.percentile(run_lens, 50)) if run_lens else 0.0
        run_len_p90 = float(np.percentile(run_lens, 90)) if run_lens else 0.0
        run_len_p99 = float(np.percentile(run_lens, 99)) if run_lens else 0.0
        base_bb, base_bi = boundary_interior_counts(list(base_set_snapshot), surface_mask=is_surface.detach().cpu())
        non_base_kept = [i for i in final_indices if i not in base_set_snapshot]
        topk_fill_surface_kept = sum(1 for i in non_base_kept if bool(is_surface[i].item()))
        topk_fill_interior_kept = len(non_base_kept) - topk_fill_surface_kept
        kept_b_idx = [i for i in final_indices if bool(is_surface[i].item())]
        kept_i_idx = [i for i in final_indices if not bool(is_surface[i].item())]
        kept_score_stats_surface = (
            summarize_vec(scores[torch.tensor(kept_b_idx, device=device, dtype=torch.long)])
            if kept_b_idx
            else summarize_vec([])
        )
        kept_score_stats_interior = (
            summarize_vec(scores[torch.tensor(kept_i_idx, device=device, dtype=torch.long)])
            if kept_i_idx
            else summarize_vec([])
        )

        voxel_surface_share = float(is_surface.float().mean().item())
        cdf = _run_coverage_cdf(run_lens)

        log_tag = kwargs.get("_log_tag", "")
        log_si = kwargs.get("_log_sample_idx")
        log_kr = kwargs.get("_log_keep_ratio", self.keep_ratio)
        logger = get_pruner_logger("runlength_curve")
        msg = (
            f"kr={float(log_kr):.4g} tag={log_tag} n_runs={len(runs)} n_runs_tgt={n_runs_target} "
            f"eps_eff={eps_eff:.4g} max_run_len={max_run_len} dblk_w={d_block_weight:.3g} "
            f"run_len(avg/p50/p90/max)={avg_run:.3g}/{run_len_p50:.3g}/{run_len_p90:.3g}/{max_run} "
            f"d_mix(mean/p50/p90)={d_pair_stats['mean']:.4g}/{d_pair_stats['p50']:.4g}/{d_pair_stats['p90']:.4g} "
            f"bb_scalar={bb_scalar:.4g} score(mean/std)={float(scores.mean().item()):.4g}/{float(scores.std(unbiased=False).item()):.4g} "
            f"voxel_surf_share={voxel_surface_share:.4g} run_cdf(p10/p50/p90)={cdf[0]:.3g}/{cdf[1]:.3g}/{cdf[2]:.3g} "
            f"base={len(base_set_snapshot)}(B={base_bb}/I={base_bi}) topk={len(picked)} kept(surf/other)={b_kept}/{i_kept} shell_kept(B/I)={b_shell}/{i_shell}"
        )
        logger.info(msg)

        try:
            log_si_int = int(log_si) if log_si is not None else None
        except (TypeError, ValueError):
            log_si_int = None
        if should_deep_dump(log_si_int):
            runs_dump = [[int(a), int(s), int(l)] for (a, s, l) in runs]
            picked_set = set(picked)
            write_deep_dump(
                "runlength_curve",
                {
                    "tag": str(log_tag),
                    "kr": float(log_kr),
                    "sample_idx": log_si_int,
                    "curve_type": curve_type,
                    "eps_eff": float(eps_eff),
                    "runs": runs_dump,
                    "d_mix": d_mix,
                    "bb_scalar": float(bb_scalar),
                    "scores": scores,
                    "is_surface": is_surface,
                    "base_set": sorted(base_set_snapshot),
                    "topk_picked": sorted(picked_set),
                    "final_indices": final_indices,
                },
            )

        meta: Dict[str, Any] = {
            "method": "runlength_curve",
            "version": "v9_rank_surface_scalar",
            "curve_type": curve_type,
            "epsilon": epsilon,
            "eps_eff": float(eps_eff),
            "epsilon_pct": epsilon_pct,
            "adaptive_epsilon": adaptive_epsilon,
            "target_run_ratio": target_run_ratio,
            "max_run_len": max_run_len,
            "d_block_weight": d_block_weight,
            "run_coverage_alpha": run_coverage_alpha,
            "boundary_bonus_scale": boundary_bonus_scale,
            "base_reserve_ratio": base_reserve_ratio,
            "surface_source": surface_source,
            "num_runs_initial": len(runs),
            "n_runs_target": int(n_runs_target),
            "k": int(pruned.numel()),
            "indices": final_indices,
            "diagnostics": {
                "version": "v9_rank_surface_scalar",
                "voxel_surface_share": voxel_surface_share,
                "run_coverage_cdf": cdf,
                "bb_scalar": float(bb_scalar),
                "base_count": len(base_set),
                "base_ratio_of_k": float(len(base_set) / max(k_target, 1)),
                "base_surface_count": int(base_bb),
                "base_nonsurface_count": int(base_bi),
                "topk_fill_count": int(remaining if remaining > 0 else 0),
                "score_stats": tensor_score_stats(scores),
                "surface_token_kept": b_kept,
                "nonsurface_token_kept": i_kept,
                "grid_shell_kept_boundary": int(b_shell),
                "grid_shell_kept_interior": int(i_shell),
                "surface_score_mean": b_mean,
                "nonsurface_score_mean": i_mean,
                "num_runs_initial": len(runs),
                "avg_run_length": avg_run,
                "max_run_length": int(max_run),
                "run_length_distribution": {str(k): int(v) for k, v in sorted(run_len_hist.items())},
                "surface_anchors_in_base": sum(1 for a in base_set if bool(is_surface[int(a)].item())),
                "curve_type": curve_type,
                "d_mix_stats": d_pair_stats,
                "run_len_p50": run_len_p50,
                "run_len_p90": run_len_p90,
                "run_len_p99": run_len_p99,
                "base_count_surface": int(base_bb),
                "base_count_nonsurface": int(base_bi),
                "topk_fill_surface_kept": int(topk_fill_surface_kept),
                "topk_fill_interior_kept": int(topk_fill_interior_kept),
                "kept_score_stats_surface": kept_score_stats_surface,
                "kept_score_stats_interior": kept_score_stats_interior,
            },
        }
        return pruned, meta
