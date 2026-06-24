# -*- coding: utf-8 -*-
"""
方案六：Hierarchical octree-style VQ token merging（层次八叉树式 Token 合并）

V9：体素真表面、L1-only 块合并、``geom_complex_bonus_scale``、边界 bonus 用量纲 ``median(L1 spread)``；已移除 L2 层。
V7 行为见文档 §6.2 / git 历史（此处不重复贴全文）。
'''V7_BEGIN
(verbatim V7 OctreeMergePruner omitted — restore from git/docs if needed)
V7_END'''
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch

from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings, nearest_codebook_ids, require_vq_embeddings, target_keep_count
from eval.pruners import BasePruner, register_pruner
from eval.proposed._diag import boundary_interior_counts, tensor_score_stats
from eval.proposed._logging import get_pruner_logger, should_deep_dump, summarize_vec, write_deep_dump
from eval.proposed._spatial import _is_grid_boundary, coord_to_flat_index, flat_index_to_coord, latent_surface_mask


def _l1_block_flats(bx: int, by: int, bz: int) -> List[int]:
    out: List[int] = []
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                x, y, z = 2 * bx + dx, 2 * by + dy, 2 * bz + dz
                out.append(coord_to_flat_index(x, y, z))
    return out


def _block_spread(emb8: torch.Tensor, metric: Literal["l2", "cosine"]) -> torch.Tensor:
    m = emb8.mean(dim=0, keepdim=True)
    if metric == "cosine":
        en = emb8 / emb8.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mn = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (1.0 - (en * mn).sum(dim=-1)).mean()
    return ((emb8 - m) ** 2).sum(dim=-1).mean()


def _l1_key_from_flat(fi: int) -> Tuple[int, int, int]:
    x, y, z = flat_index_to_coord(fi)
    return x // 2, y // 2, z // 2


@register_pruner("octree_merge")
class OctreeMergePruner(BasePruner):
    """Octree-style L1 block merge + per-flat scoring + quota-aware selection (V9, no L2)."""

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

        tau_merge = float(self.extra.get("tau_merge", 0.05))
        tau_pct = float(self.extra.get("tau_pct", 1.0))
        adaptive_tau = bool(self.extra.get("adaptive_tau", False))
        similarity_metric = str(self.extra.get("similarity_metric", "l2"))
        if similarity_metric not in ("l2", "cosine"):
            similarity_metric = "l2"
        metric: Literal["l2", "cosine"] = similarity_metric  # type: ignore

        surface_include_grid = bool(self.extra.get("surface_include_grid", True))
        surface_source = str(self.extra.get("surface_source", "voxel"))
        surface_tau_scale = float(self.extra.get("surface_tau_scale", 0.7))
        geometry_complex_min_unique = int(self.extra.get("geometry_complex_min_unique", 4))
        geom_complex_bonus_scale = float(self.extra.get("geom_complex_bonus_scale", 0.5))
        score_alpha = float(self.extra.get("score_alpha", 0.3))
        boundary_bonus_scale = float(self.extra.get("boundary_bonus_scale", 0.0))
        base_reserve_ratio = float(self.extra.get("base_reserve_ratio", 0.2))

        device = embed.weight.device
        t_dev = t.to(device)
        emb = gather_embeddings(embed, t_dev)

        if voxel_grid is not None and surface_source == "voxel":
            is_surface_t, _ie, _if, _occ = latent_surface_mask(voxel_grid.cpu())
            is_surface_t = is_surface_t.to(device=device)
        elif surface_include_grid:
            is_surface_t = torch.tensor(
                [_is_grid_boundary(*flat_index_to_coord(i)) for i in range(MESH_SEQ_LEN)],
                device=device,
                dtype=torch.bool,
            )
        else:
            is_surface_t = torch.zeros(MESH_SEQ_LEN, dtype=torch.bool, device=device)

        spreads_l1: List[float] = []
        block_spreads: Dict[Tuple[int, int, int], float] = {}
        for bx in range(4):
            for by in range(4):
                for bz in range(8):
                    flats = _l1_block_flats(bx, by, bz)
                    e8 = torch.stack([emb[fi] for fi in flats], dim=0)
                    sp = float(_block_spread(e8, metric).item())
                    block_spreads[(bx, by, bz)] = sp
                    spreads_l1.append(sp)

        med_sp_all = float(np.median(spreads_l1)) if spreads_l1 else float(tau_merge)
        tau_eff_L1 = float(tau_pct) * med_sp_all if adaptive_tau else float(tau_merge)

        states: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for bx in range(4):
            for by in range(4):
                for bz in range(8):
                    flats = _l1_block_flats(bx, by, bz)
                    e8 = torch.stack([emb[fi] for fi in flats], dim=0)
                    sp = float(block_spreads[(bx, by, bz)])
                    tids = [int(t_dev[fi].item()) for fi in flats]
                    nuniq = len(set(tids))
                    is_surface_blk = any(bool(is_surface_t[fi].item()) for fi in flats)
                    is_geom_complex = nuniq >= geometry_complex_min_unique
                    strict = bool(is_surface_blk or is_geom_complex)
                    eff_tau_blk = float(tau_eff_L1) * float(surface_tau_scale) if strict else float(tau_eff_L1)

                    if sp < eff_tau_blk:
                        mean_e = e8.mean(dim=0)
                        tid = int(nearest_codebook_ids(embed, mean_e.unsqueeze(0))[0].item())
                        states[(bx, by, bz)] = {
                            "merged": True,
                            "rep_flat": min(flats),
                            "tid": tid,
                            "spread": sp,
                            "flats": flats,
                            "is_surface": is_surface_blk,
                            "is_geom_complex": is_geom_complex,
                            "eff_tau_blk": float(eff_tau_blk),
                            "nuniq": nuniq,
                        }
                    else:
                        states[(bx, by, bz)] = {
                            "merged": False,
                            "flats": flats,
                            "tids": tids,
                            "spread": sp,
                            "is_surface": is_surface_blk,
                            "is_geom_complex": is_geom_complex,
                            "eff_tau_blk": float(eff_tau_blk),
                            "nuniq": nuniq,
                        }

        tau_bnd_scale = float(max(med_sp_all, 1e-8))

        gmean = emb.mean(dim=0)
        dev_block = torch.zeros(MESH_SEQ_LEN, device=device, dtype=torch.float32)
        global_dev = torch.zeros(MESH_SEQ_LEN, device=device, dtype=torch.float32)
        block_weight_t = torch.zeros(MESH_SEQ_LEN, device=device, dtype=torch.float32)
        boundary_bonus_t = torch.zeros(MESH_SEQ_LEN, device=device, dtype=torch.float32)
        flat_to_eff_tau: Dict[int, float] = {}
        flat_to_spread: Dict[int, float] = {}
        flat_to_mvec: Dict[int, torch.Tensor] = {}

        for (_bx, _by, _bz), st in states.items():
            flats = st["flats"]
            e8 = torch.stack([emb[fi] for fi in flats], dim=0)
            mvec = e8.mean(dim=0)
            sp = float(st["spread"])
            eff_tau_blk = float(st.get("eff_tau_blk", tau_eff_L1))
            for fi in flats:
                flat_to_spread[fi] = sp
                flat_to_mvec[fi] = mvec
                flat_to_eff_tau[fi] = eff_tau_blk

        for fi in range(MESH_SEQ_LEN):
            sp = flat_to_spread.get(fi, tau_eff_L1)
            mvec = flat_to_mvec.get(fi, emb[fi])
            eff_tau = flat_to_eff_tau.get(fi, tau_eff_L1)
            dev_block[fi] = torch.norm(emb[fi] - mvec, p=2)
            global_dev[fi] = torch.norm(emb[fi] - gmean, p=2)
            block_weight_t[fi] = float(sp) / (float(sp) + float(eff_tau) + 1e-8)
            if bool(is_surface_t[fi].item()):
                boundary_bonus_t[fi] = float(tau_bnd_scale) * float(boundary_bonus_scale)

        term_dev_block = dev_block * block_weight_t
        term_global = score_alpha * global_dev
        term_boundary = boundary_bonus_t

        geom_bonus = torch.zeros(MESH_SEQ_LEN, device=device, dtype=torch.float32)
        for _key, st in states.items():
            if (not st["merged"]) and bool(st.get("is_geom_complex")):
                for fi in st["flats"]:
                    geom_bonus[fi] = float(geom_complex_bonus_scale) * float(tau_eff_L1)

        scores = term_dev_block + term_global + term_boundary + geom_bonus

        eff_tau_vals = torch.tensor([flat_to_eff_tau.get(i, tau_eff_L1) for i in range(MESH_SEQ_LEN)], dtype=torch.float32)
        spread_vals = torch.tensor([flat_to_spread.get(i, 0.0) for i in range(MESH_SEQ_LEN)], dtype=torch.float32)

        base_candidates: List[int] = []
        for _key, st in states.items():
            if st["merged"] and bool(st.get("is_surface")):
                base_candidates.append(int(st["rep_flat"]))
            if (not st["merged"]) and bool(st.get("is_geom_complex")):
                base_candidates.extend(int(x) for x in st["flats"])

        max_base = min(k_target, max(0, int(round(k_target * base_reserve_ratio))))
        base_candidates = list(dict.fromkeys(base_candidates))
        if max_base > 0 and base_candidates:
            base_candidates.sort(key=lambda fi: float(scores[fi].item()), reverse=True)
            base_set = set(base_candidates[:max_base])
        else:
            base_set = set()

        if len(base_set) > k_target:
            base_list = sorted(base_set, key=lambda fi: float(scores[fi].item()), reverse=True)[:k_target]
            base_set = set(base_list)

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
            final_flats = sorted(set(base_set) | set(picked))
        else:
            final_flats = sorted(base_set)

        if len(final_flats) > k_target:
            final_flats = sorted(final_flats, key=lambda fi: (-float(scores[fi].item()), fi))[:k_target]
            final_flats.sort()
        elif len(final_flats) < k_target:
            rest = [i for i in range(MESH_SEQ_LEN) if i not in final_flats]
            rest.sort(key=lambda fi: (-float(scores[fi].item()), fi))
            for fi in rest:
                if len(final_flats) >= k_target:
                    break
                final_flats.append(fi)
            final_flats = sorted(final_flats)

        pruned_list = [int(t_dev[i].item()) for i in final_flats]
        pruned = torch.tensor(pruned_list, dtype=torch.long, device=t.device)

        n_merged_blocks = sum(1 for _k, st in states.items() if st["merged"])
        n_unmerged = sum(1 for _k, st in states.items() if not st["merged"])
        n_surface = sum(1 for _k, st in states.items() if st.get("is_surface"))
        n_geom = sum(1 for _k, st in states.items() if st.get("is_geom_complex"))

        b_kept, i_kept = boundary_interior_counts(final_flats, surface_mask=is_surface_t.detach().cpu())
        b_shell, i_shell = boundary_interior_counts(final_flats)

        b_sc = [float(scores[i].item()) for i in final_flats if bool(is_surface_t[i].item())]
        i_sc = [float(scores[i].item()) for i in final_flats if not bool(is_surface_t[i].item())]

        term_dev_stats = summarize_vec(term_dev_block)
        term_global_stats = summarize_vec(term_global)
        term_boundary_stats = summarize_vec(term_boundary)
        term_geom_stats = summarize_vec(geom_bonus)
        spread_s = summarize_vec(spread_vals)

        group_scores: Dict[str, List[float]] = {
            "surface": [],
            "geom": [],
            "merged_normal": [],
            "unmerged": [],
        }
        group_spread: Dict[str, List[float]] = {k: [] for k in group_scores}
        for fi in range(MESH_SEQ_LEN):
            bx, by, bz = _l1_key_from_flat(fi)
            key = (bx, by, bz)
            st = states[key]
            if not st["merged"]:
                lab = "unmerged"
            elif st.get("is_surface"):
                lab = "surface"
            elif st.get("is_geom_complex"):
                lab = "geom"
            else:
                lab = "merged_normal"
            group_scores[lab].append(float(scores[fi].item()))
            group_spread[lab].append(float(spread_vals[fi].item()))

        group_stats_surface = {"scores": summarize_vec(group_scores["surface"]), "spread": summarize_vec(group_spread["surface"])}
        group_stats_geom = {"scores": summarize_vec(group_scores["geom"]), "spread": summarize_vec(group_spread["geom"])}
        group_stats_merged_normal = {
            "scores": summarize_vec(group_scores["merged_normal"]),
            "spread": summarize_vec(group_spread["merged_normal"]),
        }
        group_stats_unmerged = {
            "scores": summarize_vec(group_scores["unmerged"]),
            "spread": summarize_vec(group_spread["unmerged"]),
        }

        base_count_from_surface = 0
        base_count_from_geom = 0
        for fi in base_set_snapshot:
            for _key, st in states.items():
                if st.get("merged") and int(st["rep_flat"]) == fi and st.get("is_surface"):
                    base_count_from_surface += 1
                    break
                if (not st.get("merged")) and fi in st["flats"] and st.get("is_geom_complex"):
                    base_count_from_geom += 1
                    break

        bbf, bbi = boundary_interior_counts(list(base_set_snapshot), surface_mask=is_surface_t.detach().cpu())
        non_base_kept = [i for i in final_flats if i not in base_set_snapshot]
        topk_fill_surface_kept = sum(1 for i in non_base_kept if bool(is_surface_t[i].item()))
        topk_fill_interior_kept = len(non_base_kept) - topk_fill_surface_kept

        grid_shell = torch.tensor(
            [_is_grid_boundary(*flat_index_to_coord(i)) for i in range(MESH_SEQ_LEN)],
            dtype=torch.bool,
        )
        true_cnt = int(is_surface_t.detach().cpu().sum().item())
        shell_cnt = int(grid_shell.sum().item())
        overlap = int((is_surface_t.detach().cpu() & grid_shell).sum().item())

        l1_decisions: List[Dict[str, Any]] = []
        for key in sorted(states.keys(), key=lambda k: k):
            st = states[key]
            bx, by, bz = key
            row: Dict[str, Any] = {
                "bx": bx,
                "by": by,
                "bz": bz,
                "merged": bool(st["merged"]),
                "spread": float(st["spread"]),
                "eff_tau_blk": float(st.get("eff_tau_blk", tau_eff_L1)),
                "nuniq": int(st.get("nuniq", 0)),
                "is_surface": bool(st.get("is_surface")),
                "is_geom_complex": bool(st.get("is_geom_complex")),
            }
            if st["merged"]:
                row["rep_flat"] = int(st["rep_flat"])
                row["tid"] = int(st["tid"])
            l1_decisions.append(row)

        log_tag = str(kwargs.get("_log_tag", ""))
        log_si = kwargs.get("_log_sample_idx")
        log_kr = kwargs.get("_log_keep_ratio", self.keep_ratio)
        logger = get_pruner_logger("octree_merge")
        logger.info(
            f"kr={float(log_kr):.4g} tag={log_tag} "
            f"L1(merged/unmerged)={n_merged_blocks}/{n_unmerged} surface={n_surface} geom={n_geom} "
            f"tau_eff_L1={tau_eff_L1:.4g} tau_bnd_scale={tau_bnd_scale:.4g} "
            f"surf_share(true/grid/overlap)={true_cnt}/{shell_cnt}/{overlap} "
            f"spread(mean/p50/p90)={spread_s['mean']:.4g}/{spread_s['p50']:.4g}/{spread_s['p90']:.4g} "
            f"terms(dev/global/bb/geom)={term_dev_stats['mean']:.4g}/{term_global_stats['mean']:.4g}/"
            f"{term_boundary_stats['mean']:.4g}/{term_geom_stats['mean']:.4g} "
            f"base={len(base_set_snapshot)}(S={base_count_from_surface}/G={base_count_from_geom} "
            f"B={bbf}/I={bbi}) topk={len(picked)} kept(surf/other)={b_kept}/{i_kept} shell_kept(B/I)={b_shell}/{i_shell}"
        )

        try:
            log_si_int = int(log_si) if log_si is not None else None
        except (TypeError, ValueError):
            log_si_int = None
        if should_deep_dump(log_si_int):
            write_deep_dump(
                "octree_merge",
                {
                    "tag": log_tag,
                    "kr": float(log_kr),
                    "sample_idx": log_si_int,
                    "tau_eff_L1": tau_eff_L1,
                    "tau_bnd_scale": tau_bnd_scale,
                    "l1_decisions": l1_decisions,
                    "dev_block": dev_block,
                    "global_dev": global_dev,
                    "boundary_bonus": boundary_bonus_t,
                    "geom_bonus": geom_bonus,
                    "scores": scores,
                    "is_surface_token": is_surface_t,
                    "base_set": sorted(base_set_snapshot),
                    "topk_picked": sorted(set(picked)),
                    "final_flats": list(final_flats),
                },
            )

        meta: Dict[str, Any] = {
            "method": "octree_merge",
            "version": "v9_l1_only_geom_bonus",
            "tau_merge": tau_merge,
            "tau_eff_L1": float(tau_eff_L1),
            "tau_bnd_scale": float(tau_bnd_scale),
            "tau_pct": tau_pct,
            "adaptive_tau": adaptive_tau,
            "geom_complex_bonus_scale": geom_complex_bonus_scale,
            "similarity_metric": similarity_metric,
            "surface_source": surface_source,
            "k": int(pruned.numel()),
            "indices": list(final_flats),
            "diagnostics": {
                "version": "v9_l1_only_geom_bonus",
                "true_surface_token_count": true_cnt,
                "grid_shell_token_count": shell_cnt,
                "true_vs_shell_overlap_count": overlap,
                "base_count": len(base_set),
                "base_ratio_of_k": float(len(base_set) / max(k_target, 1)),
                "base_surface_count": boundary_interior_counts(list(base_set), surface_mask=is_surface_t.detach().cpu())[0],
                "base_nonsurface_count": boundary_interior_counts(list(base_set), surface_mask=is_surface_t.detach().cpu())[1],
                "topk_fill_count": int(max(0, remaining)),
                "score_stats": tensor_score_stats(scores),
                "surface_token_kept": b_kept,
                "nonsurface_token_kept": i_kept,
                "grid_shell_kept_boundary": int(b_shell),
                "grid_shell_kept_interior": int(i_shell),
                "surface_score_mean": float(sum(b_sc) / len(b_sc)) if b_sc else 0.0,
                "nonsurface_score_mean": float(sum(i_sc) / len(i_sc)) if i_sc else 0.0,
                "num_merged_blocks": int(n_merged_blocks),
                "num_unmerged_blocks": int(n_unmerged),
                "num_surface_blocks": int(n_surface),
                "num_geom_complex_blocks": int(n_geom),
                "spread_stats": tensor_score_stats(spread_vals),
                "eff_tau_stats": tensor_score_stats(eff_tau_vals),
                "term_dev_block_stats": term_dev_stats,
                "term_global_stats": term_global_stats,
                "term_boundary_stats": term_boundary_stats,
                "term_geom_stats": term_geom_stats,
                "group_stats_surface": group_stats_surface,
                "group_stats_geom": group_stats_geom,
                "group_stats_merged_normal": group_stats_merged_normal,
                "group_stats_unmerged": group_stats_unmerged,
                "base_count_from_surface": int(base_count_from_surface),
                "base_count_from_geom": int(base_count_from_geom),
                "topk_fill_surface_kept": int(topk_fill_surface_kept),
                "topk_fill_interior_kept": int(topk_fill_interior_kept),
            },
        }
        return pruned.cpu(), meta
