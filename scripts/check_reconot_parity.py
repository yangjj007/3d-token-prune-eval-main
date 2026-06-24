#!/usr/bin/env python3
"""Compare reconot v4_perf4 vs v4_perf3 golden (rank_scores + final_indices)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KEEP_RATIOS = (0.75, 0.5, 0.25)


def _make_synthetic_tokens(n_mesh: int, seq_len: int, vocab: int, seed: int) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out: List[torch.Tensor] = []
    for i in range(n_mesh):
        # 偏斜频次：模拟真实 mesh 的 cap_needed 分布
        probs = torch.rand(vocab, generator=g).clamp_min(1e-6)
        probs = probs ** (1.0 + (i % 5) * 0.3)
        probs = probs / probs.sum()
        out.append(torch.multinomial(probs, seq_len, replacement=True, generator=g))
    return out


def _run_parity_mesh(
    token_ids: torch.Tensor,
    vq_emb: torch.nn.Embedding,
    extra: Dict[str, Any],
    *,
    mesh_tag: str,
) -> Tuple[int, List[str]]:
    from eval.baseline._common import MESH_SEQ_LEN, gather_embeddings
    from eval.proposed import reconot as cur
    from eval.proposed.reconot_ref import dpp_rank_scores_v4perf3

    cur.clear_mesh_score_cache()
    mismatches: List[str] = []
    n_fail = 0

    scores = cur._compute_mesh_scores(
        vq_emb,
        token_ids,
        recon_weight=float(extra.get("recon_weight", 0.5)),
        n_basis=int(extra.get("n_basis", 64)),
        ridge_lambda=float(extra.get("ridge_lambda", 1e-3)),
        prune_device="cpu",
    )
    feats = scores["feats"]
    f_norm = scores["f_norm"]
    t = token_ids.detach().long().view(-1)
    assert t.numel() == MESH_SEQ_LEN

    for kr in KEEP_RATIOS:
        gamma = float(extra.get("gamma", 0.01))
        epsilon = float(extra.get("epsilon", 1e-10))
        recon_w = float(extra.get("recon_weight", 0.5))
        rarity_w = float(extra.get("rarity_weight", 0.25))
        recon_w, rarity_w, gamma = cur._kr_adaptive_weights(kr, recon_w, rarity_w, gamma)

        q, _ = cur._build_quality(
            scores["recon_error"],
            scores["rarity"],
            recon_weight=recon_w,
            rarity_weight=rarity_w,
            fast_diagnostics=True,
            err_norm=scores["err_norm"],
            rar_norm=scores["rar_norm"],
        )
        k_target = int(round(kr * MESH_SEQ_LEN))
        k_eff = min(k_target, feats.size(0))

        rs_gold = dpp_rank_scores_v4perf3(feats, q, k_eff, gamma=gamma, epsilon=epsilon)
        rs_new = cur._dpp_rank_scores(
            feats, q, k_eff, gamma=gamma, epsilon=epsilon, f_norm=f_norm
        )
        if not torch.equal(rs_gold, rs_new):
            diff = (rs_gold - rs_new).abs().max().item()
            mismatches.append(f"{mesh_tag} kr={kr}: rank_scores max_diff={diff}")
            n_fail += 1

        per_id_cap = max(1, int(extra.get("per_id_cap", 1)))
        idx_gold, _ = cur._select_to_k(
            feats,
            t,
            q,
            k_target,
            rank_scores=rs_gold,
            per_id_cap=per_id_cap,
            epsilon=epsilon,
            f_norm=f_norm,
            inverse=scores["inverse"],
            id_freq=scores["id_freq"],
        )
        idx_new, _ = cur._select_to_k(
            feats,
            t,
            q,
            k_target,
            rank_scores=rs_new,
            per_id_cap=per_id_cap,
            epsilon=epsilon,
            f_norm=f_norm,
            inverse=scores["inverse"],
            id_freq=scores["id_freq"],
        )
        if idx_gold != idx_new:
            mismatches.append(f"{mesh_tag} kr={kr}: indices differ")
            n_fail += 1

    return n_fail, mismatches


def cmd_parity(args: argparse.Namespace) -> int:
    from eval.config import load_pruner_extra_kwargs

    extra = load_pruner_extra_kwargs(REPO_ROOT / "configs" / "eval", "reconot")
    extra["fast_diagnostics"] = True
    extra["per_id_cap"] = 1
    extra["prune_device"] = "cpu"

    from eval.baseline._common import MESH_SEQ_LEN

    vocab = args.vocab
    emb = torch.nn.Embedding(vocab, args.dim)
    torch.manual_seed(args.seed)
    emb.weight.data.normal_(0.0, 0.1)

    tokens_list = _make_synthetic_tokens(args.num_mesh, MESH_SEQ_LEN, vocab, args.seed)
    total_fail = 0
    all_mm: List[str] = []
    for i, tok in enumerate(tokens_list):
        nf, mm = _run_parity_mesh(tok, emb, extra, mesh_tag=f"synth_{i:04d}")
        total_fail += nf
        all_mm.extend(mm)

    print(f"parity meshes={args.num_mesh} kr_each=3 mismatches={total_fail}")
    for line in all_mm[:20]:
        print(f"  {line}")
    if len(all_mm) > 20:
        print(f"  ... and {len(all_mm) - 20} more")
    return 1 if total_fail else 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from eval.config import load_pruner_extra_kwargs
    from eval.proposed.reconot import ReconOTPruner, clear_mesh_score_cache

    extra = load_pruner_extra_kwargs(REPO_ROOT / "configs" / "eval", "reconot")
    extra["fast_diagnostics"] = True
    extra["profile_prune"] = bool(args.profile)

    from eval.baseline._common import MESH_SEQ_LEN

    vocab = args.vocab
    emb = torch.nn.Embedding(vocab, args.dim)
    torch.manual_seed(args.seed)
    emb.weight.data.normal_(0.0, 0.1)
    tokens_list = _make_synthetic_tokens(args.num_mesh, MESH_SEQ_LEN, vocab, args.seed)

    totals: List[float] = []
    prof_rank: List[float] = []
    prof_select: List[float] = []
    prof_diag: List[float] = []
    for i, tok in enumerate(tokens_list):
        clear_mesh_score_cache()
        t0 = time.perf_counter()
        for kr in KEEP_RATIOS:
            pr = ReconOTPruner(keep_ratio=kr, seed=42, **extra)
            _, meta = pr.prune(
                tok, None, vq_embeddings=emb, _log_tag=f"bench_{i}", _log_keep_ratio=kr
            )
            if args.profile and kr == 0.75:
                p = (meta.get("diagnostics") or {}).get("profile") or {}
                prof_rank.append(float(p.get("rank_scores", 0.0)))
                prof_select.append(float(p.get("select_to_k", 0.0)))
                prof_diag.append(float(p.get("diagnostics", 0.0)))
        totals.append(time.perf_counter() - t0)

    totals.sort()
    mean = sum(totals) / len(totals)
    p50 = totals[len(totals) // 2]
    p95 = totals[int(len(totals) * 0.95)]
    print(
        f"synth_benchmark n={len(totals)} mean={mean:.3f}s p50={p50:.3f}s "
        f"p95={p95:.3f}s max={totals[-1]:.3f}s (3 kr / mesh, cpu)"
    )
    ok = mean <= args.target_mean
    print(f"target mean<={args.target_mean}s: {'PASS' if ok else 'FAIL'}")
    if args.profile and prof_rank:
        nr = sum(prof_rank) / len(prof_rank)
        ns = sum(prof_select) / len(prof_select)
        nd = sum(prof_diag) / len(prof_diag)
        tot = nr + ns + nd
        print(
            f"kr=0.75 phase means: rank_scores={nr:.3f}s select_to_k={ns:.3f}s "
            f"diagnostics={nd:.3f}s  (rank share {100*nr/max(tot,1e-9):.0f}%)"
        )
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="ReconOT parity & synth benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("parity", help="Compare perf4 vs v4perf3 golden")
    pp.add_argument("--num-mesh", type=int, default=100)
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--vocab", type=int, default=512)
    pp.add_argument("--dim", type=int, default=64)

    pb = sub.add_parser("benchmark", help="Synthetic CPU prune timing")
    pb.add_argument("--num-mesh", type=int, default=300)
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--vocab", type=int, default=512)
    pb.add_argument("--dim", type=int, default=64)
    pb.add_argument("--target-mean", type=float, default=5.0)
    pb.add_argument("--profile", action="store_true")

    args = p.parse_args()
    if args.cmd == "parity":
        return cmd_parity(args)
    if args.cmd == "benchmark":
        return cmd_benchmark(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
