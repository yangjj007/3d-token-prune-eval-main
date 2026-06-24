#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断前 N 个 mesh 样本的 ``token_ids`` 是否重复（6.0.3：前 9 个物体统计完全一致可疑）。

用法（在仓库根 ``ShapeLLM-Omni-main`` 下，需可导入 ``trellis`` / VQVAE）::

    python scripts/diagnose_sample_dup.py --csv path/to/meta.csv --glb-dir path/to/glbs --device cuda:0 --limit 30

仅打印哈希与前 16 个 id，不修改任何评测代码。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.data_loader import iter_dataset, mesh_to_tokens  # noqa: E402
from trellis.models.sparse_structure_vqvae import VQVAE3D  # noqa: E402


def _sha1_token_ids(t: torch.Tensor) -> str:
    b = t.detach().cpu().long().numpy().tobytes()
    return hashlib.sha1(b).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="数据集 CSV（含 file_identifier）")
    p.add_argument("--glb-dir", type=str, required=True, help="GLB 根目录")
    p.add_argument("--ckpt", type=str, default="", help="VQVAE 权重路径（若为空则仅检查导入失败）")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args()

    device = torch.device(args.device)
    if not args.ckpt:
        print("WARN: --ckpt 未提供，跳过 mesh_to_tokens（仅验证 iter_dataset 顺序）")
        samples = list(iter_dataset(args.csv, args.glb_dir, num_samples=args.limit))
        for i, s in enumerate(samples):
            print(f"idx={i} tag={s.file_identifier} glb={s.glb_path}")
        return

    vqvae = VQVAE3D()
    ck = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ck, dict) and "state_dict" in ck:
        vqvae.load_state_dict(ck["state_dict"], strict=False)
    else:
        vqvae.load_state_dict(ck, strict=False)
    vqvae.to(device)
    vqvae.eval()

    hashes: list[str] = []
    heads16: list[list[int]] = []
    samples = list(iter_dataset(args.csv, args.glb_dir, num_samples=args.limit))
    for i, s in enumerate(samples):
        tok, vox = mesh_to_tokens(s.glb_path, vqvae, device)
        h = _sha1_token_ids(tok)
        hashes.append(h)
        uq = int(tok.unique().numel())
        vs = int(vox.sum().item())
        head = tok[:16].tolist()
        heads16.append([int(x) for x in head])
        print(
            f"idx={i} tag={s.file_identifier} sha1={h[:12]}… "
            f"unique={uq}/1024 voxel_sum={vs} head16={head}"
        )

    nchk = min(9, len(hashes))
    if nchk >= 2 and len(set(hashes[:nchk])) == 1:
        print("\n[ALERT] 前 9 个样本 token_ids SHA1 完全相同 → 编码/数据管线可疑（6.0.3）")
    else:
        print(f"\n[OK] 前 min(9,n) 个样本哈希种类数 = {len(set(hashes[:nchk]))}")

    if len(heads16) >= 9 and all(h == heads16[0] for h in heads16[:9]):
        print(
            "[ALERT] 前 9 个样本 token_ids[:16] 完全一致 → 与评测日志「不同 tag 统计相同」现象一致，"
            "请结合 SHAPELLM_EVAL_TOKEN_HEAD_DUMP 落盘与 GLB 路径核对（6.0.3 / 6.7.3）。"
        )
    elif len(heads16) >= 2:
        nhead = min(9, len(heads16))
        nuniq_head = len({tuple(h) for h in heads16[:nhead]})
        print(f"[head16] 前 min(9,n) 个样本 head16 种类数 = {nuniq_head}")


if __name__ == "__main__":
    main()
