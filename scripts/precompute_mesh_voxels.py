#!/usr/bin/env python3
"""Precompute Open3D mesh voxel coords for eval (Med-style offline cache)."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.data_loader import iter_dataset  # noqa: E402
from eval.mesh_voxelize import load_vertices, vertices_to_coords  # noqa: E402
from eval.voxel_cache import (  # noqa: E402
    cache_path,
    load_cached_coords,
    save_cached_coords,
)


def _process_one(glb_path: str, file_identifier: str, mesh_cache_dir: str) -> dict:
    cpath = cache_path(mesh_cache_dir, file_identifier)
    if load_cached_coords(cpath) is not None:
        return {"file_identifier": file_identifier, "status": "skip", "path": str(cpath)}
    position_recon = load_vertices(glb_path)
    coords = vertices_to_coords(position_recon)
    save_cached_coords(
        cpath, coords, glb_path=glb_path, position_recon=position_recon
    )
    return {
        "file_identifier": file_identifier,
        "status": "ok",
        "path": str(cpath),
        "num_voxels": int(coords.shape[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Precompute mesh voxel coords (.npz cache)")
    p.add_argument("--data-csv", type=str, default="../data/metadata.csv")
    p.add_argument("--glb-dir", type=str, default="../data")
    p.add_argument("--mesh-cache-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=-1)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    csv_path = Path(args.data_csv)
    if not csv_path.is_absolute():
        csv_path = (REPO_ROOT / csv_path).resolve()
    glb_dir = Path(args.glb_dir)
    if not glb_dir.is_absolute():
        glb_dir = (REPO_ROOT / glb_dir).resolve()
    cache_dir = Path(args.mesh_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (REPO_ROOT / cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples = list(
        iter_dataset(
            str(csv_path),
            str(glb_dir),
            num_samples=args.num_samples,
            skip_missing_glb=True,
        )
    )
    print(f"Precomputing voxels for {len(samples)} meshes -> {cache_dir}")

    ok = skip = fail = 0
    if args.num_workers <= 1:
        for s in samples:
            try:
                r = _process_one(s.glb_path, s.file_identifier, str(cache_dir))
                if r["status"] == "skip":
                    skip += 1
                else:
                    ok += 1
                    print(f"  ok {s.file_identifier} n={r.get('num_voxels', '?')}")
            except Exception as exc:
                fail += 1
                print(f"  fail {s.file_identifier}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futs = {
                pool.submit(
                    _process_one, s.glb_path, s.file_identifier, str(cache_dir)
                ): s
                for s in samples
            }
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    r = fut.result()
                    if r["status"] == "skip":
                        skip += 1
                    else:
                        ok += 1
                except Exception as exc:
                    fail += 1
                    print(f"  fail {s.file_identifier}: {exc}")

    print(f"Done: ok={ok} skip={skip} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
