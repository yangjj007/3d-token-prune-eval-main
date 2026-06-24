"""Disk cache for mesh voxel coords (skip repeated Open3D voxelization)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from eval.mesh_voxelize import load_vertices, vertices_to_coords


def cache_path(mesh_cache_dir: str, file_identifier: str) -> Path:
    root = Path(mesh_cache_dir)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in file_identifier)[:120]
    return root / f"{safe}.npz"


def load_cached_coords(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    data = np.load(path)
    if "coords" not in data:
        return None
    coords = np.asarray(data["coords"], dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    return coords


def save_cached_coords(
    path: Path,
    coords: np.ndarray,
    *,
    glb_path: str = "",
    position_recon: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"coords": np.ascontiguousarray(coords.astype(np.int32))}
    if glb_path:
        payload["glb_path"] = np.array(glb_path)
    if position_recon is not None:
        payload["voxel_centers"] = np.ascontiguousarray(
            position_recon.astype(np.float32)
        )
    np.savez_compressed(path, **payload)


def resolve_coords(
    glb_path: str,
    file_identifier: str,
    mesh_cache_dir: str,
    *,
    cache_readonly: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Return ``(coords int32 [N,3], from_cache)``.

    On cache miss runs ``load_vertices`` and optionally writes cache.
    """
    if not mesh_cache_dir:
        position_recon = load_vertices(glb_path)
        return vertices_to_coords(position_recon), False

    cpath = cache_path(mesh_cache_dir, file_identifier)
    cached = load_cached_coords(cpath)
    if cached is not None:
        return cached, True

    if cache_readonly:
        raise FileNotFoundError(
            f"Mesh voxel cache miss (readonly): {cpath} for {glb_path}"
        )

    position_recon = load_vertices(glb_path)
    coords = vertices_to_coords(position_recon)
    save_cached_coords(
        cpath, coords, glb_path=glb_path, position_recon=position_recon
    )
    return coords, False
