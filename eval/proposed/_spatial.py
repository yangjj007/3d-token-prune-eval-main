"""
Shared spatial utilities for Tier-1 mesh VQ token pruners (8×8×16 = 1024).

Token layout matches ``VQVAE3D``: latent is viewed as ``[bs, 8, 8, 16, 32]`` before VQ;
flatten order is C-contiguous on ``(8, 8, 16)``, i.e. z (last dim) varies fastest::

    flat_index = x * 128 + y * 16 + z,   x,y ∈ [0,7], z ∈ [0,15]
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F

# Grid shape (X, Y, Z) matching VQVAE mesh token tensor view
GRID_X = GRID_Y = 8
GRID_Z = 16
NUM_TOKENS = GRID_X * GRID_Y * GRID_Z  # 1024


def _is_grid_boundary(x: int, y: int, z: int) -> bool:
    """True if voxel lies on the outer shell of the 8×8×16 grid."""
    return (
        x == 0
        or x == GRID_X - 1
        or y == 0
        or y == GRID_Y - 1
        or z == 0
        or z == GRID_Z - 1
    )


def flat_index_to_coord(idx: int | torch.Tensor) -> tuple:
    """Scalar idx -> (x, y, z)."""
    if isinstance(idx, torch.Tensor):
        idx = int(idx.item())
    x = idx // (GRID_Y * GRID_Z)
    rem = idx % (GRID_Y * GRID_Z)
    y = rem // GRID_Z
    z = rem % GRID_Z
    return x, y, z


def coord_to_flat_index(x: int, y: int, z: int) -> int:
    return x * (GRID_Y * GRID_Z) + y * GRID_Z + z


def all_flat_indices(device: torch.device | None = None) -> torch.Tensor:
    """``[1024]`` long indices 0..1023."""
    return torch.arange(NUM_TOKENS, device=device, dtype=torch.long)


def all_coords_tensor(device: torch.device | None = None) -> torch.Tensor:
    """``[1024, 3]`` long with rows ``(x, y, z)`` in flat-index order."""
    idx = all_flat_indices(device)
    x = idx // (GRID_Y * GRID_Z)
    rem = idx % (GRID_Y * GRID_Z)
    y = rem // GRID_Z
    z = rem % GRID_Z
    return torch.stack([x, y, z], dim=-1)


def morton_encode_xyz(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, depth: int = 4) -> torch.Tensor:
    """
    3D Morton (Z-order) key; interleave bits like ``extensions/vox2seq`` KeyLUT.

    ``depth`` bits per dimension (use 4 so z up to 15 fits; x,y only use lower 3 bits).
    """
    x = x.long()
    y = y.long()
    z = z.long()
    key = torch.zeros_like(x)
    for i in range(depth):
        mask = 1 << i
        key = key | ((x & mask) << (2 * i + 2))
        key = key | ((y & mask) << (2 * i + 1))
        key = key | ((z & mask) << (2 * i + 0))
    return key


def _try_hilbert_keys(coords_xyz: torch.Tensor) -> torch.Tensor | None:
    """Hilbert keys ``[N]`` via repo ``extensions/vox2seq`` if importable; else ``None``."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    ext = repo / "extensions" / "vox2seq"
    if ext.is_dir():
        p = str(ext)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from vox2seq.pytorch.hilbert import encode as hilbert_encode_fn  # type: ignore

        return hilbert_encode_fn(coords_xyz.long(), num_dims=3, num_bits=4).reshape(-1)
    except Exception:
        return None


def curve_sort_order(curve_type: Literal["z_order", "hilbert"], device: torch.device) -> torch.Tensor:
    """
    Returns permutation ``perm`` of shape ``[1024]`` such that ``perm[i]`` is the flat index
    of the i-th token along the curve (sorted by curve key).
    """
    coords = all_coords_tensor(device)  # [1024, 3], row i is coord of flat index i
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    if curve_type == "z_order":
        keys = morton_encode_xyz(x, y, z, depth=4)
    elif curve_type == "hilbert":
        keys = _try_hilbert_keys(coords)
        if keys is None:
            keys = morton_encode_xyz(x, y, z, depth=4)
        else:
            keys = keys.to(device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unknown curve_type: {curve_type}")

    _, perm = torch.sort(keys)
    return perm.long()


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Given perm with perm[i]=j, return inv such that inv[j]=i."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def voxel_occupancy_to_latent(voxel_grid: torch.Tensor) -> torch.Tensor:
    """
    Downsample ``[64, 64, 64]`` occupancy to latent ``[8, 8, 16]`` mean occupancy in ``[0, 1]``.

    Each latent cell aggregates an ``8×8×4`` voxel region (8·8=64 / 8, 64/8, 64/16=4 along z).
    """
    v = voxel_grid.detach()
    if v.dim() != 3:
        raise ValueError(f"voxel_grid must be [64,64,64], got {tuple(v.shape)}")
    x = v.float().unsqueeze(0).unsqueeze(0)
    occ = F.avg_pool3d(x, kernel_size=(8, 8, 4))
    return occ[0, 0]


def latent_surface_mask(
    voxel_grid: torch.Tensor,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per latent token (1024 flat): true object surface vs empty vs solid interior.

    Returns:
        is_surface: bool [1024] — occupancy in (eps, 1-eps)
        is_empty: bool [1024] — occupancy <= eps
        is_filled: bool [1024] — occupancy >= 1-eps
        occ_flat: float [1024] — mean occupancy per latent cell
    """
    occ = voxel_occupancy_to_latent(voxel_grid)
    is_surface = ((occ > eps) & (occ < 1.0 - eps)).reshape(-1)
    is_empty = (occ <= eps).reshape(-1)
    is_filled = (occ >= 1.0 - eps).reshape(-1)
    occ_flat = occ.reshape(-1)
    return is_surface, is_empty, is_filled, occ_flat


def per_token_intra_l1_mean_edge_norm(emb_grid: torch.Tensor) -> torch.Tensor:
    """
    For each latent position, mean L2 distance to the other 7 tokens in the same 2×2×2 L1 block.

    ``emb_grid`` shape ``[GRID_X, GRID_Y, GRID_Z, D]``; returns ``[NUM_TOKENS]`` float32 on same device.
    """
    gx, gy, gz, d = emb_grid.shape
    assert (gx, gy, gz) == (GRID_X, GRID_Y, GRID_Z)
    out = torch.zeros(NUM_TOKENS, device=emb_grid.device, dtype=torch.float32)
    for bx in range(4):
        for by in range(4):
            for bz in range(8):
                xs = slice(2 * bx, 2 * bx + 2)
                ys = slice(2 * by, 2 * by + 2)
                zs = slice(2 * bz, 2 * bz + 2)
                block = emb_grid[xs, ys, zs, :].reshape(8, d)
                dists = torch.cdist(block, block, p=2)
                mean_excl_self = (dists.sum(dim=1) / 7.0).clamp_min(0.0)
                for k, (dx, dy, dz) in enumerate(
                    (
                        (0, 0, 0),
                        (0, 0, 1),
                        (0, 1, 0),
                        (0, 1, 1),
                        (1, 0, 0),
                        (1, 0, 1),
                        (1, 1, 0),
                        (1, 1, 1),
                    )
                ):
                    x, y, z = 2 * bx + dx, 2 * by + dy, 2 * bz + dz
                    fi = coord_to_flat_index(x, y, z)
                    out[fi] = mean_excl_self[k].to(dtype=torch.float32)
    return out
