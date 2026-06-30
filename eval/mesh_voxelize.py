"""Open3D/trimesh mesh voxelization (shared; no imports from data_loader)."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import torch
import trimesh

GRID_RES = 64


class UnsupportedMeshError(ValueError):
    """GLB contains no voxelizable triangle mesh (e.g. Path3D-only)."""


def _load_triangle_mesh(filepath: str) -> trimesh.Trimesh:
    """Load a GLB/OBJ and return a single triangle mesh (concatenate scene parts)."""
    loaded = trimesh.load(filepath, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        candidates = [loaded]
    elif isinstance(loaded, trimesh.Scene):
        candidates = [
            g
            for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
        ]
    else:
        candidates = []

    if not candidates:
        raise UnsupportedMeshError(
            f"No triangle mesh in {filepath} (top-level type: {type(loaded).__name__})"
        )
    mesh = candidates[0] if len(candidates) == 1 else trimesh.util.concatenate(candidates)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise UnsupportedMeshError(f"Empty triangle mesh in {filepath}")
    return mesh


def convert_trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(trimesh_mesh.faces, dtype=np.int32)
    )
    return o3d_mesh


def rotate_points(points, axis="x", angle_deg=90):
    angle_rad = np.deg2rad(angle_deg)
    if axis == "x":
        R = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])[:3, :3]
    elif axis == "y":
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])[:3, :3]
    elif axis == "z":
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])[:3, :3]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return points @ R.T


def load_vertices(filepath: str) -> np.ndarray:
    """Match archived demo voxelization: normalized mesh -> voxel centers -> x-rotation."""
    mesh = convert_trimesh_to_open3d(_load_triangle_mesh(filepath))
    vertices = np.asarray(mesh.vertices)
    min_vals = vertices.min()
    max_vals = vertices.max()
    vertices_normalized = (vertices - min_vals) / (max_vals - min_vals)
    vertices = vertices_normalized * 1.0 - 0.5
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / GRID_RES,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    if not (np.all(vertices >= 0) and np.all(vertices < GRID_RES)):
        raise ValueError("Some vertices are out of bounds after voxelization")
    vertices = (vertices + 0.5) / GRID_RES - 0.5
    return rotate_points(vertices, axis="x", angle_deg=90)


def vertices_to_coords(position_recon: np.ndarray) -> np.ndarray:
    """Same indexing as ``mesh_to_tokens``: int coords in ``[0, GRID_RES-1]^3``."""
    coords = ((torch.from_numpy(position_recon) + 0.5) * GRID_RES).int().numpy()
    return np.ascontiguousarray(coords.astype(np.int32))
