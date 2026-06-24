"""Load Objaverse-style metadata and convert meshes to VQ mesh tokens."""

from __future__ import annotations

import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urlparse

import numpy as np
import torch

from eval.cuda_env import release_cuda_device_memory, vqvae_encode
from eval.mesh_voxelize import GRID_RES, load_vertices  # noqa: F401 — re-export
from eval.progress import log_phase, phase_timer
from eval.voxel_cache import resolve_coords
from trellis.models.sparse_structure_vqvae import VQVAE3D


@dataclass
class MeshSample:
    file_identifier: str
    glb_path: str
    captions: List[str]
    sha256: Optional[str] = None


def occupancy_from_coords(
    coords: torch.Tensor | np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``ss`` [1,1,64,64,64] on device and ``voxel_grid`` [64,64,64] long on CPU."""
    if isinstance(coords, np.ndarray):
        coords_t = torch.from_numpy(coords).int()
    else:
        coords_t = coords.int().cpu()
    voxel_grid = torch.zeros(GRID_RES, GRID_RES, GRID_RES, dtype=torch.long)
    if coords_t.numel() > 0:
        voxel_grid[coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = 1
    ss = torch.zeros(
        1, 1, GRID_RES, GRID_RES, GRID_RES, device=device, dtype=torch.float32
    )
    if coords_t.numel() > 0:
        c = coords_t.to(device=device, non_blocking=device.type == "cuda")
        ss[0, 0, c[:, 0], c[:, 1], c[:, 2]] = 1.0
    return ss, voxel_grid


def prepare_mesh_coords(
    filepath: str,
    file_identifier: str,
    mesh_cache_dir: str = "",
    *,
    mesh_cache_readonly: bool = False,
) -> np.ndarray:
    """Load or voxelize mesh; return ``coords`` int32 ``[N,3]`` (CPU, for prefetch)."""
    coords, from_cache = resolve_coords(
        filepath,
        file_identifier,
        mesh_cache_dir,
        cache_readonly=mesh_cache_readonly,
    )
    if from_cache:
        log_phase(f"mesh voxelize cache hit id={file_identifier} n={coords.shape[0]}")
    return coords


def mesh_to_tokens(
    filepath: str,
    vqvae: VQVAE3D,
    device: torch.device,
    token_head_dump_path: Optional[str] = None,
    *,
    file_identifier: str = "",
    mesh_cache_dir: str = "",
    mesh_cache_readonly: bool = False,
    prefetched_coords: Optional[np.ndarray] = None,
    vlm_device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        token_ids: Long tensor ``[1024]`` on CPU (for pruning).
        voxel_grid: Long tensor ``[64, 64, 64]`` occupancy on CPU.

    If ``token_head_dump_path`` is set, writes JSON with ``token_ids[:16]``
    (§6.0.3 / 6.7.3 前 N 样本统计雷同排查)。
    """
    fid = file_identifier or Path(filepath).stem
    log_phase(f"mesh_to_tokens glb={filepath}")
    if prefetched_coords is not None:
        coords = prefetched_coords
        log_phase(f"mesh voxelize prefetched id={fid} n={coords.shape[0]}")
    else:
        label = "mesh voxelize (cache)" if mesh_cache_dir else "mesh voxelize (Open3D/trimesh, CPU)"
        with phase_timer(label):
            coords, from_cache = resolve_coords(
                filepath,
                fid,
                mesh_cache_dir,
                cache_readonly=mesh_cache_readonly,
            )
            if from_cache:
                log_phase(f"cache hit id={fid} n={coords.shape[0]}")

    with phase_timer(f"build occupancy + VQVAE Encode (device={device})"):
        ss, voxel_grid = occupancy_from_coords(coords, device)
        try:
            enc = vqvae_encode(vqvae, ss, device, vlm_dev=vlm_device)
            token_ids = enc.reshape(-1).detach().cpu().long().view(1024)
        finally:
            del ss, enc
            if device.type == "cuda":
                release_cuda_device_memory(device)
    if token_head_dump_path:
        dump_dir = os.path.dirname(os.path.abspath(token_head_dump_path))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        head = [int(x) for x in token_ids[:16].tolist()]
        payload = {"glb_path": filepath, "token_ids_head16": head}
        with open(token_head_dump_path, "w", encoding="utf-8") as wf:
            json.dump(payload, wf, ensure_ascii=False, indent=2)
    return token_ids, voxel_grid


def _parse_captions_cell(cell: str) -> List[str]:
    """Parse CSV ``captions`` field: JSON array of strings, or fallback."""
    cell = cell.strip()
    if not cell:
        return []

    def _from_list(data) -> List[str]:
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []

    try:
        return _from_list(json.loads(cell))
    except json.JSONDecodeError:
        pass
    try:
        return _from_list(ast.literal_eval(cell))
    except (ValueError, SyntaxError):
        pass
    return [cell]


def _normalize_file_identifier(raw: str) -> str:
    """
    Map CSV ``file_identifier`` to the basename used for ``{id}.glb``.

    Objaverse-style CSVs use a plain hex id; some exports store a Sketchfab
    model URL instead — take the last URL path segment and the trailing 32 hex
    chars when present (``.../title-abc...def``).
    """
    s = (raw or "").strip()
    if not s:
        return ""
    low = s.lower()
    if low.startswith("http://") or low.startswith("https://"):
        last = urlparse(s).path.rstrip("/").split("/")[-1]
        m = re.search(r"([0-9a-f]{32})$", last)
        return m.group(1).lower() if m else last.lower()
    return s


def _glb_stem_candidates(row: dict) -> List[str]:
    """
    Possible ``{stem}.glb`` basenames for one CSV row (first existing file wins).

    Some dumps use Sketchfab URLs in ``file_identifier`` (resolved to the 32-hex
    id) while meshes on disk are named ``{sha256}.glb`` (64-hex content id).
    """
    out: List[str] = []
    seen: set[str] = set()

    def add(stem: str) -> None:
        t = (stem or "").strip()
        if not t or t in seen:
            return
        seen.add(t)
        out.append(t)

    add(_normalize_file_identifier(str(row.get("file_identifier", "") or "")))

    sha = row.get("sha256")
    if sha is not None:
        s = str(sha).strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", s):
            add(s)

    return out


def iter_dataset(
    csv_path: str,
    glb_dir: str,
    num_samples: int = -1,
    skip_missing_glb: bool = True,
) -> Iterator[MeshSample]:
    """
    Yield samples with an existing ``.glb`` under ``glb_dir``.

    Basename is the first hit among stems from :func:`_glb_stem_candidates`
    (normalized ``file_identifier``, then ``sha256`` when present).
    """
    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stems = _glb_stem_candidates(row)
            if not stems:
                continue
            glb_path: Optional[str] = None
            resolved_stem: Optional[str] = None
            for stem in stems:
                cand = os.path.join(glb_dir, f"{stem}.glb")
                if os.path.isfile(cand):
                    glb_path = cand
                    resolved_stem = stem
                    break
            if glb_path is None or resolved_stem is None:
                if skip_missing_glb:
                    continue
                tried = ", ".join(os.path.join(glb_dir, f"{s}.glb") for s in stems)
                raise FileNotFoundError(f"no glb found; tried: {tried}")
            caps = _parse_captions_cell(row.get("captions", ""))
            sha = row.get("sha256")
            yield MeshSample(
                file_identifier=resolved_stem,
                glb_path=os.path.abspath(glb_path),
                captions=caps,
                sha256=sha,
            )
            count += 1
            if num_samples > 0 and count >= num_samples:
                break


def list_dataset_paths(csv_path: str, glb_dir: str) -> List[MeshSample]:
    return list(iter_dataset(csv_path, glb_dir, num_samples=-1, skip_missing_glb=True))
