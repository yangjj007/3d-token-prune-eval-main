"""CUDA device helpers for VQVAE encode (dual-GPU eval, cudnn warmup)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from trellis.models.sparse_structure_vqvae import VQVAE3D

# Per-GPU cuDNN probe cache (False -> use reference conv3d for VQVAE on that GPU).
_CUDNN_DEVICE_OK: dict[int, bool] = {}


def resolve_torch_device(spec: str) -> torch.device:
    """Parse ``cpu``, ``cuda``, ``cuda:0``, ``cuda:1``, etc."""
    s = (spec or "cpu").strip()
    if s == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda:0")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {s!r} but CUDA is not available")
        dev = torch.device(s)
        idx = dev.index if dev.index is not None else 0
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {s!r} but only {torch.cuda.device_count()} GPU(s) visible "
                f"(check CUDA_VISIBLE_DEVICES)"
            )
        return dev
    return torch.device(s)


def _cudnn_benchmark_enabled() -> bool:
    return os.environ.get("SHAPELLM_CUDNN_BENCHMARK", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def _cuda_index(dev: torch.device) -> int:
    return dev.index if dev.index is not None else 0


def probe_cudnn_on_device(dev: torch.device) -> bool:
    """Return True if a tiny conv3d can run with cuDNN on ``dev`` (never raises)."""
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False
    if not torch.backends.cudnn.is_available():
        return False
    idx = _cuda_index(dev)
    if idx in _CUDNN_DEVICE_OK:
        return _CUDNN_DEVICE_OK[idx]
    ok = False
    try:
        with torch.cuda.device(idx):
            torch.cuda.set_device(idx)
            probe = torch.zeros(1, 1, 4, 4, 4, device=dev, dtype=torch.float32)
            weight = torch.zeros(1, 1, 3, 3, 3, device=dev, dtype=torch.float32)
            with torch.inference_mode():
                torch.nn.functional.conv3d(probe, weight, padding=1)
            torch.cuda.synchronize(dev)
        ok = True
    except RuntimeError:
        ok = False
    _CUDNN_DEVICE_OK[idx] = ok
    return ok


def _vqvae_use_reference_conv(vqvae_dev: torch.device, vlm_dev: torch.device | None) -> bool:
    """VQVAE conv3d without cuDNN (required on many dual-GPU / container setups)."""
    if vqvae_dev.type != "cuda":
        return False
    if os.environ.get("SHAPELLM_VQVAE_NO_CUDNN", "").lower() in ("1", "true", "yes"):
        return True
    if vlm_dev is not None and vlm_dev.type == "cuda" and vqvae_dev != vlm_dev:
        return True
    return not probe_cudnn_on_device(vqvae_dev)


@contextmanager
def vqvae_cuda_exec(
    vqvae_dev: torch.device,
    vlm_dev: torch.device | None = None,
) -> Iterator[None]:
    """Set VQVAE CUDA device; temporarily disable cuDNN when the probe failed or GPUs differ."""
    ref_conv = _vqvae_use_reference_conv(vqvae_dev, vlm_dev)
    prev_cudnn = torch.backends.cudnn.enabled
    try:
        if vqvae_dev.type == "cuda":
            with torch.cuda.device(_cuda_index(vqvae_dev)):
                torch.cuda.set_device(_cuda_index(vqvae_dev))
                if ref_conv:
                    torch.backends.cudnn.enabled = False
                yield
        else:
            yield
    finally:
        torch.backends.cudnn.enabled = prev_cudnn


def init_cuda_for_eval(vqvae_dev: torch.device, vlm_dev: torch.device) -> None:
    """Configure cudnn and log device layout."""
    seen: set[int] = set()
    for dev in (vqvae_dev, vlm_dev):
        if dev.type != "cuda":
            continue
        idx = _cuda_index(dev)
        if idx in seen:
            continue
        seen.add(idx)
        ok = probe_cudnn_on_device(dev)
        _log_phase(f"cuda:{idx} cuDNN probe: {'ok' if ok else 'unavailable (reference conv)'}")

    # VLM / flash-attn need cuDNN enabled globally before load & generate.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True

    if _cudnn_benchmark_enabled() and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if vqvae_dev.type == "cuda":
        _log_phase(f"VQVAE will use {vqvae_dev}")
    if vlm_dev.type == "cuda":
        _log_phase(f"VLM will use {vlm_dev}")
    if _vqvae_use_reference_conv(vqvae_dev, vlm_dev):
        _log_phase(
            "VQVAE encode uses reference conv3d (cuDNN off on VQVAE GPU only); "
            "VLM keeps cuDNN enabled"
        )
    if vqvae_dev.type == "cuda" and vlm_dev.type == "cuda" and vqvae_dev == vlm_dev:
        _log_phase("VQVAE and VLM share the same GPU; use --load-in-4bit if OOM")


def warmup_vqvae(
    vqvae: VQVAE3D,
    vqvae_dev: torch.device,
    *,
    vlm_dev: torch.device | None = None,
) -> None:
    """Run one dummy Encode on the VQVAE device."""
    if vqvae_dev.type != "cuda":
        return
    _log_phase(f"VQVAE warmup on {vqvae_dev}")
    dummy = torch.zeros(1, 1, 64, 64, 64, device=vqvae_dev, dtype=torch.float32)
    with torch.inference_mode():
        vqvae_encode(vqvae, dummy, vqvae_dev, vlm_dev=vlm_dev)
    torch.cuda.synchronize(vqvae_dev)
    _log_phase(f"VQVAE warmup done on {vqvae_dev}")


def release_cuda_device_memory(dev: torch.device) -> None:
    """Synchronize and return cached blocks on ``dev`` (no-op on CPU)."""
    if dev.type != "cuda" or not torch.cuda.is_available():
        return
    idx = _cuda_index(dev)
    with torch.cuda.device(idx):
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()


def vqvae_encode(
    vqvae: VQVAE3D,
    ss: torch.Tensor,
    vqvae_dev: torch.device,
    *,
    vlm_dev: torch.device | None = None,
) -> torch.Tensor:
    """
    Encode occupancy ``ss`` with shape ``[1, 1, 64, 64, 64]`` (or broadcastable).

    Uses fp16 autocast on CUDA. If VQVAE shares a GPU with VLM, clears cache first.
    """
    if vqvae_dev.type == "cuda" and vlm_dev is not None and vlm_dev.type == "cuda":
        if vqvae_dev == vlm_dev:
            torch.cuda.empty_cache()

    x = ss.to(device=vqvae_dev, dtype=torch.float32)
    if x.dim() == 4:
        x = x.unsqueeze(0)
    if x.dim() != 5 or x.shape[1] != 1:
        raise ValueError(f"Expected ss shape [B,1,64,64,64], got {tuple(x.shape)}")

    try:
        with torch.inference_mode():
            with vqvae_cuda_exec(vqvae_dev, vlm_dev):
                if vqvae_dev.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = vqvae.Encode(x)
                else:
                    out = vqvae.Encode(x)
    finally:
        del x
        if vqvae_dev.type == "cuda" and vlm_dev is not None and vqvae_dev != vlm_dev:
            release_cuda_device_memory(vqvae_dev)
    return out


def _log_phase(msg: str) -> None:
    from eval.progress import log_phase

    log_phase(msg)

