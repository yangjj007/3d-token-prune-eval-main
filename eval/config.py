"""CLI and defaults for token pruning evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@dataclass
class EvalConfig:
    data_csv: str = "data/metadata.csv"
    glb_dir: str = "sampled_objaverse_data"
    model_backend: str = "shapellm"
    model_id: str = "yejunliang23/ShapeLLM-7B-omni"
    eva01_model_id: str = "SEELE-AI/EVA01-2B-Instruct-LoRA"
    eva01_base_model_name_or_path: str = ""
    vqvae_repo: str = "yejunliang23/3DVQVAE"
    vqvae_filename: str = "3DVQVAE.bin"
    caption_prompt: str = "Give a quick overview of the object represented by this 3D mesh."
    pruners: List[str] = field(default_factory=lambda: ["no_pruning", "random", "uniform"])
    keep_ratios: List[float] = field(default_factory=lambda: [1.0, 0.9, 0.7, 0.5, 0.3, 0.1])
    seed: int = 42
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 8192
    output_dir: str = "eval/results"
    num_samples: int = -1
    device: str = "cuda:0"
    # VQVAE encode device: cuda:1 for dual-GPU (VLM on cuda:0), or cpu if OOM.
    vqvae_device: str = "cuda:1"
    mesh_cache_dir: str = ""
    mesh_cache_readonly: bool = False
    mesh_prefetch_workers: int = 2
    # Debug-only path: skip model/VQVAE loading and emit deterministic captions.
    mock_model: bool = False
    # Default: full bf16/fp16 VLM on GPU (matches typical inference). Use --load-in-4bit if VRAM is tight.
    load_in_4bit: bool = False
    # When not using 4bit: auto = bf16 if supported else fp16; or force float16 / bfloat16.
    vlm_torch_dtype: str = "auto"
    # Per-pruner JSON: ``{eval_config_dir}/{pruner_name}.json`` merged into ``BasePruner(..., **kwargs)``.
    eval_config_dir: str = "configs/eval"

    @staticmethod
    def from_args(argv: List[str] | None = None) -> "EvalConfig":
        p = argparse.ArgumentParser(description="ShapeLLM-Omni mesh token pruning caption eval")
        p.add_argument("--data-csv", type=str, default="data/metadata.csv")
        p.add_argument("--glb-dir", type=str, default="sampled_objaverse_data")
        p.add_argument(
            "--model-backend",
            type=str,
            default="shapellm",
            choices=("shapellm", "eva01"),
            help="Caption backend. shapellm keeps the existing VQVAE+ShapeLLM path; eva01 uses OpenEVA EVA01.",
        )
        p.add_argument("--model-id", type=str, default="yejunliang23/ShapeLLM-7B-omni")
        p.add_argument(
            "--eva01-model-id",
            type=str,
            default="SEELE-AI/EVA01-2B-Instruct-LoRA",
            help="EVA01 checkpoint for --model-backend eva01.",
        )
        p.add_argument(
            "--eva01-base-model-name-or-path",
            type=str,
            default="",
            help="Optional EVA01 base model path passed through to OpenEVA for LoRA checkpoints.",
        )
        p.add_argument("--output-dir", type=str, default="eval/results")
        p.add_argument(
            "--pruners",
            type=str,
            default="no_pruning,random,uniform",
            help="Comma-separated pruner names (registered in eval.pruners)",
        )
        p.add_argument("--keep-ratios", type=str, default="1.0,0.9,0.7,0.5,0.3,0.1")
        p.add_argument("--num-samples", type=int, default=-1, help="-1 = all available with GLB (default)")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--max-new-tokens", type=int, default=512)
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--top-p", type=float, default=0.7)
        p.add_argument("--top-k", type=int, default=8192)
        p.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="VLM device, e.g. cuda:0",
        )
        p.add_argument(
            "--vqvae-device",
            type=str,
            default="cuda:1",
            help="VQVAE encode device: cuda:1 (dual GPU), cuda, cpu, etc.",
        )
        p.add_argument(
            "--mesh-cache-dir",
            type=str,
            default="",
            help="Directory of precomputed {id}.npz voxel coords (skip Open3D on hit).",
        )
        p.add_argument(
            "--mesh-cache-readonly",
            action="store_true",
            help="Do not write mesh voxel cache; fail if cache miss.",
        )
        p.add_argument(
            "--mesh-prefetch-workers",
            type=int,
            default=2,
            help="Thread workers to prefetch next sample coords (0=off).",
        )
        p.add_argument(
            "--mock-model",
            action="store_true",
            help="Debug only: skip model/VQVAE loading and emit deterministic mock captions.",
        )
        p.add_argument(
            "--load-in-4bit",
            action="store_true",
            help="Load VLM in 4-bit NF4 (saves VRAM). Default is full bf16/fp16 on CUDA.",
        )
        p.add_argument(
            "--vlm-torch-dtype",
            type=str,
            default="auto",
            choices=("auto", "float16", "bfloat16"),
            help="VLM weight dtype when not using --load-in-4bit. auto: bf16 if GPU supports else fp16.",
        )
        p.add_argument("--caption-prompt", type=str, default=EvalConfig.caption_prompt)
        p.add_argument(
            "--eval-config-dir",
            type=str,
            default="configs/eval",
            help="Directory with optional ``{pruner_name}.json`` extra kwargs per method.",
        )
        args = p.parse_args(argv)
        return EvalConfig(
            data_csv=args.data_csv,
            glb_dir=args.glb_dir,
            model_backend=args.model_backend,
            model_id=args.model_id,
            eva01_model_id=args.eva01_model_id,
            eva01_base_model_name_or_path=args.eva01_base_model_name_or_path,
            output_dir=args.output_dir,
            pruners=_parse_str_list(args.pruners),
            keep_ratios=_parse_float_list(args.keep_ratios),
            num_samples=args.num_samples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=args.device,
            caption_prompt=args.caption_prompt,
            vqvae_device=args.vqvae_device,
            mesh_cache_dir=args.mesh_cache_dir,
            mesh_cache_readonly=args.mesh_cache_readonly,
            mesh_prefetch_workers=args.mesh_prefetch_workers,
            mock_model=args.mock_model,
            load_in_4bit=args.load_in_4bit,
            vlm_torch_dtype=args.vlm_torch_dtype,
            eval_config_dir=args.eval_config_dir,
        )


def load_pruner_extra_kwargs(config_dir: Path, pruner_name: str) -> Dict[str, Any]:
    """Load ``{config_dir}/{pruner_name}.json`` if present; otherwise empty dict."""
    path = config_dir / f"{pruner_name}.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data)}")
    return data


def resolve_repo_paths(cfg: EvalConfig, repo_root: Path) -> EvalConfig:
    """Turn relative paths into absolute paths under ``repo_root``."""

    def _abs(p: str) -> str:
        pp = Path(p)
        return str(pp) if pp.is_absolute() else str((repo_root / p).resolve())

    cfg.data_csv = _abs(cfg.data_csv)
    cfg.glb_dir = _abs(cfg.glb_dir)
    cfg.output_dir = _abs(cfg.output_dir)
    cfg.eval_config_dir = _abs(cfg.eval_config_dir)
    if cfg.mesh_cache_dir:
        cfg.mesh_cache_dir = _abs(cfg.mesh_cache_dir)
    return cfg
