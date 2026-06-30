"""CLI and defaults for token pruning evaluation."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _coerce_float_list(value: Any) -> List[float]:
    if isinstance(value, str):
        return _parse_float_list(value)
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(f"Expected a comma-separated string or list of floats, got {type(value)}")


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return _parse_str_list(value)
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    raise TypeError(f"Expected a comma-separated string or list of strings, got {type(value)}")


def _load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment packaging
        raise RuntimeError("YAML config requires PyYAML. Install it with `pip install PyYAML`.") from exc

    cfg_path = Path(path)
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {cfg_path}, got {type(data)}")
    if "eval" in data:
        data = data["eval"] or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected `eval` YAML object in {cfg_path}, got {type(data)}")
    return dict(data)


def _apply_overrides(cfg: "EvalConfig", values: Dict[str, Any], *, source: str) -> "EvalConfig":
    allowed = {f.name for f in fields(EvalConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        known = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown config key(s) in {source}: {', '.join(unknown)}. Known keys: {known}")

    for key, value in values.items():
        if value is None:
            continue
        if key == "pruners":
            value = _coerce_str_list(value)
        elif key == "keep_ratios":
            value = _coerce_float_list(value)
        elif key == "env":
            if not isinstance(value, dict):
                raise TypeError(f"`env` in {source} must be a mapping, got {type(value)}")
            value = {str(k): str(v) for k, v in value.items()}
        setattr(cfg, key, value)
    return cfg


@dataclass
class EvalConfig:
    data_csv: str = "../data/metadata.csv"
    glb_dir: str = "../data"
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
    output_dir: str = "../output/eval-results"
    run_log_file: str = ""
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
    # Optional environment variables applied at run start, after config parsing.
    env: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_args(argv: List[str] | None = None) -> "EvalConfig":
        p = argparse.ArgumentParser(description="ShapeLLM-Omni mesh token pruning caption eval")
        p.add_argument("--config", type=str, default="", help="YAML run config under configs/runs/. CLI flags override it.")
        p.add_argument("--data-csv", type=str, default=None)
        p.add_argument("--glb-dir", type=str, default=None)
        p.add_argument(
            "--model-backend",
            type=str,
            default=None,
            choices=("shapellm", "eva01"),
            help="Caption backend. shapellm keeps the existing VQVAE+ShapeLLM path; eva01 uses OpenEVA EVA01.",
        )
        p.add_argument("--model-id", type=str, default=None)
        p.add_argument(
            "--eva01-model-id",
            type=str,
            default=None,
            help="EVA01 checkpoint for --model-backend eva01.",
        )
        p.add_argument(
            "--eva01-base-model-name-or-path",
            type=str,
            default=None,
            help="Optional EVA01 base model path passed through to OpenEVA for LoRA checkpoints.",
        )
        p.add_argument("--output-dir", type=str, default=None)
        p.add_argument("--run-log-file", type=str, default=None, help="Optional stdout/stderr tee log path.")
        p.add_argument(
            "--pruners",
            type=str,
            default=None,
            help="Comma-separated pruner names (registered in eval.pruners)",
        )
        p.add_argument("--keep-ratios", type=str, default=None)
        p.add_argument("--num-samples", type=int, default=None, help="-1 = all available with GLB")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--max-new-tokens", type=int, default=None)
        p.add_argument("--temperature", type=float, default=None)
        p.add_argument("--top-p", type=float, default=None)
        p.add_argument("--top-k", type=int, default=None)
        p.add_argument(
            "--device",
            type=str,
            default=None,
            help="VLM device, e.g. cuda:0",
        )
        p.add_argument(
            "--vqvae-device",
            type=str,
            default=None,
            help="VQVAE encode device: cuda:1 (dual GPU), cuda, cpu, etc.",
        )
        p.add_argument(
            "--mesh-cache-dir",
            type=str,
            default=None,
            help="Directory of precomputed {id}.npz voxel coords (skip Open3D on hit).",
        )
        p.add_argument(
            "--mesh-cache-readonly",
            action="store_true",
            default=None,
            help="Do not write mesh voxel cache; fail if cache miss.",
        )
        p.add_argument(
            "--mesh-prefetch-workers",
            type=int,
            default=None,
            help="Thread workers to prefetch next sample coords (0=off).",
        )
        p.add_argument(
            "--mock-model",
            action="store_true",
            default=None,
            help="Debug only: skip model/VQVAE loading and emit deterministic mock captions.",
        )
        p.add_argument(
            "--load-in-4bit",
            action="store_true",
            default=None,
            help="Load VLM in 4-bit NF4 (saves VRAM). Default is full bf16/fp16 on CUDA.",
        )
        p.add_argument(
            "--vlm-torch-dtype",
            type=str,
            default=None,
            choices=("auto", "float16", "bfloat16"),
            help="VLM weight dtype when not using --load-in-4bit. auto: bf16 if GPU supports else fp16.",
        )
        p.add_argument("--caption-prompt", type=str, default=None)
        p.add_argument(
            "--eval-config-dir",
            type=str,
            default=None,
            help="Directory with optional ``{pruner_name}.json`` extra kwargs per method.",
        )
        args = p.parse_args(argv)
        cfg = EvalConfig()
        if args.config:
            cfg = _apply_overrides(cfg, _load_yaml_config(args.config), source=args.config)

        cli_values = vars(args).copy()
        cli_values.pop("config", None)
        cli_values = {k: v for k, v in cli_values.items() if v is not None}
        return _apply_overrides(cfg, cli_values, source="CLI")


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
    if cfg.run_log_file:
        cfg.run_log_file = _abs(cfg.run_log_file)
    cfg.eval_config_dir = _abs(cfg.eval_config_dir)
    if cfg.mesh_cache_dir:
        cfg.mesh_cache_dir = _abs(cfg.mesh_cache_dir)
    return cfg
