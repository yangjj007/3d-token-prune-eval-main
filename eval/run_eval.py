"""
Run mesh token pruning caption evaluation.

Usage (from ``3d-token-prune-eval-main``)::

    python -m eval.run_eval --config configs/runs/eva01-full.yaml

Ensure ``eval.pruners.baseline`` and ``eval.baseline`` are importable to register
built-in pruners; add new modules under those packages and import them here.
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback
from pathlib import Path

# Trellis / spconv expect this in many setups
os.environ.setdefault("SPCONV_ALGO", "native")
# Load only ``trellis.models`` + ``modules`` for VQVAE (see ``trellis/__init__.py``)
os.environ["SHAPELLM_EVAL_LIGHT"] = "1"


def _configure_hf_cache() -> None:
    from eval.hf_env import apply_hf_env

    apply_hf_env()


_configure_hf_cache()

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.config import EvalConfig, load_pruner_extra_kwargs, resolve_repo_paths
from eval.cuda_env import (
    init_cuda_for_eval,
    release_cuda_device_memory,
    resolve_torch_device,
    warmup_vqvae,
)
from eval.data_loader import iter_dataset, mesh_to_tokens, prepare_mesh_coords
from eval.eva01_backend import (
    EVA01_VQVAE_SPATIAL_PRUNERS,
    extract_eva01_mesh_features,
    generate_eva01_caption_from_mesh_tokens,
    map_vq_indices_to_eva_patches,
    prune_eva01_patch_embeddings,
    select_eva01_mesh_tokens,
    target_eva01_patch_count,
)
from eval.generator import generate_caption
from eval.progress import log_phase, phase_timer, progress_enabled
from eval.flops import enrich_pruner_metadata_flops, estimate_llm_tflops
from eval.metrics import compute_text_metrics
from eval.pruners import ensure_pruners_loaded, get_pruner_class
from eval.utils import (
    aggregate_summary,
    compute_pct_of_full,
    ensure_dir,
    save_json,
    split_results_by_pruner,
    tokens_to_mesh_string,
    write_summary_csv,
)

# Register all pruners (lazy loader also runs inside get_pruner_class)
ensure_pruners_loaded()


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def _apply_config_env(cfg: EvalConfig) -> None:
    for key, value in cfg.env.items():
        os.environ[str(key)] = str(value)


def _setup_run_log(path: str) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    log_f = open(path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, log_f)
    sys.stderr = _TeeStream(sys.stderr, log_f)
    print(f"run_log_file={path}", flush=True)


def load_vqvae(device: torch.device):
    from trellis.models.sparse_structure_vqvae import VQVAE3D

    vqvae = VQVAE3D(num_embeddings=8192)
    vqvae.eval()
    fp = hf_hub_download(repo_id="yejunliang23/3DVQVAE", filename="3DVQVAE.bin")
    state = torch.load(fp, map_location=device)
    vqvae.load_state_dict(state)
    vqvae = vqvae.to(device)
    return vqvae


def load_llm(
    model_id: str,
    device: torch.device,
    *,
    load_in_4bit: bool = False,
    vlm_torch_dtype: str = "auto",
):
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    if device.type != "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        model = model.to(device)
        return model, processor, tokenizer

    vlm_map = device.index if device.index is not None else 0
    if load_in_4bit:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=qconfig,
            device_map={"": vlm_map},
            torch_dtype=torch.bfloat16,
        )
    else:
        if vlm_torch_dtype == "float16":
            dtype = torch.float16
        elif vlm_torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            # auto: bf16 preferred; fp16 fallback
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map={"": vlm_map},
        )
    return model, processor, tokenizer


def validate_eva01_eval_config(cfg: EvalConfig) -> None:
    """Validate EVA01 eval options.

    Real EVA01 pruning is implemented through the patch-embedding adapter. Unknown
    pruner names are already rejected before this function is called.
    """
    if not cfg.pruners:
        raise ValueError("EVA01 eval requires at least one pruner.")


def _finalize_results(cfg: EvalConfig, results: list[dict], *, skipped_samples: int = 0) -> int:
    out_json = os.path.join(cfg.output_dir, "results.json")
    save_json(out_json, results)

    ok = [r for r in results if "generated_caption" in r]
    summary = aggregate_summary(ok)
    summary = compute_pct_of_full(summary)
    out_csv = os.path.join(cfg.output_dir, "summary.csv")
    write_summary_csv(out_csv, summary)

    by_pruner_paths = split_results_by_pruner(cfg.output_dir, results)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {len(by_pruner_paths)} files under {os.path.join(cfg.output_dir, 'by_pruner')}")
    if skipped_samples:
        print(f"Skipped {skipped_samples} mesh sample(s) due to load/eval errors.", flush=True)

    if ok and Path(out_csv).is_file():
        from eval.score_eval_run import score_from_summary_file

        for pname in cfg.pruners:
            score, extras = score_from_summary_file(Path(out_csv), pname)
            print(f"eval_score={score:.6f}", flush=True)
            print(f"rouge_l_mean={extras.get('rouge_l_mean', 0.0):.6f}", flush=True)
            print(f"bleu_4_mean={extras.get('bleu_4_mean', 0.0):.6f}", flush=True)
            print(f"bleu_1_mean={extras.get('bleu_1_mean', 0.0):.6f}", flush=True)
            print(f"sentence_bert_mean={extras.get('sentence_bert_mean', 0.0):.6f}", flush=True)
            print(f"simcse_mean={extras.get('simcse_mean', 0.0):.6f}", flush=True)
            print(f"pruner={pname}", flush=True)

    return 0 if ok else 1



def _mock_eva01_caption(sample, pruner: str, keep_ratio: float) -> str:
    if sample.captions:
        return str(sample.captions[0])
    return (
        f"Mock EVA01 caption for {sample.file_identifier} "
        f"using {pruner} at keep_ratio={float(keep_ratio):g}."
    )


def _mock_eva01_token_counts(pruner: str, keep_ratio: float) -> tuple[int, int]:
    n_orig = 513
    if pruner == "no_pruning":
        return n_orig, n_orig
    n_patch = target_eva01_patch_count(float(keep_ratio), 512)
    return n_orig, 1 + n_patch


def _run_eva01_eval(
    cfg: EvalConfig,
    samples: list,
    model,
    processor,
    device: torch.device,
    *,
    vqvae=None,
    vqvae_dev: torch.device = torch.device("cpu"),
    eval_cfg_dir: Path | None = None,
) -> list[dict]:
    eval_cfg_dir = eval_cfg_dir or Path(cfg.eval_config_dir)
    uses_spatial_vq = any(p in EVA01_VQVAE_SPATIAL_PRUNERS for p in cfg.pruners)
    vq_emb = getattr(getattr(vqvae, "vq", None), "embeddings", None) if vqvae is not None else None
    vq_embed_dim = int(vq_emb.weight.shape[1]) if vq_emb is not None else 32
    vq_codebook_size = int(vq_emb.weight.shape[0]) if vq_emb is not None else 8192

    os.environ.setdefault("SHAPELLM_EVAL_LOG_DIR", os.path.join(cfg.output_dir, "logs"))
    os.environ.setdefault("SHAPELLM_EVAL_LOG_DEEP_EVERY", "20")
    os.makedirs(os.environ["SHAPELLM_EVAL_LOG_DIR"], exist_ok=True)
    if uses_spatial_vq:
        print(
            f"EVA01 spatial-pruner diagnostics -> SHAPELLM_EVAL_LOG_DIR={os.environ['SHAPELLM_EVAL_LOG_DIR']} "
            f"(deep_every={os.environ['SHAPELLM_EVAL_LOG_DEEP_EVERY']})"
        )

    results: list[dict] = []
    sample_iter = enumerate(samples)
    if progress_enabled() and len(samples) > 1:
        from tqdm import tqdm

        sample_iter = enumerate(tqdm(samples, desc="eval meshes", unit="mesh"))

    log_phase(
        f"EVA01 sample loop: {len(samples)} meshes "
        f"(pruners={cfg.pruners}, keep_ratios={cfg.keep_ratios}); device={device}"
    )
    for si, sample in sample_iter:
        print(f"[{si+1}/{len(samples)}] {sample.file_identifier}", flush=True)
        eva_features = None
        feature_error = None
        token_ids = None
        voxel_grid = None
        if not cfg.mock_model:
            try:
                with phase_timer(f"EVA01 mesh encoder glb={sample.glb_path}"):
                    eva_features = extract_eva01_mesh_features(
                        model,
                        processor,
                        sample.glb_path,
                        cfg.caption_prompt,
                        device=device,
                    )
            except Exception:
                feature_error = traceback.format_exc()
                print(feature_error, file=sys.stderr)

        for pname in cfg.pruners:
            Pruner = get_pruner_class(pname)
            for kr in cfg.keep_ratios:
                if pname == "no_pruning" and kr < 1.0 - 1e-9:
                    continue
                try:
                    step_label = f"{pname} kr={kr}"
                    t_gen = time.monotonic()
                    if cfg.mock_model:
                        log_phase(f"EVA01 mock generate {step_label} glb={sample.glb_path}")
                        mesh_count, pruned_count = _mock_eva01_token_counts(pname, float(kr))
                        caption = _mock_eva01_caption(sample, pname, float(kr))
                        elapsed = time.monotonic() - t_gen
                        n_in = 37 + pruned_count
                        n_out = max(1, len(caption.split()))
                        meta = {
                            "method": f"eva01_mock_{pname}",
                            "backend": "eva01",
                            "mock_model": True,
                            "mesh_value_count": mesh_count,
                        }
                        pruner_tf = 0.0
                    else:
                        if feature_error is not None:
                            raise RuntimeError(f"EVA01 mesh feature extraction failed:\n{feature_error}")
                        if eva_features is None:
                            raise RuntimeError("EVA01 mesh features were not initialized")

                        extra = load_pruner_extra_kwargs(eval_cfg_dir, pname)
                        pruner = Pruner(keep_ratio=kr, seed=cfg.seed, **extra)
                        if pname in EVA01_VQVAE_SPATIAL_PRUNERS:
                            if vqvae is None or vq_emb is None:
                                raise RuntimeError(
                                    f"EVA01 spatial pruner {pname!r} requires VQVAE; "
                                    "set --vqvae-device and ensure VQVAE loads."
                                )
                            if token_ids is None or voxel_grid is None:
                                with phase_timer(f"EVA01 spatial VQVAE tokens glb={sample.glb_path}"):
                                    token_ids, voxel_grid = mesh_to_tokens(
                                        sample.glb_path,
                                        vqvae,
                                        vqvae_dev,
                                        file_identifier=sample.file_identifier,
                                        mesh_cache_dir=cfg.mesh_cache_dir,
                                        mesh_cache_readonly=cfg.mesh_cache_readonly,
                                        vlm_device=device,
                                    )
                            with phase_timer(f"EVA01 prune {step_label} (VQVAE spatial)"):
                                _pruned_vq_ids, meta = pruner.prune(
                                    token_ids,
                                    voxel_grid,
                                    vq_embeddings=vq_emb,
                                    _log_sample_idx=si,
                                    _log_tag=sample.file_identifier,
                                    _log_keep_ratio=float(kr),
                                )
                            meta = dict(meta)
                            enrich_pruner_metadata_flops(
                                pname,
                                meta,
                                embed_dim=vq_embed_dim,
                                codebook_size=vq_codebook_size,
                            )
                            vq_indices = meta.get("indices")
                            if vq_indices is None:
                                raise ValueError(f"Spatial pruner {pname!r} did not return meta['indices']")
                            patch_target = target_eva01_patch_count(float(kr), eva_features.patch_count)
                            patch_indices, mapping_diag = map_vq_indices_to_eva_patches(
                                vq_indices,
                                eva_features.patch_centers,
                                target_count=patch_target,
                                num_patches=eva_features.patch_count,
                            )
                            meta.update(
                                {
                                    "backend": "eva01",
                                    "eva01_adapter": "vqvae_spatial_to_patch_embeddings",
                                    "vq_indices": [int(x) for x in vq_indices],
                                    "eva_patch_indices": patch_indices,
                                    "num_eva_patches_original": eva_features.patch_count,
                                    "num_eva_patches_pruned": len(patch_indices),
                                    "vq_to_eva_mapping": mapping_diag,
                                }
                            )
                            embed_dim = vq_embed_dim
                            codebook_size = vq_codebook_size
                        else:
                            with phase_timer(f"EVA01 prune {step_label} (patch embeddings)"):
                                patch_indices, meta, local_embed = prune_eva01_patch_embeddings(
                                    pruner,
                                    eva_features.patch_tokens,
                                    keep_ratio=float(kr),
                                    sample_idx=si,
                                    tag=sample.file_identifier,
                                )
                            embed_dim = int(local_embed.weight.shape[1])
                            codebook_size = int(local_embed.weight.shape[0])
                            enrich_pruner_metadata_flops(
                                pname,
                                meta,
                                embed_dim=embed_dim,
                                codebook_size=codebook_size,
                            )

                        mesh_tokens = select_eva01_mesh_tokens(eva_features.mesh_tokens, patch_indices)
                        mesh_count = eva_features.mesh_token_count
                        pruned_count = int(mesh_tokens.shape[0])
                        t_gen = time.monotonic()
                        log_phase(f"EVA01 generate {step_label} ({pruned_count}/{mesh_count} mesh tokens)")
                        caption, elapsed, n_in, n_out = generate_eva01_caption_from_mesh_tokens(
                            model,
                            processor,
                            mesh_tokens,
                            cfg.caption_prompt,
                            max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            top_k=cfg.top_k,
                            device=device,
                        )
                        diag = meta.get("diagnostics") if isinstance(meta.get("diagnostics"), dict) else {}
                        pruner_tf = float(diag.get("pruner_tflops", 0.0))
                        meta.setdefault("mesh_value_count", eva_features.mesh_value_count)
                        meta.setdefault("embedding_dim", embed_dim)
                        meta.setdefault("codebook_size", codebook_size)

                    log_phase(
                        f"EVA01 {'mock ' if cfg.mock_model else ''}generate {step_label} done "
                        f"({time.monotonic() - t_gen:.1f}s, out_tokens={n_out})"
                    )
                    scores = compute_text_metrics(caption, sample.captions)
                    llm_tf = estimate_llm_tflops(n_in, n_out, cfg.eva01_model_id) if n_in else {
                        "llm_prefill_tflops": None,
                        "llm_decode_tflops": None,
                        "llm_total_tflops": None,
                    }
                    total_tflops = llm_tf["llm_total_tflops"]
                    results.append(
                        {
                            "model_backend": "eva01",
                            "file_identifier": sample.file_identifier,
                            "reference_captions": list(sample.captions),
                            "pruner": pname,
                            "keep_ratio": float(kr),
                            "num_tokens_original": mesh_count,
                            "num_tokens_pruned": pruned_count,
                            "generation_time_sec": float(elapsed),
                            "num_input_tokens": int(n_in),
                            "num_output_tokens": int(n_out),
                            "pruner_tflops": pruner_tf,
                            "generated_caption": caption,
                            "pruner_metadata": meta,
                            **llm_tf,
                            "total_tflops": float(pruner_tf + total_tflops) if total_tflops is not None else None,
                            **scores,
                        }
                    )
                except Exception:
                    err = traceback.format_exc()
                    print(err, file=sys.stderr)
                    results.append(
                        {
                            "model_backend": "eva01",
                            "file_identifier": sample.file_identifier,
                            "reference_captions": list(sample.captions),
                            "pruner": pname,
                            "keep_ratio": float(kr),
                            "error": "eval_failed",
                            "traceback": err,
                        }
                    )
    return results


def main(argv: list[str] | None = None) -> int:
    cfg = EvalConfig.from_args(argv)
    cfg = resolve_repo_paths(cfg, REPO_ROOT)
    _apply_config_env(cfg)
    _setup_run_log(cfg.run_log_file)
    eval_cfg_dir = Path(cfg.eval_config_dir)

    for name in cfg.pruners:
        get_pruner_class(name)  # fail fast if unknown

    if cfg.model_backend == "eva01":
        try:
            validate_eva01_eval_config(cfg)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    eva_needs_vqvae = (
        cfg.model_backend == "eva01"
        and not cfg.mock_model
        and any(p in EVA01_VQVAE_SPATIAL_PRUNERS for p in cfg.pruners)
    )
    needs_cuda = cfg.device.startswith("cuda") or (
        (cfg.model_backend == "shapellm" or eva_needs_vqvae) and cfg.vqvae_device.startswith("cuda")
    )
    if not torch.cuda.is_available() and needs_cuda:
        print("Warning: CUDA not available; falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
        vqvae_dev = torch.device("cpu")
    else:
        device = resolve_torch_device(cfg.device)
        vqvae_dev = (
            resolve_torch_device(cfg.vqvae_device)
            if cfg.model_backend == "shapellm" or eva_needs_vqvae
            else torch.device("cpu")
        )

    if cfg.model_backend == "shapellm" or eva_needs_vqvae:
        init_cuda_for_eval(vqvae_dev, device)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    vqvae = None
    model = None
    processor = None
    tokenizer = None
    try:
        if cfg.model_backend == "eva01" and cfg.mock_model:
            print("Using EVA01 mock model: no OpenEVA import, no checkpoint download, no GPU model load.")
        elif cfg.model_backend == "eva01":
            from eval.eva01_backend import load_eva01_model

            print(f"Loading EVA01 ({cfg.eva01_model_id}) on {device}...")
            model, processor = load_eva01_model(
                cfg.eva01_model_id,
                device=device,
                torch_dtype=cfg.vlm_torch_dtype,
                base_model_name_or_path=cfg.eva01_base_model_name_or_path,
            )
            if eva_needs_vqvae:
                print(f"Loading VQVAE for EVA01 spatial pruners (device={vqvae_dev})...")
                vqvae = load_vqvae(vqvae_dev)
                warmup_vqvae(vqvae, vqvae_dev, vlm_dev=device)
        else:
            print(f"Loading VQVAE (device={vqvae_dev})...")
            vqvae = load_vqvae(vqvae_dev)
            warmup_vqvae(vqvae, vqvae_dev, vlm_dev=device)
            if cfg.load_in_4bit:
                mode = "4-bit NF4"
            else:
                mode = (
                    f"16-bit ({cfg.vlm_torch_dtype})"
                    if cfg.vlm_torch_dtype != "auto"
                    else "bf16/fp16 (auto)"
                )
            print(f"Loading VLM ({mode}) on {device}...")
            model, processor, tokenizer = load_llm(
                cfg.model_id,
                device,
                load_in_4bit=cfg.load_in_4bit,
                vlm_torch_dtype=cfg.vlm_torch_dtype,
            )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except torch.cuda.OutOfMemoryError:
        print(
            "CUDA 显存不足，已中止。可换更大显存、加 --load-in-4bit、--vqvae-device cpu，或释放显存后再试。",
            file=sys.stderr,
        )
        return 1

    samples = list(
        iter_dataset(
            cfg.data_csv,
            cfg.glb_dir,
            num_samples=cfg.num_samples,
            skip_missing_glb=True,
        )
    )
    if not samples:
        print(
            f"No samples found (csv={cfg.data_csv}, glb_dir={cfg.glb_dir}). "
            "Place GLB files under glb_dir or fix paths.",
            file=sys.stderr,
        )
        return 1

    print(f"Evaluating {len(samples)} mesh samples.")
    ensure_dir(cfg.output_dir)

    if cfg.model_backend == "eva01":
        results = _run_eva01_eval(
            cfg,
            samples,
            model,
            processor,
            device,
            vqvae=vqvae,
            vqvae_dev=vqvae_dev,
            eval_cfg_dir=eval_cfg_dir,
        )
        return _finalize_results(cfg, results)

    results: list[dict] = []
    # Fast-fail: abort early when a broken pruner makes every prune/eval raise.
    # A bad LLM diff (e.g. NameError) otherwise wastes a full grid of GPU time.
    eval_attempts = 0
    eval_failures = 0
    fastfail_min_attempts = len(cfg.pruners) * len(cfg.keep_ratios)
    aborted_all_failing = False
    os.environ.setdefault("SHAPELLM_EVAL_LOG_DIR", os.path.join(cfg.output_dir, "logs"))
    os.environ.setdefault("SHAPELLM_EVAL_LOG_DEEP_EVERY", "20")
    os.makedirs(os.environ["SHAPELLM_EVAL_LOG_DIR"], exist_ok=True)
    print(
        f"Pruner diagnostics -> SHAPELLM_EVAL_LOG_DIR={os.environ['SHAPELLM_EVAL_LOG_DIR']} "
        f"(deep_every={os.environ['SHAPELLM_EVAL_LOG_DEEP_EVERY']}); "
        f"proposed pruners among {cfg.pruners} will emit <method>.log / <method>.deep.jsonl"
    )

    vq_emb = getattr(vqvae.vq, "embeddings", None)
    embed_dim = int(vq_emb.weight.shape[1]) if vq_emb is not None else 32
    codebook_size = int(vq_emb.weight.shape[0]) if vq_emb is not None else 8192

    token_dump_root = os.environ.get("SHAPELLM_EVAL_TOKEN_HEAD_DUMP", "").strip()
    n_ratio = len(cfg.keep_ratios)
    steps_per_sample = len(cfg.pruners) * n_ratio
    log_phase(
        f"sample loop: {len(samples)} meshes x {steps_per_sample} "
        f"(pruners={cfg.pruners}, keep_ratios={cfg.keep_ratios}); "
        f"VQVAE={vqvae_dev}, VLM={device}"
    )
    if vqvae_dev.type == "cpu":
        log_phase(
            "VQVAE on CPU: mesh_to_tokens is slow and GPU stays idle until VLM generate; "
            "see eval_budget.yaml vqvae_device (cuda:1 for dual GPU)"
        )
    if cfg.mesh_cache_dir:
        log_phase(f"mesh voxel cache dir={cfg.mesh_cache_dir} readonly={cfg.mesh_cache_readonly}")

    from concurrent.futures import Future, ThreadPoolExecutor

    prefetch_executor: ThreadPoolExecutor | None = None
    prefetch_futures: dict[int, Future] = {}
    if cfg.mesh_prefetch_workers > 0 and len(samples) > 1:

        def _submit_prefetch(idx: int) -> None:
            if idx >= len(samples) or prefetch_executor is None:
                return
            s = samples[idx]
            prefetch_futures[idx] = prefetch_executor.submit(
                prepare_mesh_coords,
                s.glb_path,
                s.file_identifier,
                cfg.mesh_cache_dir,
                mesh_cache_readonly=cfg.mesh_cache_readonly,
            )

        prefetch_executor = ThreadPoolExecutor(max_workers=cfg.mesh_prefetch_workers)
        _submit_prefetch(0)
        log_phase(f"mesh coord prefetch workers={cfg.mesh_prefetch_workers}")

    sample_iter = enumerate(samples)
    if progress_enabled() and len(samples) > 1:
        from tqdm import tqdm

        sample_iter = enumerate(tqdm(samples, desc="eval meshes", unit="mesh"))

    skipped_samples = 0

    def _record_skipped_sample(sample, err: str, error_code: str) -> None:
        nonlocal skipped_samples
        skipped_samples += 1
        print(err, file=sys.stderr)
        print(
            f"[eval] skip sample {sample.file_identifier}: {error_code}",
            flush=True,
        )
        results.append(
            {
                "model_backend": cfg.model_backend,
                "file_identifier": sample.file_identifier,
                "reference_captions": list(sample.captions),
                "error": error_code,
                "traceback": err,
            }
        )

    try:
        for si, sample in sample_iter:
            try:
                print(f"[{si+1}/{len(samples)}] {sample.file_identifier}", flush=True)
                tok_dump_path = None
                if token_dump_root:
                    os.makedirs(token_dump_root, exist_ok=True)
                    safe_tag = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in sample.file_identifier
                    )[:80]
                    tok_dump_path = os.path.join(token_dump_root, f"{si:05d}_{safe_tag}.json")

                prefetched_coords = None
                if si in prefetch_futures:
                    try:
                        prefetched_coords = prefetch_futures.pop(si).result()
                    except Exception:
                        _record_skipped_sample(
                            sample, traceback.format_exc(), "mesh_prefetch_failed"
                        )
                        if prefetch_executor is not None:
                            _submit_prefetch(si + 1)
                        continue
                if prefetch_executor is not None:
                    _submit_prefetch(si + 1)

                try:
                    token_ids, voxel_grid = mesh_to_tokens(
                        sample.glb_path,
                        vqvae,
                        vqvae_dev,
                        token_head_dump_path=tok_dump_path,
                        file_identifier=sample.file_identifier,
                        mesh_cache_dir=cfg.mesh_cache_dir,
                        mesh_cache_readonly=cfg.mesh_cache_readonly,
                        prefetched_coords=prefetched_coords,
                        vlm_device=device,
                    )
                except Exception:
                    _record_skipped_sample(
                        sample, traceback.format_exc(), "mesh_to_tokens_failed"
                    )
                    if vqvae_dev.type == "cuda":
                        release_cuda_device_memory(vqvae_dev)
                    continue

                n_orig = int(token_ids.numel())

                for pname in cfg.pruners:
                    Pruner = get_pruner_class(pname)
                    for kr in cfg.keep_ratios:
                        if pname == "no_pruning" and kr < 1.0 - 1e-9:
                            continue
                        eval_attempts += 1
                        try:
                            extra = load_pruner_extra_kwargs(eval_cfg_dir, pname)
                            pruner = Pruner(keep_ratio=kr, seed=cfg.seed, **extra)
                            step_label = f"{pname} kr={kr}"
                            with phase_timer(f"prune {step_label}"):
                                pruned_ids, meta = pruner.prune(
                                    token_ids,
                                    voxel_grid,
                                    vq_embeddings=vq_emb,
                                    _log_sample_idx=si,
                                    _log_tag=sample.file_identifier,
                                    _log_keep_ratio=float(kr),
                                )
                            enrich_pruner_metadata_flops(
                                pname,
                                meta,
                                embed_dim=embed_dim,
                                codebook_size=codebook_size,
                            )
                            mesh_str = tokens_to_mesh_string(pruned_ids)
                            t_gen = time.monotonic()
                            log_phase(f"VLM generate {step_label} ({len(pruned_ids)} tokens)")
                            caption, elapsed, n_in, n_out = generate_caption(
                                model,
                                processor,
                                tokenizer,
                                mesh_str,
                                cfg.caption_prompt,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                                top_p=cfg.top_p,
                                top_k=cfg.top_k,
                                device=device,
                            )
                            log_phase(
                                f"VLM generate {step_label} done ({time.monotonic() - t_gen:.1f}s, "
                                f"out_tokens={n_out})"
                            )
                            scores = compute_text_metrics(caption, sample.captions)
                            llm_tf = estimate_llm_tflops(n_in, n_out, cfg.model_id)
                            diag = meta.get("diagnostics") if isinstance(meta.get("diagnostics"), dict) else {}
                            pruner_tf = float(diag.get("pruner_tflops", 0.0))
                            row = {
                                "model_backend": cfg.model_backend,
                                "file_identifier": sample.file_identifier,
                                "reference_captions": list(sample.captions),
                                "pruner": pname,
                                "keep_ratio": float(kr),
                                "num_tokens_original": n_orig,
                                "num_tokens_pruned": int(pruned_ids.numel()),
                                "generation_time_sec": float(elapsed),
                                "num_input_tokens": int(n_in),
                                "num_output_tokens": int(n_out),
                                "pruner_tflops": pruner_tf,
                                "generated_caption": caption,
                                "pruner_metadata": meta,
                                **llm_tf,
                                "total_tflops": float(pruner_tf + llm_tf["llm_total_tflops"]),
                                **scores,
                            }
                            results.append(row)
                        except Exception:
                            err = traceback.format_exc()
                            print(err, file=sys.stderr)
                            eval_failures += 1
                            results.append(
                                {
                                    "model_backend": cfg.model_backend,
                                    "file_identifier": sample.file_identifier,
                                    "reference_captions": list(sample.captions),
                                    "pruner": pname,
                                    "keep_ratio": float(kr),
                                    "error": "eval_failed",
                                    "traceback": err,
                                }
                            )
                            # If every attempt so far failed and we've completed at
                            # least one full pruner x keep_ratio sweep, the pruner is
                            # broken (e.g. NameError from a bad diff). Stop now instead
                            # of burning the whole grid on a guaranteed-empty result.
                            if (
                                eval_failures == eval_attempts
                                and eval_attempts >= fastfail_min_attempts
                            ):
                                aborted_all_failing = True
                                print(
                                    f"eval_aborted_all_failing=1 attempts={eval_attempts} "
                                    f"failures={eval_failures}",
                                    flush=True,
                                )
                                break
                    if aborted_all_failing:
                        break
                if aborted_all_failing:
                    break
                if vqvae_dev.type == "cuda":
                    release_cuda_device_memory(vqvae_dev)
            except Exception:
                _record_skipped_sample(
                    sample, traceback.format_exc(), "sample_failed"
                )
                if vqvae_dev.type == "cuda":
                    release_cuda_device_memory(vqvae_dev)
                if prefetch_executor is not None:
                    _submit_prefetch(si + 1)
                continue
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=False)

    return _finalize_results(cfg, results, skipped_samples=skipped_samples)


if __name__ == "__main__":
    raise SystemExit(main())
