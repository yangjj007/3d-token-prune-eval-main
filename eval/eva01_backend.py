"""EVA01 mesh-understanding backend for caption evaluation.

This module intentionally imports OpenEVA lazily so the default ShapeLLM
evaluation path keeps working when the optional EVA01 runtime is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn


DEFAULT_EVA01_MODEL_ID = "SEELE-AI/EVA01-2B-Instruct-LoRA"
EVA01_ORIGINAL_MESH_TOKEN_COUNT = 513
EVA01_ORIGINAL_PATCH_TOKEN_COUNT = 512
EVA01_VQVAE_SPATIAL_PRUNERS = frozenset(
    {
        "loco3d",
        "loco3d_dpp",
        "loco3d_nonempty_dpp",
        "octree_merge",
        "runlength_curve",
        "reconot",
    }
)


@dataclass
class EVA01MeshFeatures:
    """Raw EVA01 mesh-understanding tokens before the connector projection."""

    mesh_tokens: torch.Tensor  # [1 + P, D], CPU
    patch_centers: torch.Tensor | None  # [P, 3], CPU
    mesh_value_count: int | None

    @property
    def mesh_token_count(self) -> int:
        return int(self.mesh_tokens.shape[0])

    @property
    def patch_count(self) -> int:
        return max(0, int(self.mesh_tokens.shape[0]) - 1)

    @property
    def patch_tokens(self) -> torch.Tensor:
        return self.mesh_tokens[1:]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unsupported EVA01 dtype: {dtype_name}")


def _model_device(model: Any, fallback: torch.device) -> torch.device:
    dev = getattr(model, "device", None)
    if dev is not None:
        return torch.device(dev)
    try:
        return next(model.parameters()).device
    except Exception:
        return fallback


def _to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if hasattr(v, "to") else v
        return out
    return batch


def _get_input_ids(inputs: Any) -> torch.Tensor | None:
    if isinstance(inputs, dict):
        return inputs.get("input_ids")
    return getattr(inputs, "input_ids", None)


def _get_attention_mask(inputs: Any) -> torch.Tensor | None:
    if isinstance(inputs, dict):
        return inputs.get("attention_mask")
    return getattr(inputs, "attention_mask", None)


def _get_mesh_values(inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return inputs.get("mesh_und_values")
    return getattr(inputs, "mesh_und_values", None)


def _mesh_value_count(mesh_values: Any) -> int | None:
    if mesh_values is None:
        return None
    if isinstance(mesh_values, torch.Tensor):
        if mesh_values.dim() >= 2:
            return int(mesh_values.shape[-2])
        return int(mesh_values.numel())
    if isinstance(mesh_values, (list, tuple)):
        if not mesh_values:
            return 0
        first = mesh_values[0]
        if isinstance(first, torch.Tensor):
            if first.dim() >= 2:
                return int(sum(x.shape[-2] for x in mesh_values if isinstance(x, torch.Tensor)))
            return int(sum(x.numel() for x in mesh_values if isinstance(x, torch.Tensor)))
        return len(mesh_values)
    return None


def load_eva01_model(
    model_id: str = DEFAULT_EVA01_MODEL_ID,
    *,
    device: torch.device,
    torch_dtype: str = "auto",
    base_model_name_or_path: str = "",
) -> Tuple[Any, Any]:
    """Load EVA01 model and processor, raising a clear optional-dependency error."""
    try:
        from eva01 import EVA01ForConditionalGeneration, EVA01Processor
    except Exception as exc:
        raise ImportError(
            "EVA01 backend requires the OpenEVA EVA01 runtime. Install it from "
            "https://github.com/SeeleAI/OpenEVA and make the `eva01` package importable."
        ) from exc

    kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(torch_dtype),
        "trust_remote_code": True,
    }
    if base_model_name_or_path:
        kwargs["base_model_name_or_path"] = base_model_name_or_path
    if device.type == "cuda":
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    model = EVA01ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if device.type != "cuda" and hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    processor = EVA01Processor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def target_eva01_patch_count(keep_ratio: float, num_patches: int = EVA01_ORIGINAL_PATCH_TOKEN_COUNT) -> int:
    k = max(1, int(round(float(keep_ratio) * int(num_patches))))
    return min(k, int(num_patches))


def build_eva01_mesh_token_text(processor: Any, mesh_token_count: int) -> str:
    mesh_token = str(getattr(processor, "mesh_und_token", "<|mesh_und_pad|>"))
    return " ".join([mesh_token] * int(mesh_token_count))


def build_eva01_prompt_inputs(
    processor: Any,
    caption_prompt: str,
    mesh_token_count: int,
    *,
    device: torch.device | str | None = None,
) -> Any:
    """Tokenize an EVA01 prompt with a variable number of mesh placeholders."""

    token_text = build_eva01_mesh_token_text(processor, mesh_token_count)
    content = "\n".join(part for part in (token_text, caption_prompt) if str(part).strip())
    rendered = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = processor.tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if device is not None:
        encoded = _to_device(encoded, torch.device(device))
    return encoded


def _mesh_token_id(model: Any, processor: Any) -> int:
    value = getattr(processor, "mesh_und_token_id", None)
    if value is not None:
        return int(value)
    config = getattr(model, "config", None)
    value = getattr(config, "mesh_und_token_id", None)
    if value is not None:
        return int(value)
    token = str(getattr(processor, "mesh_und_token", "<|mesh_und_pad|>"))
    return int(processor.tokenizer.convert_tokens_to_ids(token))


def _connector_dtype(model: Any, fallback: torch.dtype = torch.float32) -> torch.dtype:
    value = getattr(model, "_output_dtype", None)
    if isinstance(value, torch.dtype):
        return value
    connector = getattr(model, "mesh_und_connector", None)
    if connector is not None:
        try:
            return next(connector.parameters()).dtype
        except StopIteration:
            pass
        except Exception:
            pass
    return fallback


def extract_eva01_mesh_features(
    model: Any,
    processor: Any,
    mesh_path: str,
    caption_prompt: str,
    *,
    device: str | torch.device = "cuda",
) -> EVA01MeshFeatures:
    """Load a mesh through EVA01's processor and return raw mesh encoder tokens."""

    dev = torch.device(device)
    model_dev = _model_device(model, dev)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "mesh", "mesh": mesh_path},
                {"type": "text", "text": caption_prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    mesh_values = _get_mesh_values(inputs)
    if mesh_values is None:
        raise ValueError("EVA01 processor did not return mesh_und_values")
    if not isinstance(mesh_values, torch.Tensor):
        mesh_values = torch.as_tensor(mesh_values)
    mesh_count = _mesh_value_count(mesh_values)

    encoder = getattr(model, "mesh_und_encoder", None)
    if encoder is None:
        raise AttributeError("EVA01 model is missing mesh_und_encoder")

    values = mesh_values.to(device=model_dev, dtype=torch.float32)
    with torch.inference_mode():
        if hasattr(encoder, "forward_with_centers"):
            tokens, centers = encoder.forward_with_centers(values)
        else:
            tokens = encoder(values)
            centers = None

    if tokens.dim() != 3 or tokens.shape[0] != 1:
        raise ValueError(f"Expected EVA01 mesh tokens [1,T,D], got {tuple(tokens.shape)}")
    tokens_cpu = tokens[0].detach().cpu().float()
    centers_cpu = None
    if centers is not None:
        if centers.dim() != 3 or centers.shape[0] != 1:
            raise ValueError(f"Expected EVA01 patch centers [1,P,3], got {tuple(centers.shape)}")
        centers_cpu = centers[0].detach().cpu().float()
    return EVA01MeshFeatures(
        mesh_tokens=tokens_cpu,
        patch_centers=centers_cpu,
        mesh_value_count=mesh_count,
    )


def make_embedding_from_features(features: torch.Tensor) -> nn.Embedding:
    weight = features.detach().float().cpu()
    if weight.dim() != 2:
        raise ValueError(f"Expected feature matrix [N,D], got {tuple(weight.shape)}")
    return nn.Embedding.from_pretrained(weight, freeze=True)


def normalize_patch_indices(
    indices: Sequence[int] | torch.Tensor,
    target_count: int,
    num_patches: int,
) -> tuple[list[int], dict[str, Any]]:
    """Return sorted unique patch indices, filled/truncated deterministically."""

    if isinstance(indices, torch.Tensor):
        raw = [int(x) for x in indices.detach().cpu().long().view(-1).tolist()]
    else:
        raw = [int(x) for x in indices]
    valid = [x for x in raw if 0 <= x < int(num_patches)]
    target = max(1, min(int(target_count), int(num_patches)))
    unique_sorted = sorted(set(valid))
    truncated = max(0, len(unique_sorted) - target)
    if len(unique_sorted) > target:
        out = unique_sorted[:target]
    else:
        out = list(unique_sorted)
        if len(out) < target:
            selected = set(out)
            for idx in range(int(num_patches)):
                if idx in selected:
                    continue
                out.append(idx)
                selected.add(idx)
                if len(out) >= target:
                    break
        out.sort()
    diag = {
        "raw_count": len(raw),
        "valid_count": len(valid),
        "unique_count": len(unique_sorted),
        "invalid_count": len(raw) - len(valid),
        "duplicate_count": len(valid) - len(set(valid)),
        "target_count": target,
        "filled_count": max(0, target - len(unique_sorted)),
        "truncated_count": truncated,
    }
    return out, diag


def prune_eva01_patch_embeddings(
    pruner: Any,
    patch_tokens: torch.Tensor,
    *,
    keep_ratio: float,
    sample_idx: int | None = None,
    tag: str = "",
) -> tuple[list[int], dict[str, Any], nn.Embedding]:
    """Run a token pruner directly on EVA01 patch embeddings."""

    patch_tokens_cpu = patch_tokens.detach().float().cpu()
    num_patches = int(patch_tokens_cpu.shape[0])
    if num_patches <= 0:
        raise ValueError("EVA01 mesh features contain no patch tokens")
    token_ids = torch.arange(num_patches, dtype=torch.long)
    local_embed = make_embedding_from_features(patch_tokens_cpu)
    pruned_ids, meta = pruner.prune(
        token_ids,
        None,
        vq_embeddings=local_embed,
        _log_sample_idx=sample_idx,
        _log_tag=tag,
        _log_keep_ratio=float(keep_ratio),
    )
    meta = dict(meta)
    method = str(meta.get("method", ""))
    target_count = num_patches if method == "no_pruning" else target_eva01_patch_count(keep_ratio, num_patches)
    patch_indices, norm_diag = normalize_patch_indices(pruned_ids, target_count, num_patches)
    meta.update(
        {
            "backend": "eva01",
            "eva01_adapter": "patch_embeddings",
            "eva_patch_indices": patch_indices,
            "num_eva_patches_original": num_patches,
            "num_eva_patches_pruned": len(patch_indices),
            "patch_index_normalization": norm_diag,
        }
    )
    return patch_indices, meta, local_embed


def select_eva01_mesh_tokens(mesh_tokens: torch.Tensor, patch_indices: Sequence[int]) -> torch.Tensor:
    """Keep cls token plus selected EVA01 patch tokens in original patch order."""

    raw = mesh_tokens.detach().float().cpu()
    if raw.dim() != 2 or raw.shape[0] < 1:
        raise ValueError(f"Expected EVA01 mesh token matrix [T,D], got {tuple(raw.shape)}")
    patches = raw[1:]
    selected, _diag = normalize_patch_indices(patch_indices, len(patch_indices), int(patches.shape[0]))
    idx = torch.tensor(selected, dtype=torch.long)
    return torch.cat([raw[:1], patches.index_select(0, idx)], dim=0)


def build_eva01_inputs_embeds_from_mesh_tokens(
    model: Any,
    processor: Any,
    input_ids: torch.Tensor,
    mesh_tokens: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Project raw EVA01 mesh tokens and splice them into token embeddings."""

    dev = torch.device(device)
    model_dev = _model_device(model, dev)
    input_ids = input_ids.to(model_dev)
    input_embeds = model.get_input_embeddings()(input_ids)
    dtype = _connector_dtype(model, fallback=input_embeds.dtype)
    raw = mesh_tokens.detach().unsqueeze(0).to(device=model_dev, dtype=dtype)
    projected = model.mesh_und_connector(raw).to(dtype=input_embeds.dtype)
    mesh_mask = input_ids.eq(_mesh_token_id(model, processor))
    expected_tokens = int(projected.shape[0] * projected.shape[1])
    actual_tokens = int(mesh_mask.sum().item())
    if actual_tokens != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} EVA01 mesh placeholders, found {actual_tokens}.")
    out = input_embeds.clone()
    out[mesh_mask] = projected.reshape(-1, projected.shape[-1])
    return out


def _batch_decode(processor: Any, token_ids: torch.Tensor) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    return processor.tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_eva01_caption_from_mesh_tokens(
    model: Any,
    processor: Any,
    mesh_tokens: torch.Tensor,
    caption_prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.7,
    top_k: int = 8192,
    device: str | torch.device = "cuda",
) -> Tuple[str, float, int, int]:
    """Generate from pre-pruned raw EVA01 mesh tokens."""

    dev = torch.device(device)
    model_dev = _model_device(model, dev)
    inputs = build_eva01_prompt_inputs(
        processor,
        caption_prompt,
        int(mesh_tokens.shape[0]),
        device=model_dev,
    )
    input_ids = _get_input_ids(inputs)
    if input_ids is None:
        raise ValueError("EVA01 tokenizer did not return input_ids")
    attention_mask = _get_attention_mask(inputs)
    inputs_embeds = build_eva01_inputs_embeds_from_mesh_tokens(
        model,
        processor,
        input_ids,
        mesh_tokens,
        device=model_dev,
    )
    num_input_tokens = int(input_ids.shape[1])

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids.to(model_dev),
        "inputs_embeds": inputs_embeds,
        "max_new_tokens": max_new_tokens,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask.to(model_dev)
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    generator = getattr(getattr(model, "qwen3vl", None), "generate", None)
    if generator is None:
        generator = model.generate

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = generator(**gen_kwargs)
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids[0, input_ids.shape[1] :]
    text = _batch_decode(processor, new_tokens.unsqueeze(0))
    return text.strip(), elapsed, num_input_tokens, int(new_tokens.numel())


def vqvae_latent_centers(device: torch.device | str | None = None) -> torch.Tensor:
    """Centers for the 8x8x16 VQVAE latent grid, normalized to roughly [-1, 1]."""

    dev = torch.device(device) if device is not None else torch.device("cpu")
    gx, gy, gz = 8, 8, 16
    idx = torch.arange(gx * gy * gz, device=dev, dtype=torch.float32)
    x = torch.floor(idx / (gy * gz))
    rem = idx - x * (gy * gz)
    y = torch.floor(rem / gz)
    z = rem - y * gz
    return torch.stack(
        [
            ((x + 0.5) / gx) * 2.0 - 1.0,
            ((y + 0.5) / gy) * 2.0 - 1.0,
            ((z + 0.5) / gz) * 2.0 - 1.0,
        ],
        dim=-1,
    )


def map_vq_indices_to_eva_patches(
    vq_indices: Sequence[int] | torch.Tensor,
    eva_patch_centers: torch.Tensor | None,
    *,
    target_count: int,
    num_patches: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Map selected 1024-layout VQVAE cells onto EVA01 patch-token positions."""

    if isinstance(vq_indices, torch.Tensor):
        raw = [int(x) for x in vq_indices.detach().cpu().long().view(-1).tolist()]
    else:
        raw = [int(x) for x in vq_indices]
    valid_vq = [x for x in raw if 0 <= x < 1024]

    if eva_patch_centers is None:
        if num_patches is None:
            num_patches = EVA01_ORIGINAL_PATCH_TOKEN_COUNT
        projected = [x % int(num_patches) for x in valid_vq]
        selected, diag = normalize_patch_indices(projected, target_count, int(num_patches))
        diag.update({"mapping_mode": "mod_fallback", "vq_indices_count": len(raw)})
        return selected, diag

    centers = eva_patch_centers.detach().float().cpu()
    if centers.dim() != 2 or centers.shape[1] != 3:
        raise ValueError(f"Expected EVA01 patch centers [P,3], got {tuple(centers.shape)}")
    num = int(centers.shape[0])
    target = max(1, min(int(target_count), num))
    if not valid_vq:
        selected, diag = normalize_patch_indices([], target, num)
        diag.update({"mapping_mode": "nearest_center", "vq_indices_count": len(raw)})
        return selected, diag

    vq_centers = vqvae_latent_centers().index_select(0, torch.tensor(valid_vq, dtype=torch.long))
    dists = torch.cdist(vq_centers, centers, p=2.0)
    nearest = torch.argmin(dists, dim=1)
    patch_to_selected_d = torch.cdist(centers, vq_centers, p=2.0).min(dim=1).values
    counts = torch.bincount(nearest, minlength=num)
    mapped = [int(x) for x in nearest.tolist()]
    unique = sorted(set(mapped))

    if len(unique) > target:
        ranked = sorted(unique, key=lambda i: (-int(counts[i].item()), float(patch_to_selected_d[i].item()), i))
        out = sorted(ranked[:target])
        truncated = len(unique) - target
        filled = 0
    else:
        out = list(unique)
        filled = 0
        if len(out) < target:
            selected_set = set(out)
            fill_order = sorted(
                [i for i in range(num) if i not in selected_set],
                key=lambda i: (float(patch_to_selected_d[i].item()), i),
            )
            for idx in fill_order:
                out.append(idx)
                selected_set.add(idx)
                filled += 1
                if len(out) >= target:
                    break
        out.sort()
        truncated = 0

    diag = {
        "mapping_mode": "nearest_center",
        "vq_indices_count": len(raw),
        "valid_vq_indices_count": len(valid_vq),
        "raw_mapped_count": len(mapped),
        "unique_mapped_count": len(unique),
        "duplicate_mapped_count": len(mapped) - len(unique),
        "target_count": target,
        "filled_count": filled,
        "truncated_count": truncated,
        "mean_selected_distance": float(patch_to_selected_d[out].mean().item()) if out else None,
        "max_selected_distance": float(patch_to_selected_d[out].max().item()) if out else None,
    }
    return out, diag


def generate_eva01_caption(
    model: Any,
    processor: Any,
    mesh_path: str,
    caption_prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.7,
    top_k: int = 8192,
    device: str | torch.device = "cuda",
) -> Tuple[str, float, int, int, int | None]:
    """
    Generate one caption from a ``.glb`` mesh path.

    Returns:
        caption, elapsed_sec, num_input_tokens, num_output_tokens, mesh_value_count
    """
    dev = torch.device(device)
    model_dev = _model_device(model, dev)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "mesh", "mesh": mesh_path},
                {"type": "text", "text": caption_prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _to_device(inputs, model_dev)
    input_ids = _get_input_ids(inputs)
    num_input_tokens = int(input_ids.shape[1]) if input_ids is not None else 0
    mesh_count = _mesh_value_count(_get_mesh_values(inputs))

    gen_kwargs = dict(inputs)
    gen_kwargs["max_new_tokens"] = max_new_tokens
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    elapsed = time.perf_counter() - t0

    if input_ids is not None:
        new_tokens = output_ids[0, input_ids.shape[1] :]
    else:
        new_tokens = output_ids[0]
    text = processor.batch_decode(
        new_tokens.unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip(), elapsed, num_input_tokens, int(new_tokens.numel()), mesh_count
