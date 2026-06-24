"""EVA01 mesh-understanding backend for caption evaluation.

This module intentionally imports OpenEVA lazily so the default ShapeLLM
evaluation path keeps working when the optional EVA01 runtime is not installed.
"""

from __future__ import annotations

import time
from typing import Any, Tuple

import torch


DEFAULT_EVA01_MODEL_ID = "SEELE-AI/EVA01-2B-Instruct-LoRA"


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
