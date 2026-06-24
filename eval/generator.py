"""Qwen2.5-VL caption generation with timing (no Gradio)."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.generation.logits_process import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
)


def _transform_messages(original_messages: List[Dict]) -> List[Dict]:
    """Same structure as ``app.py`` ``_transform_messages``."""
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message["content"]:
            if "image" in item:
                new_item = {"type": "image", "image": item["image"]}
            elif "text" in item:
                new_item = {"type": "text", "text": item["text"]}
            elif "video" in item:
                new_item = {"type": "video", "video": item["video"]}
            else:
                continue
            new_content.append(new_item)
        new_message = {"role": message["role"], "content": new_content}
        transformed_messages.append(new_message)
    return transformed_messages


def build_user_text(mesh_token_str: str, caption_prompt: str) -> str:
    return f"{mesh_token_str}\n{caption_prompt}"


def generate_caption(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    tokenizer: PreTrainedTokenizerBase,
    mesh_token_str: str,
    caption_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 8192,
    device: str | torch.device = "cuda",
) -> Tuple[str, float, int, int]:
    """
    Returns:
        caption_text: Decoded assistant response (no prompt).
        elapsed_sec: Wall time for ``model.generate`` only.
        num_input_tokens: Length of input_ids passed to the model.
        num_output_tokens: Number of generated tokens (excluding prompt).
    """
    user_text = build_user_text(mesh_token_str, caption_prompt)
    messages = _transform_messages([{"role": "user", "content": [{"text": user_text}]}])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    num_input_tokens = int(inputs["input_ids"].shape[1])

    eos_token_id = [tokenizer.eos_token_id, 159858]
    gen_kwargs = dict(inputs)
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["eos_token_id"] = eos_token_id
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
        # FP16 VLM 采样时 logits 容易出现 inf/nan，softmax 后 multinomial 会抛
        # "probability tensor contains either `inf`, `nan` or element < 0"。
        # 这里先用 InfNanRemoveLogitsProcessor 把 nan 置 0、inf 夹到 dtype 最大值，
        # 再配合 renormalize_logits=True 让处理后的分布重新归一化一次。
        gen_kwargs["logits_processor"] = LogitsProcessorList(
            [InfNanRemoveLogitsProcessor()]
        )
        gen_kwargs["renormalize_logits"] = True
    else:
        gen_kwargs["do_sample"] = False

    def _run_generate(kwargs):
        with torch.inference_mode():
            return model.generate(**kwargs)

    t0 = time.perf_counter()
    try:
        out_ids = _run_generate(gen_kwargs)
    except RuntimeError as e:
        msg = str(e)
        if "probability tensor contains" not in msg and "inf" not in msg and "nan" not in msg:
            raise
        # 兜底：FP16 下即使加了 InfNanRemoveLogitsProcessor 也可能偶发数值不稳，
        # 回退到贪心解码重试一次，保证评测流水线不会整条中断。
        print(
            "[generator] sampling produced inf/nan probs, falling back to greedy decoding.",
            flush=True,
        )
        for k in ("temperature", "top_p", "top_k", "logits_processor", "renormalize_logits"):
            gen_kwargs.pop(k, None)
        gen_kwargs["do_sample"] = False
        out_ids = _run_generate(gen_kwargs)
    elapsed = time.perf_counter() - t0

    # Strip prompt tokens
    in_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[0, in_len:]
    caption = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_output_tokens = int(new_tokens.numel())
    return caption.strip(), elapsed, num_input_tokens, num_output_tokens
