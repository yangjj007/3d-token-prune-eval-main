"""Default HF env for server containers (only applies if not already set)."""

from __future__ import annotations

import os


def apply_hf_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/yangjunjie/huggingface")
