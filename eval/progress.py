"""Phase logging for long eval runs (stdout, flushed)."""

from __future__ import annotations

import os
import time


def progress_enabled() -> bool:
    return os.environ.get("SHAPELLM_EVAL_PROGRESS", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def log_phase(msg: str, *, prefix: str = "eval") -> None:
    if not progress_enabled():
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{prefix}] {ts} {msg}", flush=True)


def phase_timer(label: str, *, prefix: str = "eval"):
    """Context manager: log start and done with elapsed seconds."""

    class _Timer:
        def __enter__(self):
            self._t0 = time.monotonic()
            log_phase(f"{label} ...", prefix=prefix)
            return self

        def __exit__(self, *exc):
            elapsed = time.monotonic() - self._t0
            status = "failed" if exc[0] else "done"
            log_phase(f"{label} {status} ({elapsed:.1f}s)", prefix=prefix)
            return False

    return _Timer()
