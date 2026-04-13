"""Shared helpers for inference engine implementations.

Extracted from the repeated patterns in transformers.py, vllm.py, and tensorrt.py
to reduce duplication while keeping engines thin.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from llenergymeasure.utils.exceptions import EngineError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA memory helpers (extracted from transformers.py, vllm.py, tensorrt.py)
# ---------------------------------------------------------------------------


def reset_cuda_peak_memory() -> None:
    """Reset CUDA peak memory stats before a measurement window.

    Best-effort — silently ignores failures (e.g. no CUDA, no torch).
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_cuda_peak_memory_mb() -> float:
    """Return peak GPU memory allocated in MB since last reset.

    Returns 0.0 if torch/CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# OOM / import error helpers (extracted from all 3 engines)
# ---------------------------------------------------------------------------


def is_oom_error(exc: Exception) -> bool:
    """Check whether an exception is a CUDA out-of-memory error."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def raise_engine_error(exc: Exception, engine_name: str, *, hint: str = "") -> None:
    """Raise a EngineError wrapping *exc* with a user-friendly message.

    If the error is OOM, includes remediation hints. Otherwise wraps
    generically as "<engine> inference failed".

    Args:
        exc: The original exception.
        engine_name: Human-readable engine name (e.g. "vLLM", "TRT-LLM").
        hint: Extra remediation text appended to OOM messages.
    """
    if is_oom_error(exc):
        msg = f"{engine_name} CUDA out of memory."
        if hint:
            msg = f"{msg} Try: {hint}"
        raise EngineError(f"{msg} Original error: {exc}") from exc
    raise EngineError(f"{engine_name} inference failed: {exc}") from exc


def require_import(module: str, extra_name: str) -> Any:
    """Import *module* or raise EngineError with install instructions.

    Args:
        module: Fully-qualified module name (e.g. "vllm").
        extra_name: pip extra name (e.g. "vllm", "tensorrt").

    Returns:
        The imported module object.
    """
    try:
        import importlib

        return importlib.import_module(module)
    except ImportError as e:
        raise EngineError(
            f"{module} is not installed. Install it with: pip install llenergymeasure[{extra_name}]"
        ) from e


# ---------------------------------------------------------------------------
# CV calculation helper
# ---------------------------------------------------------------------------


def compute_cv(values: list[float]) -> float:
    """Compute the coefficient of variation (std / mean) for *values*.

    Returns 0.0 when the mean is zero or negative (avoids division by zero).
    """
    mean = float(np.mean(values))
    if mean <= 0:
        return 0.0
    return float(np.std(values)) / mean


def cleanup_model(model_obj: Any, *, use_gc: bool = True) -> None:
    """Release a model object from GPU memory and clear CUDA cache.

    Args:
        model_obj: The model object to delete (e.g. HF model, vLLM LLM, TRT-LLM LLM).
        use_gc: Whether to run gc.collect() after deletion. PyTorch/Transformers
            skips this (deterministic refcount cleanup); vLLM and TRT-LLM need it
            to break circular references in their engine internals.
    """
    del model_obj
    if use_gc:
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    except Exception:
        logger.debug("CUDA cleanup failed", exc_info=True)


def warmup_single_token(
    llm: Any,
    prompts: list[str],
    sampling_params_cls: type,
    **sp_kwargs: Any,
) -> None:
    """Run a minimal single-token warmup generation.

    Used by vLLM and TRT-LLM engines to warm up the engine before the
    measurement window. Takes the first prompt and generates 1 token.

    Args:
        llm: The LLM engine object (vllm.LLM or tensorrt_llm.LLM).
        prompts: List of prompts (only the first is used).
        sampling_params_cls: The SamplingParams class to instantiate.
        **sp_kwargs: Keyword arguments for SamplingParams constructor.
            Defaults to temperature=0.0 if no kwargs provided.
    """
    if not sp_kwargs:
        sp_kwargs = {"temperature": 0.0}
    warmup_params = sampling_params_cls(**sp_kwargs)
    llm.generate(prompts[:1], warmup_params)
