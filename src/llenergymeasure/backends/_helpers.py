"""Shared helpers for inference backend implementations.

Extracted from the repeated patterns in pytorch.py, vllm.py, and tensorrt.py
to reduce duplication while keeping backends thin.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from llenergymeasure.config.models import WarmupConfig
from llenergymeasure.domain.metrics import WarmupResult
from llenergymeasure.utils.exceptions import BackendError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA memory helpers (extracted from pytorch.py, vllm.py, tensorrt.py)
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
# OOM / import error helpers (extracted from all 3 backends)
# ---------------------------------------------------------------------------


def is_oom_error(exc: Exception) -> bool:
    """Check whether an exception is a CUDA out-of-memory error."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def raise_backend_error(exc: Exception, backend_name: str, *, hint: str = "") -> None:
    """Raise a BackendError wrapping *exc* with a user-friendly message.

    If the error is OOM, includes remediation hints. Otherwise wraps
    generically as "<backend> inference failed".

    Args:
        exc: The original exception.
        backend_name: Human-readable backend name (e.g. "vLLM", "TRT-LLM").
        hint: Extra remediation text appended to OOM messages.
    """
    if is_oom_error(exc):
        msg = f"{backend_name} CUDA out of memory."
        if hint:
            msg = f"{msg} Try: {hint}"
        raise BackendError(f"{msg} Original error: {exc}") from exc
    raise BackendError(f"{backend_name} inference failed: {exc}") from exc


def require_import(module: str, extra_name: str) -> Any:
    """Import *module* or raise BackendError with install instructions.

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
        raise BackendError(
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


def warmup_until_converged(
    run_single_inference: Callable[[], float],
    config: WarmupConfig,
    *,
    show_progress: bool = True,
) -> WarmupResult:
    """Run warmup prompts until latency CV stabilises below threshold.

    Args:
        run_single_inference: Callable that runs one warmup prompt and
            returns latency in milliseconds. Decouples warmup from
            specific inference implementation.
        config: WarmupConfig controlling convergence parameters.
        show_progress: Whether to display tqdm progress bar.

    Returns:
        WarmupResult with convergence status and metrics.
    """
    # Early return if warmup is disabled
    if not config.enabled:
        logger.debug("Warmup disabled, skipping")
        return WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=0,
            target_cv=config.cv_threshold,
            max_prompts=config.max_prompts,
        )

    latencies: list[float] = []
    converged = False
    final_cv = 1.0

    # Fixed mode: run exactly n_warmup prompts without convergence checking.
    # CV mode: run up to max_prompts with convergence checking after min_prompts.
    fixed_mode = not config.convergence_detection
    iteration_limit = config.n_warmup if fixed_mode else config.max_prompts

    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(
            total=iteration_limit,
            desc="Warmup",
            unit="prompt",
        )

    try:
        for i in range(iteration_limit):
            # Run single inference, catching exceptions to avoid aborting warmup
            try:
                latency_ms = run_single_inference()
            except Exception as exc:
                logger.warning("Warmup prompt %d failed: %s", i + 1, exc)
                if progress is not None:
                    progress.update(1)
                continue

            latencies.append(latency_ms)

            # Check convergence (skip in fixed mode)
            if not fixed_mode and len(latencies) >= max(config.min_prompts, config.window_size):
                recent = latencies[-config.window_size :]
                cv = compute_cv(recent)
                final_cv = cv

                if cv < config.cv_threshold:
                    converged = True
                    if progress is not None:
                        progress.set_postfix(
                            cv=f"{final_cv:.1%}", target=f"{config.cv_threshold:.1%}"
                        )
                        progress.update(1)
                    break

            if progress is not None:
                progress.set_postfix(cv=f"{final_cv:.1%}", target=f"{config.cv_threshold:.1%}")
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    # In fixed mode, mark as converged (user chose fixed iterations)
    if fixed_mode:
        converged = True
        if latencies and len(latencies) >= config.window_size:
            final_cv = compute_cv(latencies[-config.window_size :])

    # Log result
    if converged and not fixed_mode:
        logger.info(
            f"Warmup converged after {len(latencies)} prompts "
            f"(CV={final_cv:.3f} < {config.cv_threshold})"
        )
    elif fixed_mode:
        logger.info(
            "Warmup completed %d fixed iterations (final CV=%.3f)", len(latencies), final_cv
        )
    else:
        logger.warning(
            f"Warmup did not converge after {iteration_limit} prompts "
            f"(final CV={final_cv:.3f}, target={config.cv_threshold})"
        )

    return WarmupResult(
        converged=converged,
        final_cv=final_cv,
        iterations_completed=len(latencies),
        target_cv=config.cv_threshold,
        max_prompts=config.max_prompts,
        latencies_ms=latencies,
    )


def create_warmup_inference_fn(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
) -> Callable[[], float]:
    """Create a single-inference callable for warmup.

    Returns a function that runs one inference and returns latency in ms.

    Args:
        model: HuggingFace model (or compatible).
        tokenizer: HuggingFace tokenizer.
        prompt: Warmup prompt text.
        max_new_tokens: Maximum tokens to generate per warmup prompt.

    Returns:
        Callable that runs one inference and returns latency in milliseconds.
    """

    def _run() -> float:
        import torch

        start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens)
        return (time.perf_counter() - start) * 1000.0

    return _run


def warmup_single_token(
    llm: Any,
    prompts: list[str],
    sampling_params_cls: type,
    **sp_kwargs: Any,
) -> None:
    """Run a minimal single-token warmup generation.

    Used by vLLM and TRT-LLM backends to warm up the engine before the
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
