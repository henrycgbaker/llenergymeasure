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

logger = logging.getLogger(__name__)


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
                mean = float(np.mean(recent))
                std = float(np.std(recent))
                cv = std / mean if mean > 0 else 0.0
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
            recent = latencies[-config.window_size :]
            mean = float(np.mean(recent))
            std = float(np.std(recent))
            final_cv = std / mean if mean > 0 else 0.0

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
