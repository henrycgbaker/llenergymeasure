"""Shared helpers for inference engine implementations.

Extracted from the repeated patterns in transformers.py, vllm.py, and tensorrt.py
to reduce duplication while keeping engines thin.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from llenergymeasure.utils.exceptions import EngineError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effective-params extraction (H3 source)
# ---------------------------------------------------------------------------


def extract_observed_params(
    native_obj: Any,
    *,
    private_field_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    """Dump a constructed native type's post-``__post_init__`` state.

    Used by the H3 hashing pipeline (sweep-dedup.md §3.2) — after each
    backend constructs its native type (``GenerationConfig``,
    ``SamplingParams``, ``LlmArgs``), the harness calls this to extract
    the authoritative effective parameters the library settled on.

    PoC-C finding (sweep-dedup.md §3.2 and §10 decision log): live
    libraries leak private state that poisons H3 if passed through
    unfiltered. Specific leaks observed:

    - ``transformers.GenerationConfig``: ``_commit_hash``,
      ``_from_model_config``
    - ``vllm.SamplingParams``: ``_all_stop_token_ids``,
      ``_bad_words_token_ids``, ``_eos_token_id``

    Default behaviour is to strip all ``_``-prefixed fields. Engines pass
    ``private_field_allowlist`` only when a specific private field is
    genuinely measurement-relevant (rare).

    Dispatch covers the three observed native-type shapes — Pydantic
    (``model_dump``), dataclass (``dataclasses.asdict``), and
    ``__slots__``-based (vLLM msgspec-adjacent). Fallback is ``__dict__``.
    """
    allow = private_field_allowlist or set()
    raw = _dump_raw(native_obj)
    return {k: v for k, v in raw.items() if not k.startswith("_") or k in allow}


def library_version(module_name: str) -> str:
    """Return ``__version__`` of an imported library or ``"unknown"`` on failure.

    Shared helper for the H3 capture path — all three engines publish the
    library version alongside their effective-params dump so researchers can
    disambiguate identical H3 hashes across library versions.
    """
    try:
        import importlib

        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unknown"


def assemble_observed_params_safe(
    *,
    engine_extractor: Callable[[], dict[str, Any]] | None,
    sampling_extractor: Callable[[], dict[str, Any]] | None,
    module_name: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Try-extract-and-assemble scaffold shared by all three engine capture methods.

    Each engine supplies small callables that produce its engine-specific and
    sampling-specific observed-params dicts.  Extraction failures are logged at
    DEBUG and silently produce empty dicts — sidecar writes are best-effort.

    Args:
        engine_extractor: Zero-arg callable returning engine-section params dict,
            or ``None`` to skip engine extraction.
        sampling_extractor: Zero-arg callable returning sampling-section params dict,
            or ``None`` to skip sampling extraction.
        module_name: Library module name for :func:`library_version` lookup.
        logger: Caller's module-level logger for DEBUG messages.

    Returns:
        ``{"engine": ..., "sampling": ..., "library_version": ...}`` dict.
    """
    engine_params: dict[str, Any] = {}
    if engine_extractor is not None:
        try:
            engine_params = engine_extractor()
        except Exception as exc:
            logger.debug("Engine param extraction failed: %s", exc)

    sampling: dict[str, Any] = {}
    if sampling_extractor is not None:
        try:
            sampling = sampling_extractor()
        except Exception as exc:
            logger.debug("Sampling param extraction failed: %s", exc)

    return assemble_observed_params(engine_params, sampling, module_name)


def assemble_observed_params(
    engine_params: dict[str, Any],
    sampling_params: dict[str, Any],
    module_name: str,
) -> dict[str, Any]:
    """Assemble the standard observed-params return dict for the sidecar.

    All three engines share the same return shape: ``{"engine": ...,
    "sampling": ..., "library_version": ...}``. Engine-specific extraction
    logic (BnB detection, vllm_config, llm.args) stays in each caller;
    this helper assembles the final dict and fills in the library version.

    Args:
        engine_params: Engine-specific observed params (engine section).
        sampling_params: Sampling-specific observed params.
        module_name: Library module name for :func:`library_version` lookup
            (e.g. ``"transformers"``, ``"vllm"``, ``"tensorrt_llm"``).
    """
    return {
        "engine": engine_params,
        "sampling": sampling_params,
        "library_version": library_version(module_name),
    }


def _dump_raw(native_obj: Any) -> dict[str, Any]:
    """Return a ``dict`` of every attribute observable on ``native_obj``.

    Order of attempts mirrors the three shapes :func:`extract_observed_params`
    commits to supporting; each is an explicit ``hasattr`` rather than
    ``try/except`` because some libraries raise non-``AttributeError`` for
    ``model_dump`` failures. The inner ``except Exception`` is intentional:
    sidecar-write errors should never crash a measurement run.
    """
    if hasattr(native_obj, "model_dump"):
        # Pydantic (transformers GenerationConfig inherits from PushToHubMixin;
        # its model_dump accepts mode="python").
        try:
            dumped = native_obj.model_dump(mode="python")
        except Exception:
            try:
                dumped = native_obj.model_dump()
            except Exception:
                dumped = None
        if isinstance(dumped, dict):
            return dict(dumped)
    if dataclasses.is_dataclass(native_obj) and not isinstance(native_obj, type):
        return dataclasses.asdict(native_obj)
    if hasattr(native_obj, "__slots__"):
        out: dict[str, Any] = {}
        for slot in native_obj.__slots__:
            if hasattr(native_obj, slot):
                out[slot] = getattr(native_obj, slot)
        if out:
            return out
    if hasattr(native_obj, "__dict__"):
        return dict(native_obj.__dict__)
    return {}


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
