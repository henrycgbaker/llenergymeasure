"""TensorRT-LLM inference engine — thin EnginePlugin.

Implements the EnginePlugin protocol:
  load_model, warmup, run_inference, cleanup, check_hardware

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only TRT-LLM-specific inference: model loading via tensorrt_llm.LLM(),
a minimal 1-prompt warmup, offline batch llm.generate(), and cleanup.

All tensorrt_llm and torch imports are lazy so this module can be imported on
hosts without TRT-LLM or CUDA installed.

Engine compilation must NEVER occur inside the NVML measurement window.
The load_model() call triggers compilation; run_inference() assumes the
engine is ready.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.protocol import InferenceOutput
from llenergymeasure.utils.exceptions import ConfigError, EngineError

logger = logging.getLogger(__name__)


def _validate_engine_directory(engine_path: Path, tp_size: int) -> list[str]:
    """Pre-flight validation for TRT-LLM engine directory.

    Checks directory exists, config.json exists, tp_size matches, and
    rank{N}.engine files exist. Returns list of error strings (empty = valid).
    Does NOT re-implement TRT-LLM's format detection.

    Args:
        engine_path: Path to the engine directory.
        tp_size: Expected tensor-parallel size (number of rank files to check).

    Returns:
        List of error strings. Empty list means the directory is valid.
    """
    errors: list[str] = []

    if not engine_path.is_dir():
        errors.append(f"engine_path does not exist or is not a directory: {engine_path}")
        return errors

    config_path = engine_path / "config.json"
    if not config_path.exists():
        errors.append(f"config.json not found in engine directory: {engine_path}")
    else:
        try:
            with config_path.open() as f:
                config_data = json.load(f)
            engine_tp_size = (
                config_data.get("pretrained_config", {}).get("mapping", {}).get("tp_size")
            )
            if engine_tp_size is not None and engine_tp_size != tp_size:
                errors.append(
                    f"tp_size mismatch: engine was built with tp_size={engine_tp_size} "
                    f"but config requests tp_size={tp_size}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"Failed to parse config.json in engine directory: {exc}")

    for rank in range(tp_size):
        rank_file = engine_path / f"rank{rank}.engine"
        if not rank_file.exists():
            errors.append(f"rank{rank}.engine not found in engine directory: {engine_path}")

    return errors


def _apply_default_build_cache(kwargs: dict[str, Any]) -> None:
    """Apply the env-var-gated default TRT-LLM build cache to ``kwargs``.

    The opinionated llenergymeasure default is enabled (engine compilation is
    expensive; the cache is a large time-saver for repeat runs) — shipped via
    ``LLEM_TRT_BUILD_CACHE_ENABLED=1`` in ``.env.example``. The helpers are
    pure passthrough (see :mod:`llenergymeasure.utils.env_config`), so
    removing the line reverts to TRT-LLM's disabled default.

    - Disabled by env → ``enable_build_cache`` is not set (TRT-LLM default is
      False, matching the discovered schema).
    - Enabled with a user-supplied path → build a ``BuildCacheConfig`` whose
      ``cache_root`` is that path. Falls back to bare ``True`` if the
      ``tensorrt_llm.llmapi`` import is unavailable (mirrors the behaviour of
      the explicit YAML-driven path).
    - Enabled without a path → set ``enable_build_cache = True`` (preserves the
      pre-env-var behaviour).
    """
    from llenergymeasure.utils.env_config import trt_build_cache_enabled, trt_build_cache_path

    if not trt_build_cache_enabled():
        return

    cache_root = trt_build_cache_path()
    if cache_root is not None:
        try:
            from tensorrt_llm.llmapi import BuildCacheConfig

            kwargs["enable_build_cache"] = BuildCacheConfig(cache_root=cache_root)
            return
        except ImportError:
            logger.debug(
                "tensorrt_llm.llmapi not available; falling back to bare enable_build_cache=True"
            )

    kwargs["enable_build_cache"] = True


class TensorRTEngine:
    """TensorRT-LLM inference engine — offline batch mode, thin plugin.

    Implements EnginePlugin:
    - load_model: Compile/load engine via tensorrt_llm.LLM(), record build metadata
    - warmup: Minimal 1-prompt warmup with 1-token output
    - run_inference: Single llm.generate() call with ALL prompts, returns InferenceOutput
    - cleanup: Delete LLM instance, gc.collect(), clear CUDA cache
    - check_hardware: Check SM >= 7.5 (Turing) and FP8 requires SM >= 8.9
    """

    def __init__(self) -> None:
        self._build_metadata: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "tensorrt"

    @property
    def version(self) -> str:
        """TensorRT-LLM version string."""
        try:
            import tensorrt_llm

            return str(tensorrt_llm.__version__)
        except Exception:
            return "unknown"

    # -------------------------------------------------------------------------
    # EnginePlugin: load_model
    # -------------------------------------------------------------------------

    def load_model(
        self,
        config: ExperimentConfig,
        on_substep: Callable[[str, float], None] | None = None,
    ) -> tuple[Any, Any]:
        """Compile/load engine via tensorrt_llm.LLM() and build SamplingParams.

        Engine compilation happens here — BEFORE the NVML measurement window.
        Build metadata is recorded for inclusion in InferenceOutput.extras.

        All tensorrt_llm imports are lazy so this module can be imported without TRT-LLM.

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for substep visibility.

        Returns:
            Tuple of (llm, sampling_params).

        Raises:
            EngineError: If TRT-LLM is not installed or model loading fails.
        """
        from llenergymeasure.engines._helpers import require_import

        _trt_mod = require_import("tensorrt_llm", "tensorrt")
        LLM = _trt_mod.LLM

        kwargs = self._build_llm_kwargs(config)
        logger.info("Loading TRT-LLM model %r (kwargs: %s)", config.task.model, list(kwargs.keys()))

        # Collect build metadata before construction (for config_hash)
        config_hash = hashlib.sha256(
            json.dumps(kwargs, default=str, sort_keys=True).encode()
        ).hexdigest()[:16]

        from llenergymeasure.device.gpu_info import get_gpu_architecture

        gpu_arch = get_gpu_architecture()

        trt_version = "unknown"
        try:
            import tensorrt_llm as _trt

            trt_version = getattr(_trt, "__version__", "unknown")
        except Exception:
            pass

        build_start = time.perf_counter()

        try:
            llm = LLM(**kwargs)
        except Exception as e:
            raise EngineError(f"TensorRT-LLM model loading failed: {e}") from e

        build_time_sec = time.perf_counter() - build_start
        if on_substep is not None:
            on_substep(f"engine compiled ({gpu_arch}, TRT-LLM {trt_version})", build_time_sec)
        logger.debug(
            "TRT-LLM engine built in %.1fs (arch=%s, version=%s)",
            build_time_sec,
            gpu_arch,
            trt_version,
        )

        self._build_metadata = {
            "build_time_sec": build_time_sec,
            "gpu_architecture": gpu_arch,
            "trt_llm_version": trt_version,
            "config_hash": config_hash,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }

        sampling_params = self._build_sampling_params(config)
        if on_substep is not None:
            on_substep("sampling params built", 0.0)
        return llm, sampling_params

    # -------------------------------------------------------------------------
    # EnginePlugin: warmup
    # -------------------------------------------------------------------------

    def run_warmup_prompt(self, config: ExperimentConfig, model: Any, prompt: str) -> float:
        """Run one warmup prompt via single-token kernel warmup. Returns 0.0.

        Returns 0.0 to signal the harness to skip CV-based convergence.
        TRT-LLM uses a single-token kernel warmup rather than CV convergence.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompt: Single warmup prompt text.

        Returns:
            0.0 (signals harness to skip convergence loop).
        """
        from tensorrt_llm import SamplingParams

        from llenergymeasure.engines._helpers import warmup_single_token

        llm, _sampling_params = model
        warmup_single_token(llm, [prompt], SamplingParams, max_tokens=1)
        return 0.0  # Signals harness to skip CV loop

    # -------------------------------------------------------------------------
    # EnginePlugin: run_inference
    # -------------------------------------------------------------------------

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run offline batch inference over all prompts.

        Single llm.generate() call with ALL prompts — no streaming.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, memory stats, and build_metadata.

        Raises:
            EngineError: On OOM or other inference failures.
        """
        import time

        llm, sampling_params = model

        # Reset peak stats before the measurement loop
        from llenergymeasure.engines._helpers import reset_cuda_peak_memory

        reset_cuda_peak_memory()

        logger.info(
            "Starting TRT-LLM offline batch inference: %d prompts, max_tokens=%s",
            len(prompts),
            config.task.max_output_tokens or "unlimited",
        )

        try:
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            from llenergymeasure.engines._helpers import raise_engine_error

            raise_engine_error(
                e,
                "TRT-LLM",
                hint="reduce n, use a smaller max_batch_size, or use a smaller model.",
            )

        # Capture peak memory
        from llenergymeasure.engines._helpers import get_cuda_peak_memory_mb

        peak_mb = get_cuda_peak_memory_mb()

        # Count tokens from RequestOutput objects (same pattern as vLLM)
        input_token_count = sum(
            len(o.prompt_token_ids) for o in outputs if hasattr(o, "prompt_token_ids")
        )
        output_token_count = sum(
            len(out.token_ids)
            for o in outputs
            if hasattr(o, "outputs") and o.outputs
            for out in o.outputs
        )

        logger.debug(
            "TRT-LLM inference complete: %d total tokens (in=%d, out=%d) in %.2fs",
            input_token_count + output_token_count,
            input_token_count,
            output_token_count,
            elapsed,
        )

        extras: dict[str, Any] = {}
        if self._build_metadata is not None:
            extras["build_metadata"] = self._build_metadata

        # Library-observed effective params for H3 (sweep-dedup.md §3).
        # TRT-LLM runs in NGC container only; allowlist verification is
        # deferred to a container run per the PoC-C notes.
        effective_params = self._capture_effective_params(config, llm, sampling_params)
        extras["effective_engine_params"] = effective_params["engine"]
        extras["effective_sampling_params"] = effective_params["sampling"]
        extras["library_version"] = effective_params["library_version"]

        return InferenceOutput(
            elapsed_time_sec=elapsed,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            peak_memory_mb=peak_mb,
            model_memory_mb=0.0,  # Captured by harness before run_inference
            batch_times=[elapsed],
            extras=extras,
        )

    # -------------------------------------------------------------------------
    # Private: effective-params capture (H3)
    # -------------------------------------------------------------------------

    @staticmethod
    def _capture_effective_params(
        config: ExperimentConfig,
        llm: Any,
        sampling_params: Any,
    ) -> dict[str, Any]:
        """Extract post-construction state from the TRT-LLM native types.

        TRT-LLM's ``LlmArgs`` + nested ``BuildConfig`` are Pydantic; accessible
        on ``llm.args`` in current releases. Private fields (if any surface in
        a given TRT-LLM version) are stripped by the default
        ``_``-prefix allowlist behaviour in
        :func:`extract_effective_params`.
        """
        import contextlib as _cl

        from llenergymeasure.engines._helpers import extract_effective_params, library_version

        sampling: dict[str, Any] = {}
        with _cl.suppress(Exception):
            sampling = extract_effective_params(sampling_params)

        engine_params: dict[str, Any] = {}
        with _cl.suppress(Exception):
            llm_args = getattr(llm, "args", None)
            if llm_args is not None:
                engine_params = extract_effective_params(llm_args)

        return {
            "engine": engine_params,
            "sampling": sampling,
            "library_version": library_version("tensorrt_llm"),
        }

    # -------------------------------------------------------------------------
    # EnginePlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release TRT-LLM model from memory and clear CUDA cache.

        Args:
            model: Tuple of (llm, sampling_params) from load_model().
        """
        from llenergymeasure.engines._helpers import cleanup_model

        llm, _sampling_params = model
        cleanup_model(llm)
        logger.debug("TRT-LLM model cleanup complete")

    # -------------------------------------------------------------------------
    # EnginePlugin: check_hardware
    # -------------------------------------------------------------------------

    @staticmethod
    def check_hardware(config: ExperimentConfig) -> list[str]:
        """Check SM capability + FP8 requirements against the visible GPU.

        Independent of engine-kwargs construction (fixes the pre-50.2c T5
        short-circuit: ``probe_config`` returned early on T0 kwargs failure,
        skipping hardware compat entirely). Checks:
          - SM >= 7.5 (Turing minimum for TRT-LLM)
          - FP8 requires SM >= 8.9 (Ada Lovelace / Hopper); A100 is SM 8.0

        Returns empty list when no GPU is visible (cannot block at config
        time inside a container without a visible device).
        """
        from llenergymeasure.device.gpu_info import get_compute_capability

        sm = get_compute_capability()
        if sm is None:
            return []

        major, minor = sm
        sm_float = major + minor / 10
        errors: list[str] = []

        if sm_float < 7.5:
            errors.append(
                f"TensorRT-LLM requires SM >= 7.5 (Turing). This GPU has SM {major}.{minor}."
            )

        trt = config.tensorrt
        if trt is not None and trt.quant is not None:
            if trt.quant.quant_algo == "FP8" and sm_float < 8.9:
                errors.append(
                    f"FP8 quantisation requires SM >= 8.9 (Ada Lovelace or Hopper). "
                    f"This GPU has SM {major}.{minor} "
                    f"(A100=8.0, H100=9.0, RTX4090=8.9). "
                    f"Use W8A16, W4A16_AWQ, or W4A16_GPTQ instead."
                )
            if trt.quant.kv_cache_quant_algo == "FP8" and sm_float < 8.9:
                errors.append(
                    f"FP8 KV cache quantisation requires SM >= 8.9 (Ada Lovelace or Hopper). "
                    f"This GPU has SM {major}.{minor} "
                    f"(A100=8.0, H100=9.0, RTX4090=8.9). "
                    f"Use INT8 KV cache quantisation instead."
                )

        return errors

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    def _build_llm_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs dict for tensorrt_llm.LLM() constructor.

        Starts with {"model": config.model} and applies all non-None fields
        from TensorRTConfig, including the typed ``backend`` field when set.
        When ``backend`` is unset (None), TRT-LLM auto-picks (respecting
        ``TLLM_USE_TRT_ENGINE``) — the previous hardcoded ``"trt"`` default
        is removed.

        When engine_path is set, returns early with only {"model": engine_path}
        plus ``backend`` iff the typed field was supplied. Compile-time kwargs
        are baked into the engine and must not be re-specified.
        """
        kwargs: dict[str, Any] = {
            "model": config.task.model,
        }

        trt = config.tensorrt

        # engine_path early-return: pass engine dir as model, skip all compile-time kwargs
        # engine_path is no longer a typed field (D1 drop); accessed via extra="allow" passthrough.
        raw_engine_path = getattr(trt, "engine_path", None) if trt is not None else None
        if trt is not None and raw_engine_path is not None:
            engine_path = Path(str(raw_engine_path))
            tp_size = trt.tensor_parallel_size if trt.tensor_parallel_size is not None else 1
            errors = _validate_engine_directory(engine_path, tp_size=tp_size)
            if errors:
                raise ConfigError(f"engine_path validation failed: {'; '.join(errors)}")
            # Pass engine dir as model - TRT-LLM auto-detects TLLM_ENGINE format.
            # Compile-time kwargs are baked into the engine; don't pass them.
            # enable_build_cache is not set - engine format bypasses it.
            early_kwargs: dict[str, Any] = {"model": str(raw_engine_path)}
            if trt.backend is not None:
                early_kwargs["backend"] = trt.backend
            return early_kwargs

        if trt is None:
            # No tensorrt section — apply env-var-gated default build cache.
            _apply_default_build_cache(kwargs)
            return kwargs

        # Scalar fields: map directly
        if trt.tensor_parallel_size is not None:
            kwargs["tensor_parallel_size"] = trt.tensor_parallel_size
        if trt.pipeline_parallel_size is not None:
            kwargs["pipeline_parallel_size"] = trt.pipeline_parallel_size
        if trt.max_batch_size is not None:
            kwargs["max_batch_size"] = trt.max_batch_size
        if trt.max_input_len is not None:
            kwargs["max_input_len"] = trt.max_input_len
        if trt.max_seq_len is not None:
            kwargs["max_seq_len"] = trt.max_seq_len
        if trt.max_num_tokens is not None:
            kwargs["max_num_tokens"] = trt.max_num_tokens
        if trt.dtype is not None:
            kwargs["dtype"] = trt.dtype
        if trt.fast_build is not None:
            kwargs["fast_build"] = trt.fast_build
        if trt.backend is not None:
            kwargs["backend"] = trt.backend

        # Quantisation config
        if trt.quant is not None:
            try:
                from tensorrt_llm.llmapi import QuantAlgo, QuantConfig

                qc_kwargs: dict[str, Any] = {}
                if trt.quant.quant_algo is not None:
                    qc_kwargs["quant_algo"] = QuantAlgo[trt.quant.quant_algo]
                if trt.quant.kv_cache_quant_algo is not None:
                    qc_kwargs["kv_cache_quant_algo"] = QuantAlgo[trt.quant.kv_cache_quant_algo]
                if qc_kwargs:
                    kwargs["quantization"] = QuantConfig(**qc_kwargs)
            except ImportError:
                logger.debug("tensorrt_llm.llmapi not available; skipping QuantConfig")

        # Build cache — env-var-gated default.
        # TensorRTBuildCacheConfig was dropped (D1); advanced config via extra="allow".
        _apply_default_build_cache(kwargs)

        # KV cache config
        if trt.kv_cache is not None:
            try:
                from tensorrt_llm.llmapi import KvCacheConfig

                kv = trt.kv_cache
                kv_kwargs: dict[str, Any] = {}
                if kv.enable_block_reuse is not None:
                    kv_kwargs["enable_block_reuse"] = kv.enable_block_reuse
                if kv.free_gpu_memory_fraction is not None:
                    kv_kwargs["free_gpu_memory_fraction"] = kv.free_gpu_memory_fraction
                if kv.max_tokens is not None:
                    kv_kwargs["max_tokens"] = kv.max_tokens
                if kv.host_cache_size is not None:
                    kv_kwargs["host_cache_size"] = kv.host_cache_size
                if kv_kwargs:
                    kwargs["kv_cache_config"] = KvCacheConfig(**kv_kwargs)
            except ImportError:
                logger.debug("tensorrt_llm.llmapi not available; skipping KvCacheConfig")

        # Scheduler config
        if trt.scheduler is not None:
            try:
                from tensorrt_llm.llmapi import CapacitySchedulerPolicy, SchedulerConfig

                sc = trt.scheduler
                sc_kwargs: dict[str, Any] = {}
                if sc.capacity_scheduling_policy is not None:
                    sc_kwargs["capacity_scheduling_policy"] = CapacitySchedulerPolicy[
                        sc.capacity_scheduling_policy
                    ]
                if sc_kwargs:
                    kwargs["scheduler_config"] = SchedulerConfig(**sc_kwargs)
            except ImportError:
                logger.debug("tensorrt_llm.llmapi not available; skipping SchedulerConfig")

        return kwargs

    def _build_sampling_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build the effective TRT-LLM SamplingParams kwargs (no constructor call).

        Pure extraction of the kwargs-assembly portion of
        :meth:`_build_sampling_params`. Kept as a separate method so tests and
        the probe adapter can inspect effective kwargs without importing
        tensorrt_llm.
        """
        trt = config.tensorrt
        sampling = trt.sampling if trt is not None else None

        kwargs: dict[str, Any] = (
            sampling.model_dump(exclude_none=True) if sampling is not None else {}
        )
        kwargs["random_seed"] = config.task.random_seed
        if config.task.max_output_tokens is not None:
            kwargs["max_new_tokens"] = config.task.max_output_tokens
        return kwargs

    def _build_sampling_params(self, config: ExperimentConfig) -> Any:
        """Build tensorrt_llm.SamplingParams from TensorRTSamplingConfig.

        All sampling fields live on ``config.tensorrt.sampling``. None values
        mean "use TRT-LLM's default", so we forward only explicit values.
        User writes top_k=0 to disable (TRT convention, matches HF).
        """
        from tensorrt_llm import SamplingParams

        kwargs = self._build_sampling_kwargs(config)
        return SamplingParams(**kwargs)
