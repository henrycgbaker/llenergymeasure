"""TensorRT-LLM inference backend — thin BackendPlugin.

Implements the 6-method BackendPlugin protocol:
  load_model, warmup, run_inference, cleanup, validate_config

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

from llenergymeasure.backends.protocol import InferenceOutput
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.metrics import WarmupResult
from llenergymeasure.utils.exceptions import BackendError, ConfigError

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


class TensorRTBackend:
    """TensorRT-LLM inference backend — offline batch mode, thin plugin.

    Implements BackendPlugin:
    - load_model: Compile/load engine via tensorrt_llm.LLM(), record build metadata
    - warmup: Minimal 1-prompt warmup with 1-token output
    - run_inference: Single llm.generate() call with ALL prompts, returns InferenceOutput
    - cleanup: Delete LLM instance, gc.collect(), clear CUDA cache
    - validate_config: Check SM >= 7.5 (Turing) and FP8 requires SM >= 8.9
    """

    def __init__(self) -> None:
        self._build_metadata: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "tensorrt"

    # -------------------------------------------------------------------------
    # BackendPlugin: load_model
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
            BackendError: If TRT-LLM is not installed or model loading fails.
        """
        from llenergymeasure.backends._helpers import require_import

        _trt_mod = require_import("tensorrt_llm", "tensorrt")
        LLM = _trt_mod.LLM

        kwargs = self._build_llm_kwargs(config)
        logger.info("Loading TRT-LLM model %r (kwargs: %s)", config.model, list(kwargs.keys()))

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
            raise BackendError(f"TensorRT-LLM model loading failed: {e}") from e

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

        trt = config.tensorrt
        if trt is not None and trt.engine_path is not None:
            self._build_metadata["engine_path"] = trt.engine_path

        sampling_params = self._build_sampling_params(config)
        if on_substep is not None:
            on_substep("sampling params built", 0.0)
        return llm, sampling_params

    # -------------------------------------------------------------------------
    # BackendPlugin: warmup
    # -------------------------------------------------------------------------

    def warmup(self, config: ExperimentConfig, model: Any, prompts: list[str]) -> WarmupResult:
        """Run minimal TRT-LLM warmup: 1 prompt, 1 token.

        thermal_floor_wait_s is NOT set here — MeasurementHarness sets it after
        this method returns.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            WarmupResult. thermal_floor_wait_s is left at default 0.0 (set by harness).
        """
        llm, _sampling_params = model

        if config.warmup.enabled:
            logger.debug("Running TRT-LLM warmup (1 prompt, 1 token)...")
            from tensorrt_llm import SamplingParams as _SP

            from llenergymeasure.backends._helpers import warmup_single_token

            warmup_single_token(llm, prompts, _SP, max_new_tokens=1)
            logger.debug("TRT-LLM warmup complete")

        return WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=1 if config.warmup.enabled else 0,
            target_cv=config.warmup.cv_threshold,
            max_prompts=config.warmup.max_prompts,
        )

    # -------------------------------------------------------------------------
    # BackendPlugin: run_inference
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
            BackendError: On OOM or other inference failures.
        """
        import time

        llm, sampling_params = model

        # Reset peak stats before the measurement loop
        from llenergymeasure.backends._helpers import reset_cuda_peak_memory

        reset_cuda_peak_memory()

        logger.info(
            "Starting TRT-LLM offline batch inference: %d prompts, max_tokens=%s",
            len(prompts),
            config.max_output_tokens or "unlimited",
        )

        try:
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            from llenergymeasure.backends._helpers import raise_backend_error

            raise_backend_error(
                e,
                "TRT-LLM",
                hint="reduce n, use a smaller max_batch_size, or use a smaller model.",
            )

        # Capture peak memory
        from llenergymeasure.backends._helpers import get_cuda_peak_memory_mb

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
    # BackendPlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release TRT-LLM model from memory and clear CUDA cache.

        Args:
            model: Tuple of (llm, sampling_params) from load_model().
        """
        from llenergymeasure.backends._helpers import cleanup_model

        llm, _sampling_params = model
        cleanup_model(llm)
        logger.debug("TRT-LLM model cleanup complete")

    # -------------------------------------------------------------------------
    # BackendPlugin: validate_config
    # -------------------------------------------------------------------------

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        """Validate TensorRT-LLM hardware and quantisation compatibility.

        Checks:
          - SM >= 7.5 (Turing minimum for TRT-LLM)
          - FP8 requires SM >= 8.9 (Ada Lovelace / Hopper); A100 is SM 8.0

        Args:
            config: Experiment configuration.

        Returns:
            List of error strings. Empty list means config is valid.
        """
        from llenergymeasure.device.gpu_info import get_compute_capability

        sm = get_compute_capability()
        if sm is None:
            # Cannot detect GPU — don't block (may be inside container at config time)
            return []

        major, minor = sm
        sm_float = major + minor / 10
        errors: list[str] = []

        if sm_float < 7.5:
            errors.append(
                f"TensorRT-LLM requires SM >= 7.5 (Turing). This GPU has SM {major}.{minor}."
            )

        # Check FP8 quant requirements (SM >= 8.9)
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

        Starts with {"model": config.model, "backend": "trt"} and applies
        all non-None fields from TensorRTConfig.

        When engine_path is set, returns early with only {"model": engine_path, "backend": "trt"}.
        Compile-time kwargs are baked into the engine and must not be re-specified.
        """
        kwargs: dict[str, Any] = {
            "model": config.model,
            "backend": "trt",
        }

        trt = config.tensorrt

        # engine_path early-return: pass engine dir as model, skip all compile-time kwargs
        if trt is not None and trt.engine_path is not None:
            engine_path = Path(trt.engine_path)
            tp_size = trt.tp_size if trt.tp_size is not None else 1
            errors = _validate_engine_directory(engine_path, tp_size=tp_size)
            if errors:
                raise ConfigError(f"engine_path validation failed: {'; '.join(errors)}")
            # Pass engine dir as model - TRT-LLM auto-detects TLLM_ENGINE format.
            # Compile-time kwargs are baked into the engine; don't pass them.
            # enable_build_cache is not set - engine format bypasses it.
            return {"model": str(trt.engine_path), "backend": "trt"}

        if trt is None:
            # No tensorrt section — use defaults + enable build cache
            kwargs["enable_build_cache"] = True
            return kwargs

        # Scalar fields: map directly (same name or renamed)
        if trt.tp_size is not None:
            kwargs["tensor_parallel_size"] = trt.tp_size
        if trt.max_batch_size is not None:
            kwargs["max_batch_size"] = trt.max_batch_size
        if trt.max_input_len is not None:
            kwargs["max_input_len"] = trt.max_input_len
        if trt.max_seq_len is not None:
            kwargs["max_seq_len"] = trt.max_seq_len
        if trt.dtype is not None:
            kwargs["dtype"] = trt.dtype
        if trt.fast_build is not None:
            kwargs["fast_build"] = trt.fast_build

        # TRT-LLM internal backend (overrides default "trt" if explicitly set)
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

        # Build cache config
        if trt.build_cache is not None:
            try:
                from tensorrt_llm.llmapi import BuildCacheConfig

                bc = trt.build_cache
                bc_kwargs: dict[str, Any] = {}
                if bc.cache_root is not None:
                    bc_kwargs["cache_root"] = Path(bc.cache_root)
                if bc.max_records is not None:
                    bc_kwargs["max_records"] = bc.max_records
                if bc.max_cache_storage_gb is not None:
                    bc_kwargs["max_cache_storage_gb"] = bc.max_cache_storage_gb
                kwargs["enable_build_cache"] = BuildCacheConfig(**bc_kwargs)
            except ImportError:
                logger.debug("tensorrt_llm.llmapi not available; enabling default build cache")
                kwargs["enable_build_cache"] = True
        else:
            # Enable default build cache when no explicit config given
            kwargs["enable_build_cache"] = True

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

        # Calibration config
        if trt.calib is not None:
            try:
                from tensorrt_llm.llmapi import CalibConfig

                calib = trt.calib
                calib_kwargs: dict[str, Any] = {}
                if calib.calib_batches is not None:
                    calib_kwargs["calib_batches"] = calib.calib_batches
                if calib.calib_dataset is not None:
                    calib_kwargs["calib_dataset"] = calib.calib_dataset
                if calib.calib_max_seq_length is not None:
                    calib_kwargs["calib_max_seq_length"] = calib.calib_max_seq_length
                if calib_kwargs:
                    kwargs["calib_config"] = CalibConfig(**calib_kwargs)
            except ImportError:
                logger.debug("tensorrt_llm.llmapi not available; skipping CalibConfig")

        return kwargs

    def _build_sampling_params(self, config: ExperimentConfig) -> Any:
        """Build tensorrt_llm.SamplingParams from DecoderConfig and TensorRTSamplingConfig.

        Args:
            config: Experiment configuration.

        Returns:
            tensorrt_llm.SamplingParams instance.
        """
        from tensorrt_llm import SamplingParams

        decoder = config.decoder
        kwargs: dict[str, Any] = {
            "random_seed": config.random_seed,
        }
        if config.max_output_tokens is not None:
            kwargs["max_new_tokens"] = config.max_output_tokens

        # Universal decoder params
        if decoder.temperature != 0.0:
            kwargs["temperature"] = decoder.temperature
        if decoder.top_p is not None:
            kwargs["top_p"] = decoder.top_p
        if decoder.top_k is not None and decoder.top_k != 0:
            kwargs["top_k"] = decoder.top_k
        if decoder.repetition_penalty is not None:
            kwargs["repetition_penalty"] = decoder.repetition_penalty
        if decoder.min_p is not None:
            kwargs["min_p"] = decoder.min_p
        # Map universal decoder.min_new_tokens to TRT-LLM's min_tokens.
        # Placed before trt.sampling overrides so tensorrt.sampling.min_tokens
        # can override the universal mapping if both are set.
        if decoder.min_new_tokens is not None:
            kwargs["min_tokens"] = decoder.min_new_tokens

        # TRT-LLM-specific sampling overrides
        trt = config.tensorrt
        if trt is not None and trt.sampling is not None:
            sampling = trt.sampling
            if sampling.min_tokens is not None:
                kwargs["min_tokens"] = sampling.min_tokens
            if sampling.n is not None:
                kwargs["n"] = sampling.n
            if sampling.ignore_eos is not None:
                kwargs["ignore_eos"] = sampling.ignore_eos
            if sampling.return_perf_metrics is not None:
                kwargs["return_perf_metrics"] = sampling.return_perf_metrics

        return SamplingParams(**kwargs)
