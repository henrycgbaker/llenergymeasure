"""vLLM inference backend (ground-up rewrite for offline batch inference).

This is a clean implementation using vllm.LLM() — not an adaptation of the
v1.x inference_backends/vllm.py. The streaming P0 bug (CM-07) is resolved
structurally: offline batch mode only, no streaming code exists.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    compute_measurement_config_hash,
)
from llenergymeasure.domain.metrics import ThermalThrottleInfo, WarmupResult
from llenergymeasure.exceptions import BackendError

logger = logging.getLogger(__name__)


@dataclass
class _VLLMMeasurementData:
    """Internal container for vLLM measurement results."""

    total_tokens: int = 0
    total_time_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    batch_times: list[float] = field(default_factory=list)


class VLLMBackend:
    """vLLM inference backend — offline batch mode.

    Owns the full experiment lifecycle using vllm.LLM():
    1. Environment snapshot (before model load)
    2. Baseline power measurement (before model load — CM-17)
    3. Model loading via vllm.LLM()
    4. Prompt preparation
    5. Warmup (single prompt, 1 token)
    6. Thermal floor wait
    7. Energy tracking start
    8. CUDA sync before inference
    9. Offline batch inference — single llm.generate() call with ALL prompts
    10. CUDA sync after inference
    11. Energy tracking stop
    12. FLOPs estimation (best-effort, None on failure)
    13. Timeseries write
    14. Measurement warnings
    15. Result assembly
    16. VRAM cleanup

    CM-07 (streaming bug) is resolved structurally — no streaming code exists.
    All vLLM and torch imports are lazy so this module can be imported on
    hosts without vLLM or CUDA installed.
    """

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "vllm"

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete vLLM offline batch inference experiment.

        Args:
            config: Fully resolved experiment configuration.

        Returns:
            ExperimentResult with all measurement fields populated.

        Raises:
            BackendError: If model loading or inference fails.
        """
        # 1. Environment snapshot (BEFORE model loading — CM-32)
        from llenergymeasure.domain.environment import collect_environment_snapshot

        logger.info("Collecting environment snapshot before model load")
        snapshot = collect_environment_snapshot()

        # 2. Baseline power measurement (BEFORE model load — CM-17, CM-20)
        baseline = None
        if config.baseline.enabled:
            from llenergymeasure.core.baseline import measure_baseline_power

            logger.info("Measuring baseline power (%.0fs)...", config.baseline.duration_seconds)
            baseline = measure_baseline_power(
                duration_sec=config.baseline.duration_seconds,
            )

        # 3. Load model via vllm.LLM()
        llm, sampling_params = self._load_model(config)

        try:
            # 4. Prepare prompts (M1 placeholder — same pattern as PyTorchBackend)
            prompts = self._prepare_prompts(config)

            # 5. Warmup: single prompt, 1 token (minimal GPU warm-up)
            if config.warmup.enabled:
                logger.info("Running vLLM warmup (1 prompt, 1 token)...")
                from vllm import SamplingParams as _SP

                warmup_params = _SP(max_tokens=1, temperature=0.0)
                llm.generate(prompts[:1], warmup_params)
                logger.info("vLLM warmup complete")

            warmup_result = WarmupResult(
                converged=True,
                final_cv=0.0,
                iterations_completed=1 if config.warmup.enabled else 0,
                target_cv=config.warmup.cv_threshold,
                max_prompts=config.warmup.max_prompts,
            )

            # 6. Thermal floor (CM-22) — sleep after warmup before energy tracking
            thermal_floor_wait_s = 0.0
            if config.warmup.enabled and config.warmup.thermal_floor_seconds > 0:
                logger.info(
                    "Thermal stabilisation: waiting %.0fs...",
                    config.warmup.thermal_floor_seconds,
                )
                t0 = time.monotonic()
                time.sleep(config.warmup.thermal_floor_seconds)
                thermal_floor_wait_s = time.monotonic() - t0
            warmup_result.thermal_floor_wait_s = thermal_floor_wait_s

            # 7. Select energy backend (CM-14)
            from llenergymeasure.core.energy_backends import select_energy_backend

            energy_backend = select_energy_backend(config.energy.backend)

            # 8. Start energy tracking (after warmup + thermal floor)
            energy_tracker = None
            if energy_backend is not None:
                energy_tracker = energy_backend.start_tracking()

            # 9. CUDA sync BEFORE inference (Zeus best practice — CM-15)
            self._cuda_sync()

            # 10-11. Inference + thermal sampler for timeseries and thermal info
            start_time = datetime.now()
            result_data, thermal_info, timeseries_samples = self._run_measurement(
                llm, sampling_params, config, prompts
            )

            # 12. CUDA sync AFTER inference, before stopping energy (CM-15)
            self._cuda_sync()

            # 13. Stop energy tracking
            energy_measurement = None
            if energy_backend is not None and energy_tracker is not None:
                energy_measurement = energy_backend.stop_tracking(energy_tracker)
            end_time = datetime.now()

            # 14. FLOPs estimation — vLLM doesn't expose a HuggingFace model with .config,
            # so wrap in try/except and default to None on failure (best-effort)
            flops_result = None
            try:
                from llenergymeasure.core.flops import estimate_flops_palm

                # vllm.LLM has llm_engine.model_executor.driver_worker.model_runner.model
                # This is an internal API path — attempt it, catch AttributeError gracefully
                hf_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
                flops_result = estimate_flops_palm(
                    model=hf_model,
                    n_input_tokens=result_data.input_tokens,
                    n_output_tokens=result_data.output_tokens,
                )
            except Exception as e:
                logger.debug(
                    "FLOPs estimation skipped for vLLM backend (no HuggingFace model object): %s",
                    e,
                )

        finally:
            # Always release model from memory
            self._cleanup(llm)

        # 15. Write timeseries Parquet sidecar (if output_dir set — CM-16)
        timeseries_path: str | None = None
        if config.output_dir is not None and timeseries_samples:
            from llenergymeasure.core.timeseries import write_timeseries_parquet

            ts_file = write_timeseries_parquet(
                timeseries_samples,
                Path(config.output_dir) / "timeseries.parquet",
                gpu_index=0,
            )
            timeseries_path = ts_file.name  # relative name in result JSON

        # 16. Collect measurement quality warnings (CM-25 implied)
        measurement_warnings = self._collect_warnings(
            duration_sec=(end_time - start_time).total_seconds(),
            timeseries_samples=timeseries_samples,
        )

        # 17. Build ExperimentResult with real values
        return self._build_result(
            config=config,
            data=result_data,
            snapshot=snapshot,
            start_time=start_time,
            end_time=end_time,
            thermal_info=thermal_info,
            energy_measurement=energy_measurement,
            baseline=baseline,
            flops_result=flops_result,
            warmup_result=warmup_result,
            timeseries_path=timeseries_path,
            measurement_warnings=measurement_warnings,
        )

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self, config: ExperimentConfig) -> tuple[Any, Any]:
        """Load model via vllm.LLM() and build SamplingParams.

        All vLLM imports are lazy so this module can be imported without vLLM.

        Args:
            config: Experiment configuration.

        Returns:
            Tuple of (llm, sampling_params).

        Raises:
            BackendError: If vLLM is not installed or model loading fails.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise BackendError(
                "vLLM is not installed. Install it with: pip install llenergymeasure[vllm]"
            ) from e

        kwargs = self._build_llm_kwargs(config)
        logger.info(
            "Loading model %r with vllm.LLM (kwargs: %s)", config.model, list(kwargs.keys())
        )

        try:
            llm = LLM(**kwargs)
        except Exception as e:
            raise BackendError(f"vLLM model loading failed: {e}") from e

        logger.info("vLLM model loaded successfully")

        # Build SamplingParams or BeamSearchParams depending on config
        if config.vllm is not None and config.vllm.beam_search is not None:
            sampling_params = self._build_beam_search_params(config, config.vllm.beam_search)
        else:
            sampling_params = self._build_sampling_params(config, SamplingParams)
        return llm, sampling_params

    def _build_llm_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs dict for vllm.LLM() constructor.

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs for vllm.LLM().
        """
        kwargs: dict[str, Any] = {
            "model": config.model,
            "dtype": self._map_precision(config.precision),
            "trust_remote_code": True,
            "seed": config.random_seed,
        }

        # Apply VLLMEngineConfig fields if provided — only set non-None values
        vllm_cfg = config.vllm
        if vllm_cfg is not None and vllm_cfg.engine is not None:
            engine = vllm_cfg.engine

            # Helper: only set kwargs that are explicitly configured (None = use vLLM default)
            def _set(k: str, v: Any) -> None:
                if v is not None:
                    kwargs[k] = v

            _set("gpu_memory_utilization", engine.gpu_memory_utilization)
            _set("swap_space", engine.swap_space)
            _set("cpu_offload_gb", engine.cpu_offload_gb)
            _set("block_size", engine.block_size)
            _set("kv_cache_dtype", engine.kv_cache_dtype)
            _set("enforce_eager", engine.enforce_eager)
            _set("enable_chunked_prefill", engine.enable_chunked_prefill)
            _set("max_num_seqs", engine.max_num_seqs)
            _set("max_num_batched_tokens", engine.max_num_batched_tokens)
            _set("max_model_len", engine.max_model_len)
            _set("tensor_parallel_size", engine.tensor_parallel_size)
            _set("pipeline_parallel_size", engine.pipeline_parallel_size)
            _set("enable_prefix_caching", engine.enable_prefix_caching)
            _set("quantization", engine.quantization)

            if engine.speculative_model is not None:
                # vLLM v0.6+ uses speculative_config dict, not direct speculative_model kwarg
                kwargs["speculative_config"] = {
                    "model": engine.speculative_model,
                    "num_speculative_tokens": engine.num_speculative_tokens,
                }

            # New Phase 19.2 engine fields
            _set("disable_custom_all_reduce", engine.disable_custom_all_reduce)
            _set("kv_cache_memory_bytes", engine.kv_cache_memory_bytes)
            _set("offload_group_size", engine.offload_group_size)
            _set("offload_num_in_group", engine.offload_num_in_group)
            _set("offload_prefetch_step", engine.offload_prefetch_step)
            _set("compilation_config", engine.compilation_config)

            # offload_params: list[str] -> set[str] at wiring time (vLLM expects set[str])
            if engine.offload_params is not None:
                kwargs["offload_params"] = set(engine.offload_params)

            # Attention config: wire individual fields as flat LLM() kwargs
            if engine.attention is not None:
                attn = engine.attention
                # attention.backend -> attention_backend (vLLM kwarg name)
                if attn.backend is not None:
                    kwargs["attention_backend"] = attn.backend
                _set("use_prefill_decode_attention", attn.use_prefill_decode_attention)
                _set("use_prefill_query_quantization", attn.use_prefill_query_quantization)
                _set("use_cudnn_prefill", attn.use_cudnn_prefill)
                _set("disable_flashinfer_prefill", attn.disable_flashinfer_prefill)
                _set("disable_flashinfer_q_quantization", attn.disable_flashinfer_q_quantization)
                _set("use_trtllm_attention", attn.use_trtllm_attention)
                _set("use_trtllm_ragged_deepseek_prefill", attn.use_trtllm_ragged_deepseek_prefill)
                # Passthrough extras from VLLMAttentionConfig
                if attn.model_extra:
                    kwargs.update(attn.model_extra)

            # Passthrough extras from VLLMEngineConfig (LAST — user extras can override)
            if engine.model_extra:
                kwargs.update(engine.model_extra)

        return kwargs

    @staticmethod
    def _map_precision(precision: str) -> str:
        """Map precision string to vLLM dtype string.

        Args:
            precision: One of 'fp32', 'fp16', 'bf16'.

        Returns:
            vLLM dtype string ('float32', 'float16', 'bfloat16', or 'auto').
        """
        return {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}.get(precision, "auto")

    @staticmethod
    def _build_sampling_params(config: ExperimentConfig, sampling_params_cls: Any) -> Any:
        """Build vllm.SamplingParams from DecoderConfig.

        Maps our decoder config to vLLM's SamplingParams. Key differences:
        - top_k=0 (our disabled sentinel) → top_k=-1 (vLLM's disabled sentinel)
        - Greedy: temperature=0.0 disables sampling

        Args:
            config: Experiment configuration.
            sampling_params_cls: The SamplingParams class (lazy-imported by caller).

        Returns:
            SamplingParams instance.
        """
        # Beam search path: use BeamSearchParams instead of SamplingParams
        vllm_cfg = config.vllm
        if vllm_cfg is not None and vllm_cfg.beam_search is not None:
            return VLLMBackend._build_beam_search_params(config, vllm_cfg.beam_search)

        decoder = config.decoder

        # Greedy decode: temperature == 0.0 or do_sample is False
        is_greedy = decoder.temperature == 0.0 or not decoder.do_sample

        if is_greedy:
            kwargs: dict[str, Any] = {
                "temperature": 0.0,
                "max_tokens": config.max_output_tokens,
            }
        else:
            # Map top_k: our 0 sentinel → vLLM's -1 (disabled)
            top_k = decoder.top_k if decoder.top_k != 0 else -1

            kwargs = {
                "temperature": decoder.temperature,
                "top_p": decoder.top_p,
                "top_k": top_k,
                "repetition_penalty": decoder.repetition_penalty,
                "max_tokens": config.max_output_tokens,
            }

            if decoder.min_p is not None:
                kwargs["min_p"] = decoder.min_p

        # Apply vLLM-specific sampling overrides (VLLMSamplingConfig)
        # These override universal params (e.g. max_tokens overrides config.max_output_tokens)
        vllm_cfg = config.vllm
        if vllm_cfg is not None and vllm_cfg.sampling is not None:
            sampling = vllm_cfg.sampling
            if sampling.max_tokens is not None:
                kwargs["max_tokens"] = (
                    sampling.max_tokens
                )  # Override ExperimentConfig.max_output_tokens
            if sampling.min_tokens is not None:
                kwargs["min_tokens"] = sampling.min_tokens
            if sampling.presence_penalty is not None:
                kwargs["presence_penalty"] = sampling.presence_penalty
            if sampling.frequency_penalty is not None:
                kwargs["frequency_penalty"] = sampling.frequency_penalty
            if sampling.ignore_eos is not None:
                kwargs["ignore_eos"] = sampling.ignore_eos
            if sampling.n is not None:
                kwargs["n"] = sampling.n
            # Passthrough extras from VLLMSamplingConfig
            if sampling.model_extra:
                kwargs.update(sampling.model_extra)

        return sampling_params_cls(**kwargs)

    @staticmethod
    def _build_beam_search_params(config: ExperimentConfig, beam_cfg: Any) -> Any:
        """Build vllm.BeamSearchParams from VLLMBeamSearchConfig.

        Args:
            config: Experiment configuration (for max_output_tokens fallback).
            beam_cfg: VLLMBeamSearchConfig instance.

        Returns:
            BeamSearchParams instance.
        """
        from vllm import BeamSearchParams

        kwargs: dict[str, Any] = {}
        if beam_cfg.beam_width is not None:
            kwargs["beam_width"] = beam_cfg.beam_width
        if beam_cfg.length_penalty is not None:
            kwargs["length_penalty"] = beam_cfg.length_penalty
        if beam_cfg.early_stopping is not None:
            kwargs["early_stopping"] = beam_cfg.early_stopping
        # max_tokens: beam_search overrides config.max_output_tokens if set
        kwargs["max_tokens"] = beam_cfg.max_tokens or config.max_output_tokens
        # Passthrough extras from VLLMBeamSearchConfig
        if beam_cfg.model_extra:
            kwargs.update(beam_cfg.model_extra)
        return BeamSearchParams(**kwargs)

    # -------------------------------------------------------------------------
    # Prompt preparation
    # -------------------------------------------------------------------------

    def _prepare_prompts(self, config: ExperimentConfig) -> list[str]:
        """Prepare prompts for inference using the dataset loader.

        Delegates to the datasets module which handles built-in datasets
        (aienergyscore), user-supplied JSONL files, and synthetic prompts.

        Args:
            config: Experiment configuration (uses config.dataset, config.n).

        Returns:
            List of config.n prompt strings.
        """
        from llenergymeasure.datasets.loader import load_prompts

        prompts = load_prompts(config)
        logger.debug("Prepared %d prompts via dataset loader", len(prompts))
        return prompts

    # -------------------------------------------------------------------------
    # Measurement
    # -------------------------------------------------------------------------

    def _run_measurement(
        self,
        llm: Any,
        sampling_params: Any,
        config: ExperimentConfig,
        prompts: list[str],
    ) -> tuple[_VLLMMeasurementData, ThermalThrottleInfo, list[Any]]:
        """Run offline batch inference over all prompts.

        Single llm.generate() call with ALL prompts — no streaming, no
        one-at-a-time loops. Wrapped with PowerThermalSampler for timeseries
        and thermal throttle detection.

        Args:
            llm: vllm.LLM instance.
            sampling_params: vllm.SamplingParams instance.
            config: Experiment configuration.
            prompts: Full list of prompts.

        Returns:
            Tuple of (_VLLMMeasurementData, ThermalThrottleInfo, samples).

        Raises:
            BackendError: On OOM or other inference failures.
        """
        from llenergymeasure.core.power_thermal import PowerThermalSampler

        data = _VLLMMeasurementData()

        logger.info(
            "Starting vLLM offline batch inference: %d prompts, max_tokens=%d",
            len(prompts),
            config.max_output_tokens,
        )

        try:
            with PowerThermalSampler(device_index=0) as sampler:
                t0 = time.perf_counter()
                # BeamSearchParams uses llm.beam_search(), SamplingParams uses llm.generate()
                from vllm import BeamSearchParams as _BSP

                if isinstance(sampling_params, _BSP):
                    outputs = llm.beam_search(prompts, sampling_params)
                else:
                    outputs = llm.generate(prompts, sampling_params)
                elapsed = time.perf_counter() - t0
                data.total_time_sec = elapsed
                data.batch_times.append(elapsed)

        except Exception as e:
            if "out of memory" in str(e).lower():
                raise BackendError(
                    f"vLLM CUDA out of memory. Try: reduce n, "
                    f"use gpu_memory_utilization=0.8, or use a smaller model. "
                    f"Original error: {e}"
                ) from e
            raise BackendError(f"vLLM inference failed: {e}") from e

        thermal_info = sampler.get_thermal_throttle_info()
        timeseries_samples = sampler.get_samples()

        # Count tokens from RequestOutput objects
        # input_tokens: prompt_token_ids length per output
        # output_tokens: generated token_ids length in first completion
        input_token_count = sum(len(o.prompt_token_ids) for o in outputs)
        output_token_count = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)

        data.input_tokens = input_token_count
        data.output_tokens = output_token_count
        data.total_tokens = input_token_count + output_token_count

        logger.info(
            "vLLM inference complete: %d total tokens (in=%d, out=%d) in %.2fs",
            data.total_tokens,
            data.input_tokens,
            data.output_tokens,
            data.total_time_sec,
        )
        return data, thermal_info, timeseries_samples

    # -------------------------------------------------------------------------
    # Post-measurement helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _cuda_sync() -> None:
        """Synchronize CUDA before/after inference (CM-15 — Zeus best practice).

        Best-effort — failures are non-fatal and silently ignored.
        """
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass  # Non-fatal — best effort sync

    @staticmethod
    def _check_persistence_mode() -> bool:
        """Check whether GPU persistence mode is enabled.

        Returns:
            True if persistence mode is on (or unknown), False if definitively off.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
                return bool(mode != pynvml.NVML_FEATURE_DISABLED)
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            return True  # Unknown — don't generate spurious warning

    def _collect_warnings(
        self,
        duration_sec: float,
        timeseries_samples: list[Any],
    ) -> list[str]:
        """Collect measurement quality warnings.

        Args:
            duration_sec: Measurement window duration.
            timeseries_samples: Raw PowerThermalSamples for temp + sample count.

        Returns:
            List of warning strings.
        """
        from llenergymeasure.core.measurement_warnings import collect_measurement_warnings

        # Extract start/end temperatures from samples
        temp_start: float | None = None
        temp_end: float | None = None
        if timeseries_samples:
            temps = [s.temperature_c for s in timeseries_samples if s.temperature_c is not None]
            if temps:
                temp_start = temps[0]
                temp_end = temps[-1]

        persistence_on = self._check_persistence_mode()
        nvml_count = len(timeseries_samples)

        return collect_measurement_warnings(
            duration_sec=duration_sec,
            gpu_persistence_mode=persistence_on,
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            nvml_sample_count=nvml_count,
        )

    # -------------------------------------------------------------------------
    # Result assembly
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        config: ExperimentConfig,
        data: _VLLMMeasurementData,
        snapshot: Any,
        start_time: datetime,
        end_time: datetime,
        thermal_info: ThermalThrottleInfo,
        energy_measurement: Any,
        baseline: Any,
        flops_result: Any,
        warmup_result: WarmupResult,
        timeseries_path: str | None,
        measurement_warnings: list[str],
    ) -> ExperimentResult:
        """Assemble the ExperimentResult from measurement data.

        Args:
            config: Experiment configuration.
            data: Raw measurement data from the measurement loop.
            snapshot: EnvironmentSnapshot captured before model load.
            start_time: Measurement start time.
            end_time: Measurement end time.
            thermal_info: ThermalThrottleInfo from PowerThermalSampler.
            energy_measurement: EnergyMeasurement from energy backend, or None.
            baseline: BaselineCache from baseline measurement, or None.
            flops_result: FlopsResult from estimate_flops_palm(), or None.
            warmup_result: WarmupResult (not stored in ExperimentResult v2.0).
            timeseries_path: Relative path to Parquet sidecar, or None.
            measurement_warnings: List of quality warning strings.

        Returns:
            Fully assembled ExperimentResult with backend='vllm'.
        """
        from llenergymeasure.core.baseline import create_energy_breakdown

        experiment_id = f"{config.model}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        avg_tokens_per_second = (
            data.total_tokens / data.total_time_sec if data.total_time_sec > 0 else 0.0
        )

        # Real energy values from measurement backend (CM-18, CM-19)
        total_energy_j = energy_measurement.total_j if energy_measurement is not None else 0.0
        duration_sec = (end_time - start_time).total_seconds()

        # Energy per token (CM-25): output tokens only (input tokens are not "generated")
        output_tokens = data.output_tokens if data.output_tokens > 0 else data.total_tokens
        avg_energy_per_token_j = (
            total_energy_j / output_tokens if (total_energy_j > 0 and output_tokens > 0) else 0.0
        )

        # Energy breakdown with baseline adjustment
        energy_breakdown = create_energy_breakdown(total_energy_j, baseline, duration_sec)

        # FLOPs from PaLM formula (0.0 if estimation failed for vLLM)
        total_flops = flops_result.value if flops_result is not None else 0.0

        return ExperimentResult(
            experiment_id=experiment_id,
            measurement_config_hash=compute_measurement_config_hash(config),
            measurement_methodology="total",
            backend="vllm",
            aggregation=AggregationMetadata(
                method="single_process",
                num_processes=1,
            ),
            total_tokens=data.total_tokens,
            total_energy_j=total_energy_j,
            total_inference_time_sec=data.total_time_sec,
            avg_tokens_per_second=avg_tokens_per_second,
            avg_energy_per_token_j=avg_energy_per_token_j,
            total_flops=total_flops,
            process_results=[],
            start_time=start_time,
            end_time=end_time,
            environment_snapshot=snapshot,
            thermal_throttle=thermal_info,
            energy_breakdown=energy_breakdown,
            timeseries=timeseries_path,
            effective_config=config.model_dump(),
            baseline_power_w=energy_breakdown.baseline_power_w if energy_breakdown else None,
            energy_adjusted_j=energy_breakdown.adjusted_j if energy_breakdown else None,
            measurement_warnings=measurement_warnings,
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def _cleanup(self, llm: Any) -> None:
        """Release vLLM model from memory and clear CUDA cache.

        Args:
            llm: vllm.LLM instance to delete.
        """
        import importlib.util

        del llm
        gc.collect()
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
            except Exception:
                logger.debug("CUDA cleanup failed", exc_info=True)
        logger.info("vLLM model cleanup complete")
