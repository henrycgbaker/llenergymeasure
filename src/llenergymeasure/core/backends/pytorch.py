"""PyTorch/Transformers inference backend (v2.0 rewrite).

This is a clean rewrite — not an adaptation of the v1.x inference_backends/pytorch.py.
The P0 model_kwargs bug is fixed structurally: there is no intermediate loader class.
AutoModelForCausalLM.from_pretrained() is called directly with all kwargs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

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
class _MeasurementData:
    """Internal container for measurement loop results."""

    total_tokens: int = 0
    total_time_sec: float = 0.0
    peak_memory_mb: float = 0.0
    batch_times: list[float] = field(default_factory=list)
    # Separate input/output token counts for PaLM FLOPs calculation (CM-28)
    # Warmup tokens are excluded — only measurement window tokens are counted here.
    input_tokens: int = 0
    output_tokens: int = 0
    model_memory_mb: float = 0.0


class PyTorchBackend:
    """PyTorch/Transformers inference backend.

    Owns the full experiment lifecycle:
    1. Environment snapshot (before model load)
    2. Baseline power measurement (before model load — CM-17)
    3. Model + tokenizer loading
    4. Prompt preparation
    5. Warmup iterations + thermal floor wait
    6. Energy tracking start
    7. Measurement loop
    8. CUDA sync + energy tracking stop
    9. FLOPs estimation
    10. Timeseries write
    11. Measurement warnings
    12. Result assembly
    13. VRAM cleanup

    The P0 model_kwargs bug is fixed structurally: _model_load_kwargs()
    builds the full kwargs dict and _load_model() passes it directly to
    from_pretrained(). No intermediate loader class exists.
    """

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "pytorch"

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete PyTorch inference experiment.

        Args:
            config: Fully resolved experiment configuration.

        Returns:
            ExperimentResult with all measurement fields populated.

        Raises:
            BackendError: If model loading or inference fails.
        """
        # 1. Environment snapshot (BEFORE model loading — CM-32)
        from llenergymeasure.domain.environment import (
            collect_environment_snapshot,
        )

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

        # 3. Load model + tokenizer
        model, tokenizer = self._load_model(config)

        # Capture model memory baseline immediately after model load (weights + framework overhead).
        # Must happen BEFORE warmup, which allocates KV cache and raises max_memory_allocated.
        import torch as _torch

        model_memory_mb = 0.0
        if _torch.cuda.is_available():
            model_memory_mb = _torch.cuda.max_memory_allocated() / (1024 * 1024)
        logger.info("Model memory baseline (weights): %.1f MB", model_memory_mb)

        try:
            # 4. Prepare prompts
            prompts = self._prepare_prompts(config, tokenizer)

            # 5. Warmup (CM-21, CM-24) — returns WarmupResult
            warmup_result = self._run_warmup(model, tokenizer, config, prompts)

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
            from llenergymeasure.core.energy_backends import (
                select_energy_backend,
            )

            energy_backend = select_energy_backend(config.energy.backend)

            # 8. Start energy tracking (after warmup + thermal floor)
            energy_tracker = None
            if energy_backend is not None:
                energy_tracker = energy_backend.start_tracking()

            # 9-10. Measurement loop + thermal sampler (for timeseries + thermal info)
            start_time = datetime.now()
            result_data, thermal_info, timeseries_samples = self._run_measurement(
                model, tokenizer, config, prompts
            )

            # 11. CUDA sync before stopping energy (CM-15)
            self._cuda_sync()

            # 12. Stop energy tracking
            energy_measurement = None
            if energy_backend is not None and energy_tracker is not None:
                energy_measurement = energy_backend.stop_tracking(energy_tracker)
            end_time = datetime.now()

            # 13. FLOPs estimation (CM-26, CM-28 — warmup tokens excluded)
            from llenergymeasure.core.flops import estimate_flops_palm

            flops_result = estimate_flops_palm(
                model=model,
                n_input_tokens=result_data.input_tokens,
                n_output_tokens=result_data.output_tokens,
            )

        finally:
            # Always release model from memory
            self._cleanup(model)

        # 14. Write timeseries Parquet sidecar (if output_dir set — CM-16)
        timeseries_path: str | None = None
        if config.output_dir is not None and timeseries_samples:
            from llenergymeasure.core.timeseries import write_timeseries_parquet

            ts_file = write_timeseries_parquet(
                timeseries_samples,
                Path(config.output_dir) / "timeseries.parquet",
                gpu_index=0,
            )
            timeseries_path = ts_file.name  # relative name in result JSON

        # 15. Collect measurement quality warnings (CM-25 implied)
        measurement_warnings = self._collect_warnings(
            duration_sec=(end_time - start_time).total_seconds(),
            timeseries_samples=timeseries_samples,
        )

        # 16. Build ExperimentResult with real values
        return self._build_result(
            config=config,
            data=result_data,
            model_memory_mb=model_memory_mb,
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
    # Model loading (P0 fix)
    # -------------------------------------------------------------------------

    def _load_model(self, config: ExperimentConfig):
        """Load model and tokenizer directly — no intermediate loader class.

        The P0 fix: kwargs are built by _model_load_kwargs() and passed
        directly to from_pretrained(). The v1.x bug was that _build_model_kwargs()
        built the dict but loader.load(config) ignored it.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = self._model_load_kwargs(config)
        logger.info("Loading model %r with kwargs: %s", config.model, list(kwargs.keys()))

        # trust_remote_code for tokenizer — respects config, defaults True
        trust = True
        if config.pytorch is not None and config.pytorch.trust_remote_code is not None:
            trust = config.pytorch.trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=trust)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        model.eval()

        # Apply torch.compile post-load (must be AFTER from_pretrained + eval)
        if config.pytorch is not None and config.pytorch.torch_compile:
            import torch as _torch

            mode = config.pytorch.torch_compile_mode or "default"
            backend = config.pytorch.torch_compile_backend or "inductor"
            try:
                model = _torch.compile(model, mode=mode, backend=backend)
                logger.info("torch.compile applied (mode=%s, backend=%s)", mode, backend)
            except Exception as e:
                logger.warning("torch.compile failed (non-fatal, continuing without): %s", e)

        logger.info("Model loaded successfully")
        return model, tokenizer

    def _model_load_kwargs(self, config: ExperimentConfig) -> dict:
        """Build the full kwargs dict for AutoModelForCausalLM.from_pretrained().

        This is the P0 fix location: passthrough_kwargs and pytorch config
        options are ALL collected here and ALL passed to from_pretrained().

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs ready for from_pretrained().
        """
        kwargs: dict = {
            "torch_dtype": self._precision_to_dtype(config.precision),
        }

        pt = config.pytorch

        # Device placement — default "auto" unless researcher overrides
        if pt is not None and pt.device_map is not None:
            kwargs["device_map"] = pt.device_map
        else:
            kwargs["device_map"] = "auto"

        # Trust remote code — default True unless researcher overrides
        if pt is not None and pt.trust_remote_code is not None:
            kwargs["trust_remote_code"] = pt.trust_remote_code
        else:
            kwargs["trust_remote_code"] = True

        # Apply PyTorch-specific config options
        if pt is not None:
            if pt.attn_implementation is not None:
                kwargs["attn_implementation"] = pt.attn_implementation

            # BitsAndBytes quantization — use BitsAndBytesConfig, not raw kwargs
            if pt.load_in_4bit or pt.load_in_8bit:
                from transformers import BitsAndBytesConfig

                bnb_kwargs: dict = {}
                if pt.load_in_4bit:
                    bnb_kwargs["load_in_4bit"] = True
                    if pt.bnb_4bit_compute_dtype is not None:
                        import torch as _torch

                        _dtype_map = {
                            "float16": _torch.float16,
                            "bfloat16": _torch.bfloat16,
                            "float32": _torch.float32,
                        }
                        bnb_kwargs["bnb_4bit_compute_dtype"] = _dtype_map[pt.bnb_4bit_compute_dtype]
                    if pt.bnb_4bit_quant_type is not None:
                        bnb_kwargs["bnb_4bit_quant_type"] = pt.bnb_4bit_quant_type
                    if pt.bnb_4bit_use_double_quant is not None:
                        bnb_kwargs["bnb_4bit_use_double_quant"] = pt.bnb_4bit_use_double_quant
                if pt.load_in_8bit:
                    bnb_kwargs["load_in_8bit"] = True
                kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)

            # Additional from_pretrained() fields
            if pt.revision is not None:
                kwargs["revision"] = pt.revision
            if pt.max_memory is not None:
                kwargs["max_memory"] = pt.max_memory

        # PyTorch extra="allow" passthrough: forward unknown fields to from_pretrained()
        if pt is not None and pt.model_extra:
            kwargs.update(pt.model_extra)

        # passthrough_kwargs merged LAST so researcher can override any default
        # This is the core of the P0 fix: these kwargs are now actually used
        if config.passthrough_kwargs:
            kwargs.update(config.passthrough_kwargs)

        return kwargs

    @staticmethod
    def _precision_to_dtype(precision: str):
        """Map precision string to torch dtype.

        Args:
            precision: One of 'fp32', 'fp16', 'bf16'.

        Returns:
            Corresponding torch dtype.
        """
        import torch

        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]

    # -------------------------------------------------------------------------
    # Prompt preparation
    # -------------------------------------------------------------------------

    def _prepare_prompts(self, config: ExperimentConfig, tokenizer) -> list[str]:
        """Prepare prompts for inference.

        M1 placeholder: generates simple synthetic prompts. Dataset loading
        (aienergyscore.jsonl + SyntheticDatasetConfig) is a deferred M1 item.

        Args:
            config: Experiment configuration (uses config.n for count).
            tokenizer: Loaded tokenizer (unused in M1 placeholder).

        Returns:
            List of config.n prompt strings.
        """
        # M1 placeholder — generates simple prompts of roughly max_input_tokens length
        # A single token is ~4 chars; we use "Hello, " repeated as a simple approximation
        words_per_prompt = max(1, config.max_input_tokens // 4)
        base_prompt = ("Hello, " * words_per_prompt).strip()
        prompts = [base_prompt] * config.n
        logger.debug("Prepared %d prompts (M1 placeholder)", config.n)
        return prompts

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

    def _run_warmup(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        prompts: list[str],
    ) -> WarmupResult:
        """Run warmup using warmup_until_converged() and return WarmupResult.

        Args:
            model: Loaded model.
            tokenizer: Loaded tokenizer.
            config: Experiment configuration.
            prompts: Full prompt list (uses first prompt for warmup).

        Returns:
            WarmupResult with convergence status and iteration count.
        """
        from llenergymeasure.core.warmup import (
            create_warmup_inference_fn,
            warmup_until_converged,
        )

        if not config.warmup.enabled:
            logger.info("Warmup disabled, skipping")
            return WarmupResult(
                converged=True,
                final_cv=0.0,
                iterations_completed=0,
                target_cv=config.warmup.cv_threshold,
                max_prompts=config.warmup.max_prompts,
            )

        warmup_prompt = prompts[0] if prompts else "Hello, world"
        logger.info(
            "Running warmup: %d fixed iterations (convergence_detection=%s)",
            config.warmup.n_warmup,
            config.warmup.convergence_detection,
        )

        inference_fn = create_warmup_inference_fn(
            model, tokenizer, warmup_prompt, config.max_output_tokens
        )
        result = warmup_until_converged(inference_fn, config.warmup, show_progress=False)
        logger.info("Warmup complete: %d iterations", result.iterations_completed)
        return result

    # -------------------------------------------------------------------------
    # Measurement loop
    # -------------------------------------------------------------------------

    def _run_measurement(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        prompts: list[str],
    ) -> tuple[_MeasurementData, ThermalThrottleInfo, list]:
        """Run the measurement loop over all prompts.

        Prompts are processed in batches. Each batch is tokenized, run through
        model.generate(), and output token counts are accumulated. Also collects
        PowerThermalSamples for timeseries and thermal throttle detection.

        Args:
            model: Loaded model.
            tokenizer: Loaded tokenizer.
            config: Experiment configuration.
            prompts: Full list of prompts to measure.

        Returns:
            Tuple of (_MeasurementData, ThermalThrottleInfo, samples).

        Raises:
            BackendError: On CUDA OOM or other inference failures.
        """
        import torch

        from llenergymeasure.core.power_thermal import PowerThermalSampler

        batch_size = 1
        if config.pytorch is not None and config.pytorch.batch_size is not None:
            batch_size = config.pytorch.batch_size

        # Reset peak stats BEFORE the measurement loop so max_memory_allocated() below
        # captures inference-window-only peak (KV cache + activations + batch buffers),
        # NOT model weights already allocated by _load_model().
        # Must come before PowerThermalSampler so the reset window matches the measurement window.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        data = _MeasurementData()
        generate_kwargs = self._build_generate_kwargs(config)

        logger.info(
            "Starting measurement: %d prompts, batch_size=%d, max_new_tokens=%d",
            len(prompts),
            batch_size,
            config.max_output_tokens,
        )

        with PowerThermalSampler(device_index=0) as sampler:
            # Process in batches
            for batch_start in range(0, len(prompts), batch_size):
                batch = prompts[batch_start : batch_start + batch_size]
                try:
                    batch_input_tokens, batch_output_tokens, batch_time = self._run_batch(
                        model, tokenizer, config, batch, generate_kwargs
                    )
                    data.input_tokens += batch_input_tokens
                    data.output_tokens += batch_output_tokens
                    data.total_tokens += batch_input_tokens + batch_output_tokens
                    data.total_time_sec += batch_time
                    data.batch_times.append(batch_time)

                    logger.debug(
                        "Batch %d-%d: in=%d out=%d tokens in %.2fs",
                        batch_start,
                        batch_start + len(batch) - 1,
                        batch_input_tokens,
                        batch_output_tokens,
                        batch_time,
                    )
                except Exception as e:
                    if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
                        raise BackendError(
                            f"CUDA out of memory. Try: reduce batch_size, "
                            f"use precision=fp16, or use a smaller model. "
                            f"Original error: {e}"
                        ) from e
                    raise BackendError(f"Inference failed: {e}") from e

        thermal_info = sampler.get_thermal_throttle_info()
        timeseries_samples = sampler.get_samples()

        # Track peak GPU memory
        if torch.cuda.is_available():
            data.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        logger.info(
            "Measurement complete: %d total tokens in %.2fs",
            data.total_tokens,
            data.total_time_sec,
        )
        return data, thermal_info, timeseries_samples

    def _run_batch(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        batch: list[str],
        generate_kwargs: dict,
    ) -> tuple[int, int, float]:
        """Run a single batch through model.generate() and return (input_tokens, output_tokens, time_sec).

        Args:
            model: Loaded model.
            tokenizer: Loaded tokenizer.
            config: Experiment configuration.
            batch: List of prompt strings for this batch.
            generate_kwargs: Generation kwargs (decoder params).

        Returns:
            Tuple of (input_token_count, output_token_count, elapsed_seconds).
        """
        import torch

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_input_tokens,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_token_count = inputs["input_ids"].shape[1] * len(batch)

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_output_tokens,
                **generate_kwargs,
            )
        elapsed = time.perf_counter() - t0

        # Count only the newly generated tokens per sequence, then multiply by batch size
        tokens_per_seq = outputs.shape[1] - (inputs["input_ids"].shape[1])
        output_token_count = max(0, tokens_per_seq) * len(batch)
        return input_token_count, output_token_count, elapsed

    def _build_generate_kwargs(self, config: ExperimentConfig) -> dict:
        """Build generation kwargs from DecoderConfig and PyTorchConfig.

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs for model.generate().
        """
        decoder = config.decoder
        kwargs: dict = {
            "do_sample": decoder.do_sample,
            "temperature": decoder.temperature,
            "top_k": decoder.top_k,
            "top_p": decoder.top_p,
            "repetition_penalty": decoder.repetition_penalty,
        }

        # DecoderConfig new fields
        if decoder.min_p is not None:
            kwargs["min_p"] = decoder.min_p
        if decoder.min_new_tokens is not None:
            kwargs["min_new_tokens"] = decoder.min_new_tokens

        # PyTorchConfig generate() fields
        pt = config.pytorch
        if pt is not None:
            # KV cache
            if pt.use_cache is not None:
                kwargs["use_cache"] = pt.use_cache
            if pt.cache_implementation is not None:
                kwargs["cache_implementation"] = pt.cache_implementation

            # Beam search
            if pt.num_beams is not None:
                kwargs["num_beams"] = pt.num_beams
            if pt.early_stopping is not None:
                kwargs["early_stopping"] = pt.early_stopping
            if pt.length_penalty is not None:
                kwargs["length_penalty"] = pt.length_penalty

            # N-gram repetition
            if pt.no_repeat_ngram_size is not None:
                kwargs["no_repeat_ngram_size"] = pt.no_repeat_ngram_size

            # Speculative decoding (prompt lookup)
            if pt.prompt_lookup_num_tokens is not None:
                kwargs["prompt_lookup_num_tokens"] = pt.prompt_lookup_num_tokens

        # Greedy decoding: temperature=0 or do_sample=False — strip sampling params
        if decoder.temperature == 0.0 or not decoder.do_sample:
            kwargs["do_sample"] = False
            kwargs.pop("temperature", None)
            kwargs.pop("top_k", None)
            kwargs.pop("top_p", None)
            kwargs.pop("min_p", None)

        return kwargs

    # -------------------------------------------------------------------------
    # Post-measurement helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _cuda_sync() -> None:
        """Synchronize CUDA before stopping energy measurement (CM-15).

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
                return mode != pynvml.NVML_FEATURE_DISABLED
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            return True  # Unknown — don't generate spurious warning

    def _collect_warnings(
        self,
        duration_sec: float,
        timeseries_samples: list,
    ) -> list[str]:
        """Collect measurement quality warnings.

        Args:
            duration_sec: Measurement window duration.
            timeseries_samples: Raw PowerThermalSamples for temp + sample count.

        Returns:
            List of warning strings.
        """
        from llenergymeasure.core.measurement_warnings import (
            collect_measurement_warnings,
        )

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
        data: _MeasurementData,
        model_memory_mb: float,
        snapshot,
        start_time: datetime,
        end_time: datetime,
        thermal_info: ThermalThrottleInfo,
        energy_measurement,
        baseline,
        flops_result,
        warmup_result: WarmupResult,
        timeseries_path: str | None,
        measurement_warnings: list[str],
    ) -> ExperimentResult:
        """Assemble the ExperimentResult from measurement data.

        All energy/FLOPs fields are now populated with real values (no more 0.0
        placeholders). Energy breakdown uses baseline adjustment when available.

        Note: warmup_result is accepted but not stored on ExperimentResult — it
        lives on RawProcessResult. Kept in signature for completeness and future
        use when process_results are assembled.

        Args:
            config: Experiment configuration.
            data: Raw measurement data from the measurement loop.
            model_memory_mb: GPU memory after model load, before inference (MB).
            snapshot: EnvironmentSnapshot captured before model load.
            start_time: Measurement start time.
            end_time: Measurement end time.
            thermal_info: ThermalThrottleInfo from PowerThermalSampler.
            energy_measurement: EnergyMeasurement from energy backend, or None.
            baseline: BaselineCache from baseline measurement, or None.
            flops_result: FlopsResult from estimate_flops_palm().
            warmup_result: WarmupResult (not stored in ExperimentResult v2.0).
            timeseries_path: Relative path to Parquet sidecar, or None.
            measurement_warnings: List of quality warning strings.

        Returns:
            Fully assembled ExperimentResult.
        """
        from llenergymeasure.core.baseline import create_energy_breakdown
        from llenergymeasure.domain.metrics import (
            ExtendedEfficiencyMetrics,
            MemoryEfficiencyMetrics,
        )

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

        # FLOPs from PaLM formula
        total_flops = flops_result.value if flops_result is not None else 0.0

        # FLOPs derived fields (B2 fix -- no longer hardcoded to 0.0)
        flops_per_output_token = (
            total_flops / data.output_tokens
            if (total_flops > 0 and data.output_tokens > 0)
            else None
        )
        flops_per_input_token = (
            total_flops / data.input_tokens if (total_flops > 0 and data.input_tokens > 0) else None
        )
        flops_per_second = (
            total_flops / data.total_time_sec
            if (total_flops > 0 and data.total_time_sec > 0)
            else None
        )

        # Memory metrics: inference-window-only peak (reset before loop) and derived delta.
        # inference_memory_mb = peak (inference window) - model baseline (weights).
        inference_memory_mb = max(0.0, data.peak_memory_mb - model_memory_mb)
        logger.info(
            "Memory: model=%.1fMB, peak_inference=%.1fMB, inference_delta=%.1fMB",
            model_memory_mb,
            data.peak_memory_mb,
            inference_memory_mb,
        )
        extended_metrics = ExtendedEfficiencyMetrics(
            memory=MemoryEfficiencyMetrics(
                model_memory_mb=model_memory_mb,
                peak_memory_mb=data.peak_memory_mb,
                inference_memory_mb=inference_memory_mb,
            )
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            measurement_config_hash=compute_measurement_config_hash(config),
            measurement_methodology="total",
            backend="pytorch",
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
            flops_per_output_token=flops_per_output_token,
            flops_per_input_token=flops_per_input_token,
            flops_per_second=flops_per_second,
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
            extended_metrics=extended_metrics,
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def _cleanup(self, model) -> None:
        """Release model from memory and clear CUDA cache.

        Args:
            model: Model to delete.
        """
        import importlib.util

        del model
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
            except Exception:
                logger.debug("CUDA cleanup failed", exc_info=True)
        logger.info("Model cleanup complete")
