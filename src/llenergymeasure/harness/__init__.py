"""MeasurementHarness - owns the measurement lifecycle for any BackendPlugin.

The harness extracts the ~600 lines of identical measurement infrastructure
duplicated across pytorch.py and vllm.py into a single location.
Backends become thin plugins implementing the 4-method BackendPlugin protocol.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.backends.protocol import BackendPlugin, InferenceOutput
from llenergymeasure.datasets import load_prompts
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    compute_measurement_config_hash,
)
from llenergymeasure.energy import select_energy_sampler
from llenergymeasure.harness.warmup import thermal_floor_wait

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.metrics import FlopsResult
    from llenergymeasure.domain.progress import ProgressCallback
    from llenergymeasure.energy import EnergySampler
    from llenergymeasure.harness.baseline import BaselineCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (extracted from both backends — byte-identical copies)
# ---------------------------------------------------------------------------


def _cuda_sync() -> None:
    """Synchronise CUDA at measurement boundary (CM-15).

    Best-effort — failures are non-fatal and silently ignored.
    """
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass  # Non-fatal — best effort sync


def _check_persistence_mode(gpu_indices: list[int] | None = None) -> bool:
    """Check whether GPU persistence mode is enabled on all specified GPUs.

    Checks every GPU in gpu_indices (defaults to [0] when None). Returns False
    only if persistence mode is definitively off on at least one GPU.

    Args:
        gpu_indices: GPU device indices to check. Defaults to [0] when None.

    Returns:
        True if persistence mode is on (or unknown) for all GPUs,
        False if definitively off on any GPU.
    """
    indices = gpu_indices if gpu_indices is not None else [0]
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        with nvml_context():
            for idx in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
                if mode == pynvml.NVML_FEATURE_DISABLED:
                    return False
        return True
    except Exception:
        return True  # Unknown — don't generate spurious warning


# ---------------------------------------------------------------------------
# Top-level imports used in harness (lazy in backends, top-level here for
# patching in tests).  The actual heavy work is inside the backend plugins.
# ---------------------------------------------------------------------------


def collect_environment_snapshot() -> EnvironmentSnapshot:  # pragma: no cover
    from llenergymeasure.harness.environment import (
        collect_environment_snapshot as _snap,
    )

    return _snap()


def collect_environment_snapshot_async() -> Future[EnvironmentSnapshot]:  # pragma: no cover
    from llenergymeasure.harness.environment import (
        collect_environment_snapshot_async as _snap_async,
    )

    return _snap_async()


def measure_baseline_power(
    duration_sec: float,
    gpu_indices: list[int] | None = None,
) -> BaselineCache | None:  # pragma: no cover
    from llenergymeasure.harness.baseline import measure_baseline_power as _mbp

    return _mbp(duration_sec=duration_sec, gpu_indices=gpu_indices)


def select_energy_backend(
    explicit: str | None,
    gpu_indices: list[int] | None = None,
) -> EnergySampler | None:  # pragma: no cover

    return select_energy_sampler(explicit, gpu_indices=gpu_indices)


def estimate_flops_palm(
    model: Any, n_input_tokens: int, n_output_tokens: int
) -> FlopsResult:  # pragma: no cover
    from llenergymeasure.harness.flops import estimate_flops_palm as _efp

    return _efp(model=model, n_input_tokens=n_input_tokens, n_output_tokens=n_output_tokens)


def estimate_flops_palm_from_config(
    model_name: str, n_input_tokens: int, n_output_tokens: int
) -> FlopsResult | None:  # pragma: no cover
    from llenergymeasure.harness.flops import estimate_flops_palm_from_config as _efpc

    return _efpc(
        model_name=model_name,
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
    )


def write_timeseries_parquet(
    samples: list[PowerThermalSample], path: Path
) -> Path:  # pragma: no cover
    from llenergymeasure.harness.timeseries import write_timeseries_parquet as _wts

    return _wts(samples, path)


def collect_measurement_warnings(
    duration_sec: float,
    gpu_persistence_mode: bool,
    temp_start_c: float | None,
    temp_end_c: float | None,
    nvml_sample_count: int,
) -> list[str]:  # pragma: no cover
    from llenergymeasure.harness.measurement_warnings import (
        collect_measurement_warnings as _cmw,
    )

    return _cmw(
        duration_sec=duration_sec,
        gpu_persistence_mode=gpu_persistence_mode,
        temp_start_c=temp_start_c,
        temp_end_c=temp_end_c,
        nvml_sample_count=nvml_sample_count,
    )


class PowerThermalSampler:  # pragma: no cover
    """Thin re-export so tests can patch llenergymeasure.harness.PowerThermalSampler."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        from llenergymeasure.device.power_thermal import PowerThermalSampler as _PTS

        return _PTS(*args, **kwargs)

    def __enter__(self) -> Any: ...  # type: ignore[empty-body]

    def __exit__(self, *args: Any) -> None: ...  # type: ignore[empty-body]


# ---------------------------------------------------------------------------
# MeasurementHarness
# ---------------------------------------------------------------------------


class MeasurementHarness:
    """Orchestrates the measurement lifecycle for any BackendPlugin.

    Backends are thin plugins implementing BackendPlugin (load_model, warmup,
    run_inference, cleanup). The harness owns everything else:
    environment snapshot, baseline power, energy tracking, CUDA sync,
    thermal floor wait, FLOPs estimation, timeseries, warnings, and result assembly.
    """

    def run(
        self,
        backend: BackendPlugin,
        config: ExperimentConfig,
        snapshot: EnvironmentSnapshot | None = None,
        gpu_indices: list[int] | None = None,
        progress: ProgressCallback | None = None,
    ) -> ExperimentResult:
        """Run a complete measurement using the given backend plugin.

        Args:
            backend: BackendPlugin instance (pytorch, vllm, tensorrt, ...).
            config: Fully resolved experiment configuration.
            snapshot: Pre-collected environment snapshot (study-level cache).
                      When None, collected in a background thread during model load.
            gpu_indices: GPU device indices to monitor for energy/thermal measurement.
                         Defaults to [0] (single GPU, backward compatible) when None.
            progress: Optional callback for step-by-step progress reporting.
                      When None, no progress events are emitted (backward compatible).

        Returns:
            ExperimentResult with all measurement fields populated.

        Raises:
            BackendError: If model loading or inference fails.
            PreFlightError: If pre-flight checks fail before GPU allocation.
        """
        _p = progress  # short alias

        # 1. Environment snapshot — start background thread (BEFORE model loading — CM-32)
        snapshot_future: Future[EnvironmentSnapshot] | None = None
        if snapshot is None:
            logger.debug("Collecting environment snapshot (background thread)")
            snapshot_future = collect_environment_snapshot_async()

        # 2. Baseline power measurement (BEFORE model load — CM-17, CM-20)
        baseline = None
        if config.baseline.enabled:
            logger.debug("Measuring baseline power (%.0fs)...", config.baseline.duration_seconds)
            if _p:
                _p.on_step_start(
                    "baseline",
                    "Measuring baseline",
                    f"{config.baseline.duration_seconds:.0f}s idle power",
                )
                t0 = time.perf_counter()
            baseline = measure_baseline_power(
                config.baseline.duration_seconds, gpu_indices=gpu_indices
            )
            if _p:
                _p.on_step_done("baseline", time.perf_counter() - t0)

        # 3. Load model via backend plugin
        if _p:
            _p.on_step_start("model", "Loading model", config.model)
            t0 = time.perf_counter()
        model = backend.load_model(config)
        if _p:
            _p.on_step_done("model", time.perf_counter() - t0)

        # 3b. Join snapshot future — collection hidden behind model loading
        if snapshot_future is not None:
            snapshot = snapshot_future.result(timeout=10)

        # 4. Capture model memory baseline immediately after model load.
        # Must happen BEFORE warmup, which allocates KV cache.
        model_memory_mb = self._capture_model_memory_mb(gpu_indices=gpu_indices)

        # 4b. Load prompts — BEFORE measurement window (methodology fix)
        prompts = load_prompts(config)
        logger.debug("Loaded %d prompts via dataset loader", len(prompts))

        try:
            # 5. Warmup (CM-21, CM-24) — returns WarmupResult
            if _p:
                _p.on_step_start(
                    "warmup", "Warming up", f"up to {config.warmup.max_prompts} prompts"
                )
                t0_warmup = time.perf_counter()
            warmup_result = backend.warmup(config, model, prompts)
            if _p:
                cv_info = ""
                if warmup_result.final_cv > 0:
                    cv_info = f"  CV={warmup_result.final_cv:.1%}"
                _p.on_step_update(
                    "warmup",
                    f"{warmup_result.iterations_completed}/{config.warmup.max_prompts} prompts{cv_info}",
                )
                _p.on_step_done("warmup", time.perf_counter() - t0_warmup)

            # 6. Thermal floor (CM-22) — harness sets thermal_floor_wait_s on warmup_result
            wait_s = thermal_floor_wait(config)
            warmup_result.thermal_floor_wait_s = wait_s

            # 7. Select energy backend (CM-14)
            energy_backend = select_energy_backend(config.energy.backend, gpu_indices=gpu_indices)

            # 8. Start energy tracking (after warmup + thermal floor)
            energy_tracker = None
            if energy_backend is not None:
                energy_tracker = energy_backend.start_tracking()

            # 9. CUDA sync BEFORE inference (CM-15 — Zeus best practice)
            _cuda_sync()

            if _p:
                _p.on_step_start("measure", "Measuring", f"{config.n} prompts")
            t_inference_start = time.perf_counter()  # Canonical timer start (H1)
            # 10. Record start time and run inference
            start_time = datetime.now()

            # Start thermal sampler around inference for timeseries + throttle detection
            with PowerThermalSampler(gpu_indices=gpu_indices) as sampler:
                output = backend.run_inference(config, model, prompts)

            t_inference_end = time.perf_counter()  # Canonical timer end (H1)
            if _p:
                _p.on_step_done("measure", t_inference_end - t_inference_start)

            thermal_info = sampler.get_thermal_throttle_info()
            timeseries_samples = sampler.get_samples()

            # 11. CUDA sync AFTER inference, before stopping energy (CM-15)
            _cuda_sync()

            # Harness sets canonical inference timer (H1) — overrides backend's elapsed_time_sec
            output.inference_time_sec = t_inference_end - t_inference_start

            # 12. Stop energy tracking
            energy_measurement = None
            if energy_backend is not None and energy_tracker is not None:
                energy_measurement = energy_backend.stop_tracking(energy_tracker)
            end_time = datetime.now()

            # 13. FLOPs estimation (CM-26, CM-28 — warmup tokens excluded)
            flops_result = self._estimate_flops(backend, config, output)

        finally:
            # Always release model from memory even on exception
            backend.cleanup(model)

        # 14. Write timeseries Parquet sidecar (if output_dir set — CM-16)
        if _p:
            _p.on_step_start(
                "save",
                "Saving",
                str(config.output_dir) if config.output_dir else "results",
            )
            t0_save = time.perf_counter()

        timeseries_path: str | None = None
        if config.output_dir is not None and timeseries_samples:
            ts_file = write_timeseries_parquet(
                timeseries_samples,
                Path(config.output_dir) / "timeseries.parquet",
            )
            timeseries_path = ts_file.name  # relative name in result JSON

        # 15. Collect measurement quality warnings (CM-25 implied)
        duration_sec = (end_time - start_time).total_seconds()
        measurement_warnings = self._collect_warnings(duration_sec, timeseries_samples, gpu_indices)

        # 16. Assemble ExperimentResult
        result = self._build_result(
            backend_name=backend.name,
            config=config,
            output=output,
            model_memory_mb=model_memory_mb,
            snapshot=snapshot,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration_sec,
            thermal_info=thermal_info,
            energy_measurement=energy_measurement,
            baseline=baseline,
            flops_result=flops_result,
            timeseries_path=timeseries_path,
            measurement_warnings=measurement_warnings,
            warmup_result=warmup_result,
        )

        if _p:
            _p.on_step_done("save", time.perf_counter() - t0_save)

        return result

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _capture_model_memory_mb(self, gpu_indices: list[int] | None = None) -> float:
        """Query torch for GPU max_memory_allocated() after model load.

        Iterates over gpu_indices (defaults to [0] when None), queries
        max_memory_allocated(device=idx) per GPU, and returns the max across
        all ranks. This ensures tensor-parallel experiments report the peak
        across all participating GPUs (M18).

        Returns 0.0 if torch is unavailable or CUDA is not available.
        """
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    indices = gpu_indices if gpu_indices is not None else [0]
                    if not indices:
                        return 0.0
                    peak = max(torch.cuda.max_memory_allocated(device=idx) for idx in indices)
                    return float(peak / (1024 * 1024))
            except Exception:
                pass
        return 0.0

    def _estimate_flops(
        self,
        backend: BackendPlugin,
        config: ExperimentConfig,
        output: InferenceOutput,
    ) -> Any:
        """Estimate FLOPs from model and token counts.

        Fallback chain (M5):
        1. AutoConfig path — uses estimate_flops_palm_from_config(config.model).
           Works for ALL backends including vLLM and TensorRT-LLM (no model weights needed).
        2. hf_model path — uses estimate_flops_palm(hf_model) when extras['hf_model'] is set.
           Higher-confidence since it counts actual loaded parameters.
        3. None — FLOPs unavailable.
        """
        # Step 1: AutoConfig path (works without loaded model weights)
        try:
            result = estimate_flops_palm_from_config(
                model_name=config.model,
                n_input_tokens=output.input_tokens,
                n_output_tokens=output.output_tokens,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.debug("AutoConfig FLOPs estimation failed: %s", e)

        # Step 2: hf_model path (higher confidence — uses actual loaded parameters)
        model_obj = output.extras.get("hf_model")
        if model_obj is not None:
            try:
                return estimate_flops_palm(
                    model=model_obj,
                    n_input_tokens=output.input_tokens,
                    n_output_tokens=output.output_tokens,
                )
            except Exception as e:
                logger.debug("hf_model FLOPs estimation failed: %s", e)

        # Step 3: FLOPs unavailable
        return None

    def _collect_warnings(
        self,
        duration_sec: float,
        timeseries_samples: list[PowerThermalSample],
        gpu_indices: list[int] | None = None,
    ) -> list[str]:
        """Collect measurement quality warnings from timeseries samples."""
        temp_start: float | None = None
        temp_end: float | None = None
        if timeseries_samples:
            temps = [s.temperature_c for s in timeseries_samples if s.temperature_c is not None]
            if temps:
                temp_start = temps[0]
                temp_end = temps[-1]

        persistence_on = _check_persistence_mode(gpu_indices)
        nvml_count = len(timeseries_samples)

        return collect_measurement_warnings(
            duration_sec=duration_sec,
            gpu_persistence_mode=persistence_on,
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            nvml_sample_count=nvml_count,
        )

    def _build_result(
        self,
        backend_name: str,
        config: ExperimentConfig,
        output: InferenceOutput,
        model_memory_mb: float,
        snapshot: Any,
        start_time: datetime,
        end_time: datetime,
        duration_sec: float,
        thermal_info: Any,
        energy_measurement: Any,
        baseline: Any,
        flops_result: Any,
        timeseries_path: str | None,
        measurement_warnings: list[str],
        warmup_result: Any = None,
    ) -> ExperimentResult:
        """Assemble ExperimentResult from measurement data.

        All energy/FLOPs fields are populated with real values (no 0.0 placeholders).
        Energy breakdown uses baseline adjustment when available.

        Args:
            backend_name: Backend identifier string (from backend.name).
            config: Experiment configuration.
            output: InferenceOutput from backend.run_inference().
            model_memory_mb: GPU memory after model load, before inference (MB).
            snapshot: EnvironmentSnapshot captured before model load.
            start_time: Measurement start time.
            end_time: Measurement end time.
            duration_sec: Pre-computed (end_time - start_time).total_seconds().
            thermal_info: ThermalThrottleInfo from PowerThermalSampler.
            energy_measurement: EnergyMeasurement from energy backend, or None.
            baseline: BaselineCache from baseline measurement, or None.
            flops_result: FlopsResult from estimate_flops_palm(), or None.
            timeseries_path: Relative path to Parquet sidecar, or None.
            measurement_warnings: List of quality warning strings.
            warmup_result: WarmupResult from backend.warmup(), or None.

        Returns:
            Fully assembled ExperimentResult.
        """
        from llenergymeasure.domain.metrics import (
            ExtendedEfficiencyMetrics,
            MemoryEfficiencyMetrics,
            MultiGPUMetrics,
        )
        from llenergymeasure.harness.baseline import create_energy_breakdown

        experiment_id = f"{config.model}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        avg_tokens_per_second = (
            output.total_tokens / output.inference_time_sec
            if output.inference_time_sec > 0
            else 0.0
        )

        # Real energy values from measurement backend (CM-18, CM-19)
        total_energy_j = energy_measurement.total_j if energy_measurement is not None else 0.0
        # duration_sec is passed in from run() — computed once, not recalculated here

        # Energy per token (CM-25): output tokens only (input tokens are not "generated")
        output_tokens = output.output_tokens if output.output_tokens > 0 else output.total_tokens
        avg_energy_per_token_j = (
            total_energy_j / output_tokens if (total_energy_j > 0 and output_tokens > 0) else 0.0
        )

        # Energy breakdown with baseline adjustment.
        # H1: use energy backend's sampler window duration for baseline adjustment,
        # not harness datetime duration, to avoid CUDA sync latency skew.
        energy_duration = (
            energy_measurement.duration_sec if energy_measurement is not None else duration_sec
        )
        energy_breakdown = create_energy_breakdown(total_energy_j, baseline, energy_duration)

        # Per-GPU energy breakdown from EnergyMeasurement.per_gpu_j
        energy_per_device_j = None
        multi_gpu = None
        if energy_measurement is not None and energy_measurement.per_gpu_j:
            sorted_indices = sorted(energy_measurement.per_gpu_j.keys())
            energy_per_device_j = [energy_measurement.per_gpu_j[i] for i in sorted_indices]
            if len(sorted_indices) > 1:
                multi_gpu = MultiGPUMetrics(
                    num_gpus=len(sorted_indices),
                    energy_per_gpu_j=energy_per_device_j,
                    energy_total_j=total_energy_j,
                    energy_per_output_token_j=(
                        total_energy_j / output_tokens if output_tokens > 0 else 0.0
                    ),
                )

        # FLOPs from PaLM formula (0.0 if estimation unavailable)
        total_flops = flops_result.value if flops_result is not None else 0.0

        # FLOPs derived fields (B2 fix — no longer hardcoded to 0.0)
        flops_per_output_token = (
            total_flops / output.output_tokens
            if (total_flops > 0 and output.output_tokens > 0)
            else None
        )
        flops_per_input_token = (
            total_flops / output.input_tokens
            if (total_flops > 0 and output.input_tokens > 0)
            else None
        )
        flops_per_second = (
            total_flops / output.inference_time_sec
            if (total_flops > 0 and output.inference_time_sec > 0)
            else None
        )

        # Memory metrics: inference-window-only peak and derived delta.
        inference_memory_mb = max(0.0, output.peak_memory_mb - model_memory_mb)
        logger.debug(
            "Memory: model=%.1fMB, peak_inference=%.1fMB, inference_delta=%.1fMB",
            model_memory_mb,
            output.peak_memory_mb,
            inference_memory_mb,
        )
        extended_metrics = ExtendedEfficiencyMetrics(
            memory=MemoryEfficiencyMetrics(
                model_memory_mb=model_memory_mb,
                peak_memory_mb=output.peak_memory_mb,
                inference_memory_mb=inference_memory_mb,
            )
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            measurement_config_hash=compute_measurement_config_hash(config),
            measurement_methodology="total",
            backend=backend_name,
            aggregation=AggregationMetadata(
                method="single_process",
                num_processes=1,
            ),
            total_tokens=output.total_tokens,
            total_energy_j=total_energy_j,
            total_inference_time_sec=output.inference_time_sec,
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
            energy_per_device_j=energy_per_device_j,
            multi_gpu=multi_gpu,
            warmup_result=warmup_result,
            measurement_warnings=measurement_warnings,
            extended_metrics=extended_metrics,
        )
