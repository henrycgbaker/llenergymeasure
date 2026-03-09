"""MeasurementHarness - owns the measurement lifecycle for any BackendPlugin.

The harness extracts the ~600 lines of identical measurement infrastructure
duplicated across pytorch.py and vllm.py into a single location.
Backends become thin plugins implementing the 4-method BackendPlugin protocol.
"""

from __future__ import annotations

import importlib.util
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.core.backends.protocol import BackendPlugin, InferenceOutput
from llenergymeasure.core.warmup import thermal_floor_wait
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    compute_measurement_config_hash,
)

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.baseline import BaselineCache
    from llenergymeasure.core.energy_backends import EnergyBackend
    from llenergymeasure.core.power_thermal import PowerThermalSample
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.metrics import FlopsResult

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


def _check_persistence_mode() -> bool:
    """Check whether GPU persistence mode is enabled.

    Returns:
        True if persistence mode is on (or unknown), False if definitively off.
    """
    try:
        import pynvml

        from llenergymeasure.core.gpu_info import nvml_context

        result: bool = True  # Default: unknown — don't generate spurious warning
        with nvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
            result = bool(mode != pynvml.NVML_FEATURE_DISABLED)
        return result
    except Exception:
        return True  # Unknown — don't generate spurious warning


# ---------------------------------------------------------------------------
# Top-level imports used in harness (lazy in backends, top-level here for
# patching in tests).  The actual heavy work is inside the backend plugins.
# ---------------------------------------------------------------------------


def collect_environment_snapshot() -> EnvironmentSnapshot:  # pragma: no cover
    from llenergymeasure.domain.environment import (
        collect_environment_snapshot as _snap,
    )

    return _snap()


def measure_baseline_power(duration_sec: float) -> BaselineCache | None:  # pragma: no cover
    from llenergymeasure.core.baseline import measure_baseline_power as _mbp

    return _mbp(duration_sec=duration_sec)


def select_energy_backend(explicit: str | None) -> EnergyBackend | None:  # pragma: no cover
    from llenergymeasure.core.energy_backends import select_energy_backend as _seb

    return _seb(explicit)


def estimate_flops_palm(
    model: Any, n_input_tokens: int, n_output_tokens: int
) -> FlopsResult:  # pragma: no cover
    from llenergymeasure.core.flops import estimate_flops_palm as _efp

    return _efp(model=model, n_input_tokens=n_input_tokens, n_output_tokens=n_output_tokens)


def write_timeseries_parquet(
    samples: list[PowerThermalSample], path: Path, gpu_index: int = 0
) -> Path:  # pragma: no cover
    from llenergymeasure.core.timeseries import write_timeseries_parquet as _wts

    return _wts(samples, path, gpu_index=gpu_index)


def collect_measurement_warnings(
    duration_sec: float,
    gpu_persistence_mode: bool,
    temp_start_c: float | None,
    temp_end_c: float | None,
    nvml_sample_count: int,
) -> list[str]:  # pragma: no cover
    from llenergymeasure.core.measurement_warnings import (
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
    """Thin re-export so tests can patch llenergymeasure.core.harness.PowerThermalSampler."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        from llenergymeasure.core.power_thermal import PowerThermalSampler as _PTS

        return _PTS(*args, **kwargs)


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

    def run(self, backend: BackendPlugin, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete measurement using the given backend plugin.

        Args:
            backend: BackendPlugin instance (pytorch, vllm, tensorrt, ...).
            config: Fully resolved experiment configuration.

        Returns:
            ExperimentResult with all measurement fields populated.

        Raises:
            BackendError: If model loading or inference fails.
            PreFlightError: If pre-flight checks fail before GPU allocation.
        """
        # 1. Environment snapshot (BEFORE model loading — CM-32)
        logger.info("Collecting environment snapshot before model load")
        snapshot = collect_environment_snapshot()

        # 2. Baseline power measurement (BEFORE model load — CM-17, CM-20)
        baseline = None
        if config.baseline.enabled:
            logger.info("Measuring baseline power (%.0fs)...", config.baseline.duration_seconds)
            baseline = measure_baseline_power(config.baseline.duration_seconds)

        # 3. Load model via backend plugin
        model = backend.load_model(config)

        # 4. Capture model memory baseline immediately after model load.
        # Must happen BEFORE warmup, which allocates KV cache.
        model_memory_mb = self._capture_model_memory_mb()

        try:
            # 5. Warmup (CM-21, CM-24) — returns WarmupResult
            warmup_result = backend.warmup(config, model)

            # 6. Thermal floor (CM-22) — harness sets thermal_floor_wait_s on warmup_result
            wait_s = thermal_floor_wait(config)
            warmup_result.thermal_floor_wait_s = wait_s

            # 7. Select energy backend (CM-14)
            energy_backend = select_energy_backend(config.energy.backend)

            # 8. Start energy tracking (after warmup + thermal floor)
            energy_tracker = None
            if energy_backend is not None:
                energy_tracker = energy_backend.start_tracking()

            # 9. CUDA sync BEFORE inference (CM-15 — Zeus best practice)
            _cuda_sync()

            # 10. Record start time and run inference
            start_time = datetime.now()

            # Start thermal sampler around inference for timeseries + throttle detection
            from llenergymeasure.core.power_thermal import PowerThermalSampler as _PTS

            with _PTS(device_index=0) as sampler:
                output = backend.run_inference(config, model)

            thermal_info = sampler.get_thermal_throttle_info()
            timeseries_samples = sampler.get_samples()

            # 11. CUDA sync AFTER inference, before stopping energy (CM-15)
            _cuda_sync()

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
        timeseries_path: str | None = None
        if config.output_dir is not None and timeseries_samples:
            ts_file = write_timeseries_parquet(
                timeseries_samples,
                Path(config.output_dir) / "timeseries.parquet",
                gpu_index=0,
            )
            timeseries_path = ts_file.name  # relative name in result JSON

        # 15. Collect measurement quality warnings (CM-25 implied)
        duration_sec = (end_time - start_time).total_seconds()
        measurement_warnings = self._collect_warnings(duration_sec, timeseries_samples)

        # 16. Assemble ExperimentResult
        return self._build_result(
            backend_name=backend.name,
            config=config,
            output=output,
            model_memory_mb=model_memory_mb,
            snapshot=snapshot,
            start_time=start_time,
            end_time=end_time,
            thermal_info=thermal_info,
            energy_measurement=energy_measurement,
            baseline=baseline,
            flops_result=flops_result,
            timeseries_path=timeseries_path,
            measurement_warnings=measurement_warnings,
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _capture_model_memory_mb(self) -> float:
        """Query torch for current GPU max_memory_allocated() after model load.

        Returns 0.0 if torch is unavailable or CUDA is not available.
        """
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
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

        Backend plugins that expose a model object via extras['hf_model'] will
        have FLOPs estimated using the PaLM formula. Otherwise returns None.
        """
        model_obj = output.extras.get("hf_model")
        if model_obj is None:
            # Try the backend's load_model result — backends may stash hf_model in extras.
            # For backends that don't expose it, FLOPs is 0.0 (acceptable).
            return None
        try:
            return estimate_flops_palm(
                model=model_obj,
                n_input_tokens=output.input_tokens,
                n_output_tokens=output.output_tokens,
            )
        except Exception as e:
            logger.debug("FLOPs estimation failed: %s", e)
            return None

    def _collect_warnings(
        self,
        duration_sec: float,
        timeseries_samples: list[PowerThermalSample],
    ) -> list[str]:
        """Collect measurement quality warnings from timeseries samples."""
        temp_start: float | None = None
        temp_end: float | None = None
        if timeseries_samples:
            temps = [s.temperature_c for s in timeseries_samples if s.temperature_c is not None]
            if temps:
                temp_start = temps[0]
                temp_end = temps[-1]

        persistence_on = _check_persistence_mode()
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
        thermal_info: Any,
        energy_measurement: Any,
        baseline: Any,
        flops_result: Any,
        timeseries_path: str | None,
        measurement_warnings: list[str],
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
            thermal_info: ThermalThrottleInfo from PowerThermalSampler.
            energy_measurement: EnergyMeasurement from energy backend, or None.
            baseline: BaselineCache from baseline measurement, or None.
            flops_result: FlopsResult from estimate_flops_palm(), or None.
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
            output.total_tokens / output.elapsed_time_sec if output.elapsed_time_sec > 0 else 0.0
        )

        # Real energy values from measurement backend (CM-18, CM-19)
        total_energy_j = energy_measurement.total_j if energy_measurement is not None else 0.0
        duration_sec = (end_time - start_time).total_seconds()

        # Energy per token (CM-25): output tokens only (input tokens are not "generated")
        output_tokens = output.output_tokens if output.output_tokens > 0 else output.total_tokens
        avg_energy_per_token_j = (
            total_energy_j / output_tokens if (total_energy_j > 0 and output_tokens > 0) else 0.0
        )

        # Energy breakdown with baseline adjustment
        energy_breakdown = create_energy_breakdown(total_energy_j, baseline, duration_sec)

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
            total_flops / output.elapsed_time_sec
            if (total_flops > 0 and output.elapsed_time_sec > 0)
            else None
        )

        # Memory metrics: inference-window-only peak and derived delta.
        inference_memory_mb = max(0.0, output.peak_memory_mb - model_memory_mb)
        logger.info(
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
            total_inference_time_sec=output.elapsed_time_sec,
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
