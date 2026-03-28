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
    from llenergymeasure.energy import EnergySampler as EnergySampler
    from llenergymeasure.harness.baseline import BaselineCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (extracted from both backends — byte-identical copies)
# ---------------------------------------------------------------------------


def _cuda_sync() -> None:
    """Synchronise CUDA at measurement boundary.

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
        output_dir: Path | str | None = None,
        save_timeseries: bool = True,
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
            output_dir: Directory for timeseries parquet output. None = no disk writes.
                        Passed as runtime param by the study runner, not from config.
            save_timeseries: Whether to persist GPU timeseries to Parquet sidecar.
                             Controlled by OutputConfig.save_timeseries at study level.

        Returns:
            ExperimentResult with all measurement fields populated.

        Raises:
            BackendError: If model loading or inference fails.
            PreFlightError: If pre-flight checks fail before GPU allocation.
        """
        _p = progress  # short alias

        def _substep(step: str, text: str, elapsed: float = 0.0) -> None:
            """Emit a substep event to the progress callback."""
            if _p:
                _p.on_substep(step, text, elapsed)

        # 1. Environment snapshot — start background thread (before model loading)
        snapshot_future: Future[EnvironmentSnapshot] | None = None
        if snapshot is None:
            logger.debug("Collecting environment snapshot (background thread)")
            snapshot_future = collect_environment_snapshot_async()

        # 2. Baseline power measurement (before model load)
        baseline = None
        if config.baseline.enabled:
            dur = config.baseline.duration_seconds
            logger.debug("Measuring baseline power (%.0fs)...", dur)
            if _p:
                _p.on_step_start("baseline", "Measuring", f"baseline idle power ({dur:.0f}s)")
                t0 = time.perf_counter()
            baseline = measure_baseline_power(dur, gpu_indices=gpu_indices)
            if baseline is not None:
                _substep(
                    "baseline",
                    f"baseline: {baseline.power_w:.1f}W ({baseline.sample_count} samples)",
                )
            if _p:
                _p.on_step_done("baseline", time.perf_counter() - t0)
        elif _p:
            _p.on_step_skip("baseline", "disabled")

        # 3. Load model via backend plugin
        if _p:
            _p.on_step_start("model", "Loading", f"model {config.model}")
            t0 = time.perf_counter()

        # Build model substep callback
        def _on_model_substep(text: str, elapsed: float) -> None:
            _substep("model", text, elapsed)

        model = backend.load_model(config, on_substep=_on_model_substep)

        if _p:
            _p.on_step_done("model", time.perf_counter() - t0)

        # 3b. Join snapshot future — collection hidden behind model loading
        if snapshot_future is not None:
            snapshot = snapshot_future.result(timeout=10)

        # 4. Capture model memory baseline immediately after model load.
        # Must happen BEFORE warmup, which allocates KV cache.
        model_memory_mb = self._capture_model_memory_mb(gpu_indices=gpu_indices)
        if model_memory_mb > 0:
            _substep("model", f"model memory: {model_memory_mb:.0f}MB")

        # 4b. Load prompts — BEFORE measurement window (methodology fix)
        if _p:
            _p.on_step_start(
                "prompts",
                "Loading",
                f"prompts ({config.dataset.n_prompts} {config.dataset.source})",
            )
            t0_prompts = time.perf_counter()
        prompts = load_prompts(config)
        logger.debug("Loaded %d prompts via dataset loader", len(prompts))
        _substep("prompts", f"tokenised {len(prompts)} prompts")
        if _p:
            _p.on_step_done("prompts", time.perf_counter() - t0_prompts)

        try:
            # 5. Warmup
            if _p:
                _p.on_step_start(
                    "warmup", "Warming up", f"up to {config.warmup.max_prompts} prompts"
                )
                t0_warmup = time.perf_counter()

            warmup_result = backend.warmup(config, model, prompts)

            if _p:
                iters = warmup_result.iterations_completed
                cv_info = f"  CV={warmup_result.final_cv:.1%}" if warmup_result.final_cv > 0 else ""
                converged = "converged" if warmup_result.converged else "not converged"
                _p.on_step_update("warmup", f"{iters} iterations ({converged}{cv_info})")
                _p.on_step_done("warmup", time.perf_counter() - t0_warmup)
            _substep(
                "warmup",
                f"{warmup_result.iterations_completed} iterations"
                + (f"  CV={warmup_result.final_cv:.1%}" if warmup_result.final_cv > 0 else ""),
            )

            # 6. Thermal floor — show step before sleeping
            floor_secs = config.warmup.thermal_floor_seconds if config.warmup.enabled else 0
            if _p:
                if floor_secs > 0:
                    _p.on_step_start(
                        "thermal_floor", "Waiting", f"thermal floor ({floor_secs:.0f}s)"
                    )
                else:
                    _p.on_step_skip("thermal_floor", "wait=0s")

            wait_s = thermal_floor_wait(config)
            warmup_result.thermal_floor_wait_s = wait_s

            if _p and wait_s > 0:
                _p.on_step_done("thermal_floor", wait_s)

            # 7. Select energy sampler
            if _p:
                _p.on_step_start("energy_select", "Selecting", "energy sampler")
                t0_energy = time.perf_counter()
            energy_sampler = select_energy_sampler(config.energy_sampler, gpu_indices=gpu_indices)
            sampler_name = type(energy_sampler).__name__ if energy_sampler else "none"
            _substep("energy_select", f"selected: {sampler_name}")
            if _p:
                _p.on_step_update("energy_select", f"energy sampler ({sampler_name})")
                _p.on_step_done("energy_select", time.perf_counter() - t0_energy)

            # 8. Start energy tracking (after warmup + thermal floor)
            energy_tracker = None
            if energy_sampler is not None:
                energy_tracker = energy_sampler.start_tracking()

            # 9. CUDA sync before inference (Zeus best practice)
            _cuda_sync()
            _substep("measure", "CUDA sync (pre)")

            if _p:
                _p.on_step_start(
                    "measure", "Measuring", f"inference ({config.dataset.n_prompts} prompts)"
                )
            _substep("measure", "energy tracker started")

            t_inference_start = time.perf_counter()
            start_time = datetime.now()

            # Start thermal sampler around inference for timeseries + throttle detection
            with PowerThermalSampler(gpu_indices=gpu_indices) as thermal_sampler:
                output = backend.run_inference(config, model, prompts)

            t_inference_end = time.perf_counter()

            # 11. CUDA sync after inference, before stopping energy
            _cuda_sync()
            _substep("measure", "CUDA sync (post)")

            if _p:
                _p.on_step_done("measure", t_inference_end - t_inference_start)

            thermal_info = thermal_sampler.get_thermal_throttle_info()
            timeseries_samples = thermal_sampler.get_samples()

            # Harness sets canonical inference timer — overrides backend's elapsed_time_sec
            output.inference_time_sec = t_inference_end - t_inference_start

            # 12. Stop energy tracking
            energy_measurement = None
            if energy_sampler is not None and energy_tracker is not None:
                energy_measurement = energy_sampler.stop_tracking(energy_tracker)
            end_time = datetime.now()

            # 13. FLOPs estimation (warmup tokens excluded)
            if _p:
                _p.on_step_start("flops", "Estimating", "FLOPs (PaLM formula)")
                t0_flops = time.perf_counter()
            flops_result = self._estimate_flops(backend, config, output)
            if _p:
                if flops_result is not None:
                    _p.on_step_update("flops", f"FLOPs: {flops_result.value:.2e}")
                _p.on_step_done("flops", time.perf_counter() - t0_flops)
            if flops_result is not None:
                _substep("flops", f"FLOPs: {flops_result.value:.2e}")

        finally:
            # Always release model from memory even on exception
            backend.cleanup(model)

        # 14. Write timeseries Parquet sidecar (if output_dir set and save_timeseries enabled)
        resolved_output_dir = Path(output_dir) if output_dir is not None else None
        if _p:
            _p.on_step_start(
                "save",
                "Saving",
                f"results to {resolved_output_dir}" if resolved_output_dir else "results",
            )
            t0_save = time.perf_counter()

        timeseries_path: str | None = None
        if save_timeseries and resolved_output_dir is not None and timeseries_samples:
            ts_file = write_timeseries_parquet(
                timeseries_samples,
                resolved_output_dir / "timeseries.parquet",
            )
            timeseries_path = ts_file.name  # relative name in result JSON
            _substep("save", "timeseries parquet written")

        # 15. Collect measurement quality warnings
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
        _substep("save", "result assembled")

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
        across all participating GPUs.

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

        Fallback chain:
        1. AutoConfig path — uses estimate_flops_palm_from_config(config.model).
           Works for all backends (no model weights needed).
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
            energy_measurement: EnergyMeasurement from energy sampler, or None.
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

        # Real energy values from energy sampler
        total_energy_j = energy_measurement.total_j if energy_measurement is not None else 0.0
        # duration_sec is passed in from run() — computed once, not recalculated here

        # Energy per token: output tokens only (input tokens are not "generated")
        output_tokens = output.output_tokens if output.output_tokens > 0 else output.total_tokens
        avg_energy_per_token_j = (
            total_energy_j / output_tokens if (total_energy_j > 0 and output_tokens > 0) else 0.0
        )

        # Energy breakdown with baseline adjustment.
        # Use energy sampler's window duration for baseline adjustment,
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

        # FLOPs derived fields
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
