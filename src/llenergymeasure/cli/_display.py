"""Plain-text formatting utilities for CLI output.

All result output goes to stdout (scientific record).
Progress/header output goes to stderr (transient display area).
No Rich imports — plain print() and sys.stderr.write() only.
"""

from __future__ import annotations

import difflib
import math
import sys
import traceback

from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult
from llenergymeasure.exceptions import LLEMError


def _sig3(value: float) -> str:
    """Format a float to 3 significant figures.

    Examples:
        312.4  -> "312"
        3.12   -> "3.12"
        0.00312 -> "0.00312"
        847.0  -> "847"
        0      -> "0"
        1234   -> "1230"
    """
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    # Number of decimal places needed for 3 sig figs
    decimal_places = max(0, 2 - magnitude)
    rounded = round(value, decimal_places - int(magnitude >= 3) * 0)
    # Recompute for clarity: round to 3 sig figs
    factor = 10 ** (magnitude - 2)
    rounded = round(value / factor) * factor
    # Format without trailing zeros
    if decimal_places <= 0:
        return str(int(rounded))
    formatted = f"{rounded:.{decimal_places}f}"
    # Strip trailing zeros after decimal point
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Examples:
        4.2  -> "4.2s"
        272  -> "4m 32s"
        3900 -> "1h 05m"
    """
    if seconds < 60:
        # Show 1 decimal place for sub-minute durations
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}m"


def print_result_summary(result: ExperimentResult) -> None:
    """Print grouped result summary to stdout.

    Sections: Energy, Performance, Timing, Warnings.
    Strictly raw metrics only — no derived ratios.
    All numeric values formatted to 3 significant figures.
    """
    # Header
    print(f"Result: {result.experiment_id}")
    print()

    # --- Energy ---
    print("Energy")
    print(f"  Total          {_sig3(result.total_energy_j)} J")
    if result.baseline_power_w is not None:
        print(f"  Baseline       {_sig3(result.baseline_power_w)} W")
    if result.energy_adjusted_j is not None:
        print(f"  Adjusted       {_sig3(result.energy_adjusted_j)} J")
    print()

    # --- Performance ---
    print("Performance")
    print(f"  Throughput     {_sig3(result.avg_tokens_per_second)} tok/s")

    if result.total_flops > 0:
        flops_val = f"{result.total_flops:.2e}"
        # Try to get method/confidence from process_results
        method: str | None = None
        confidence: str | None = None
        if result.process_results:
            cm = result.process_results[0].compute_metrics
            method = cm.flops_method if cm.flops_method != "unknown" else None
            confidence = cm.flops_confidence if cm.flops_confidence != "unknown" else None
        if method and confidence:
            print(f"  FLOPs          {flops_val} ({method}, {confidence})")
        elif method:
            print(f"  FLOPs          {flops_val} ({method})")
        else:
            print(f"  FLOPs          {flops_val}")

    if result.latency_stats is not None:
        ls = result.latency_stats
        if hasattr(ls, "ttft_ms") and ls.ttft_ms is not None:
            print(f"  Latency TTFT   {_sig3(ls.ttft_ms)} ms")
        if hasattr(ls, "itl_ms") and ls.itl_ms is not None:
            print(f"  Latency ITL    {_sig3(ls.itl_ms)} ms")
    print()

    # --- Timing ---
    print("Timing")
    print(f"  Duration       {_format_duration(result.duration_sec)}")
    if result.warmup_excluded_samples is not None:
        print(f"  Warmup         {result.warmup_excluded_samples} prompts excluded")
    print()

    # --- Warnings ---
    if result.measurement_warnings:
        print("Warnings")
        for warning in result.measurement_warnings:
            print(f"  {warning}")
        print()


def print_dry_run(
    config: ExperimentConfig,
    vram: dict[str, float] | None,
    gpu_vram_gb: float | None,
    verbose: bool = False,
) -> None:
    """Print dry-run output to stdout.

    Shows resolved config and VRAM estimate.
    With verbose=True, adds source annotations.
    """
    # Determine non-default fields for annotations
    defaults = {
        "backend": "pytorch",
        "precision": "bf16",
        "n": 100,
        "dataset": "aienergyscore",
        "output_dir": None,
    }

    def _annotate(field: str, value: object) -> str:
        """Return annotation string if value differs from default."""
        if not verbose:
            return ""
        default = defaults.get(field)
        if value == default:
            return f" ({field} default)" if field not in ("backend", "precision") else " (default)"
        return ""

    print("Config (resolved)")
    print(f"  Model          {config.model}")
    print(f"  Backend        {config.backend}{_annotate('backend', config.backend)}")
    print(f"  Precision      {config.precision}{_annotate('precision', config.precision)}")

    # Batch size — from pytorch section if present
    batch_size: int | None = None
    if config.pytorch is not None and hasattr(config.pytorch, "batch_size"):
        batch_size = config.pytorch.batch_size
    if batch_size is not None:
        print(f"  Batch size     {batch_size}")

    # Dataset display
    if isinstance(config.dataset, str):
        dataset_str = f"{config.dataset} ({config.n} prompts)"
    else:
        # SyntheticDatasetConfig
        dataset_str = f"synthetic ({config.dataset.n} prompts, {config.dataset.input_len} in / {config.dataset.output_len} out)"
    print(f"  Dataset        {dataset_str}")

    output_dir = config.output_dir or "results/ (default)"
    print(f"  Output         {output_dir}")
    print()

    # --- VRAM estimate ---
    print("VRAM estimate")
    if vram is None:
        print("  (unavailable)")
    else:
        weights_line = f"  Weights        {_sig3(vram['weights_gb'])} GB ({config.precision})"
        print(weights_line)
        print(f"  KV cache       {_sig3(vram['kv_cache_gb'])} GB")
        print(f"  Overhead       {_sig3(vram['overhead_gb'])} GB")

        total_line = f"  Total          ~{_sig3(vram['total_gb'])} GB"
        if gpu_vram_gb is not None:
            fits = vram["total_gb"] <= gpu_vram_gb
            status = "OK" if fits else "WARNING: may not fit"
            total_line += f" / {_sig3(gpu_vram_gb)} GB available   {status}"
        print(total_line)
    print()

    print("Config valid. Run without --dry-run to start.")


def format_error(error: LLEMError, verbose: bool = False) -> str:
    """Format an LLEMError for stderr output.

    With verbose=True, includes full traceback.
    Otherwise, just the error class name and message.
    """
    class_name = type(error).__name__
    message = f"{class_name}: {error}"
    if verbose:
        tb = traceback.format_exc()
        if tb and tb.strip() != "NoneType: None":
            return f"{tb}\n{message}"
    return message


def format_validation_error(e: ValidationError) -> str:
    """Format a Pydantic ValidationError with a friendly header.

    Includes did-you-mean suggestions for literal_error types.
    Does NOT catch or re-wrap the error — only formats it.
    """
    from llenergymeasure.config.ssot import PRECISION_SUPPORT

    errors = e.errors()
    n = len(errors)
    header = f"Config validation failed ({n} error{'s' if n > 1 else ''}):"
    lines = [header]

    # Build a set of valid values for did-you-mean suggestions
    valid_backends = list(PRECISION_SUPPORT.keys())
    valid_precisions = list({p for precs in PRECISION_SUPPORT.values() for p in precs})

    for err in errors:
        loc_parts = [str(part) for part in err.get("loc", [])]
        loc_str = " -> ".join(loc_parts) if loc_parts else "(root)"
        msg = err.get("msg", "")
        lines.append(f"  {loc_str}: {msg}")

        # Did-you-mean for literal errors on known enum fields
        if err.get("type") == "literal_error":
            # Try to extract the bad value from the error input
            bad_value = err.get("input")
            if bad_value is not None and isinstance(bad_value, str):
                # Determine which pool to search based on location
                last_loc = loc_parts[-1] if loc_parts else ""
                if last_loc == "backend":
                    pool = valid_backends
                elif last_loc == "precision":
                    pool = valid_precisions
                else:
                    pool = valid_backends + valid_precisions

                suggestions = difflib.get_close_matches(bad_value, pool, n=3, cutoff=0.6)
                if suggestions:
                    lines.append(f"    Did you mean: {', '.join(suggestions)}?")
                    lines.append(f"    Valid values: {', '.join(pool)}")

    return "\n".join(lines)


def print_study_dry_run(
    study_config: object,
    verbose: bool = False,
) -> None:
    """Print dry-run output for a study to stdout.

    Shows grid summary, per-experiment configs, and VRAM estimate for the
    largest model. Mirrors the single-experiment dry-run format.
    """
    from llenergymeasure.cli._vram import estimate_vram, get_gpu_vram_gb
    from llenergymeasure.config.grid import format_preflight_summary
    from llenergymeasure.config.models import StudyConfig

    assert isinstance(study_config, StudyConfig)

    # Pre-flight summary (configs x cycles = runs, order)
    summary = format_preflight_summary(study_config)
    print(summary)
    print()

    # Per-experiment config table
    header = f"{'#':>3}  {'Model':<25}  {'Backend':<10}  {'Precision':<10}  {'n':>5}"
    print(header)
    print("-" * len(header))
    for i, exp in enumerate(study_config.experiments, 1):
        model_str = exp.model
        if len(model_str) > 25:
            model_str = "..." + model_str[-22:]
        print(f"{i:>3}  {model_str:<25}  {exp.backend:<10}  {exp.precision:<10}  {exp.n:>5}")
    print()

    # VRAM estimate for the peak model (largest weight estimate)
    gpu_vram_gb = get_gpu_vram_gb()
    peak_vram: dict[str, float] | None = None
    peak_config: ExperimentConfig | None = None
    for exp in study_config.experiments:
        vram = estimate_vram(exp)
        if vram is not None and (peak_vram is None or vram["total_gb"] > peak_vram["total_gb"]):
            peak_vram = vram
            peak_config = exp

    print("VRAM estimate (peak)")
    if peak_vram is not None and peak_config is not None:
        print(f"  Weights        {_sig3(peak_vram['weights_gb'])} GB ({peak_config.precision})")
        print(f"  KV cache       {_sig3(peak_vram['kv_cache_gb'])} GB")
        print(f"  Overhead       {_sig3(peak_vram['overhead_gb'])} GB")
        total_line = f"  Total          ~{_sig3(peak_vram['total_gb'])} GB"
        if gpu_vram_gb is not None:
            fits = peak_vram["total_gb"] <= gpu_vram_gb
            status = "OK" if fits else "WARNING: may not fit"
            total_line += f" / {_sig3(gpu_vram_gb)} GB available   {status}"
        print(total_line)
    else:
        print("  (unavailable)")
    print()

    print("Config valid. Run without --dry-run to start.")


def print_experiment_header(config: ExperimentConfig) -> None:
    """Print one-line experiment header to stderr (progress area).

    Shows model + backend always, plus non-default parameters worth noting.
    """
    parts = [config.model, config.backend]

    # Always show precision (key for reproducibility)
    parts.append(config.precision)

    # Show n if non-default
    if config.n != 100:
        parts.append(f"n={config.n}")

    # Show max_output_tokens if non-default
    if config.max_output_tokens != 128:
        parts.append(f"max_out={config.max_output_tokens}")

    # Show batch_size if configured
    if config.pytorch is not None and hasattr(config.pytorch, "batch_size"):
        bs = config.pytorch.batch_size
        if bs is not None and bs != 1:
            parts.append(f"batch={bs}")

    print(f"Experiment: {' | '.join(parts)}", file=sys.stderr)


def print_study_progress(
    index: int,
    total: int,
    config: ExperimentConfig,
    status: str = "running",
    elapsed: float | None = None,
    energy: float | None = None,
) -> None:
    """Print a per-experiment progress line to stderr.

    Format: [3/12] <icon> model backend precision -- elapsed (energy)
    Icons: completed=OK, failed=FAIL, running=...

    Args:
        index: 1-based experiment index.
        total: Total experiments in study.
        config: ExperimentConfig for this experiment.
        status: "running", "completed", or "failed".
        elapsed: Elapsed time in seconds (None if not yet available).
        energy: Energy in joules (None if not yet available).
    """
    icons = {"running": "...", "completed": "OK", "failed": "FAIL"}
    icon = icons.get(status, "?")

    parts = [f"[{index}/{total}]", icon, config.model, config.backend, config.precision]

    if elapsed is not None:
        parts.append("--")
        parts.append(_format_duration(elapsed))
    if energy is not None:
        parts.append(f"({_sig3(energy)} J)")

    line = " ".join(parts)
    print(line, file=sys.stderr)


def print_study_summary(result: StudyResult) -> None:
    """Print study summary table to stdout.

    Columns: #, Config, Status, Time, Energy, tok/s
    Failed experiments show error type instead of metrics.
    Footer with totals.

    Args:
        result: Completed StudyResult.
    """
    print()
    print(f"Study: {result.name or 'unnamed'}")
    if result.study_design_hash:
        print(f"Hash:  {result.study_design_hash}")
    print()

    # Table header
    header = f"{'#':>3}  {'Config':<40}  {'Status':<8}  {'Time':>8}  {'Energy':>10}  {'tok/s':>8}"
    print(header)
    print("-" * len(header))

    # Table rows
    for i, exp in enumerate(result.experiments, 1):
        model_short = exp.effective_config.get("model", "unknown")
        if len(model_short) > 20:
            model_short = "..." + model_short[-17:]
        backend = exp.backend
        precision = getattr(exp, "precision", "?")
        if hasattr(exp, "effective_config"):
            precision = exp.effective_config.get("precision", precision)
        config_str = f"{model_short} / {backend} / {precision}"
        if len(config_str) > 40:
            config_str = config_str[:37] + "..."

        time_str = _format_duration(exp.duration_sec)
        energy_str = f"{_sig3(exp.total_energy_j)} J"
        toks_str = _sig3(exp.avg_tokens_per_second)

        print(
            f"{i:>3}  {config_str:<40}  {'OK':<8}  {time_str:>8}  {energy_str:>10}  {toks_str:>8}"
        )

    print("-" * len(header))

    # Footer with totals
    if result.summary:
        s = result.summary
        print(
            f"Total: {s.completed}/{s.total_experiments} completed"
            f"  |  {_format_duration(s.total_wall_time_s)}"
            f"  |  {_sig3(s.total_energy_j)} J"
        )
        if s.failed > 0:
            print(f"Failed: {s.failed} experiment(s)")
        if s.warnings:
            for w in s.warnings:
                print(f"  Warning: {w}")
    print()

    # Output paths
    if result.result_files:
        print(f"Results saved: {len(result.result_files)} file(s)")
        for path in result.result_files[:3]:
            print(f"  {path}")
        if len(result.result_files) > 3:
            print(f"  ... and {len(result.result_files) - 3} more")
