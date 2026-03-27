"""CLI output formatting utilities.

All result output goes to stdout (scientific record).
Progress/header output goes to stderr (transient display area).
"""

from __future__ import annotations

import difflib
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

if TYPE_CHECKING:
    from llenergymeasure.infra.runner_resolution import RunnerSpec

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult
from llenergymeasure.utils.exceptions import DockerError, LLEMError
from llenergymeasure.utils.formatting import compute_mj_per_tok as _compute_mj_per_tok
from llenergymeasure.utils.formatting import format_elapsed as _format_duration
from llenergymeasure.utils.formatting import model_short_name
from llenergymeasure.utils.formatting import sig3 as _sig3


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
    # Per-token energy (mJ/tok)
    mj_tok = _compute_mj_per_tok(
        result.total_energy_j, result.avg_tokens_per_second, result.duration_sec
    )
    if mj_tok is not None:
        print(f"  Per token      {_sig3(mj_tok)} mJ/tok")
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
    print(f"  Meas. window   {_format_duration(result.duration_sec)}")
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
        "dtype": "bfloat16",
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
            return f" ({field} default)" if field not in ("backend", "dtype") else " (default)"
        return ""

    print("Config (resolved)")
    print(f"  Model          {config.model}")
    print(f"  Backend        {config.backend}{_annotate('backend', config.backend)}")
    print(f"  Precision      {config.dtype}{_annotate('precision', config.dtype)}")

    # Batch size — from pytorch section if present
    batch_size: int | None = None
    if config.pytorch is not None and hasattr(config.pytorch, "batch_size"):
        batch_size = config.pytorch.batch_size
    if batch_size is not None:
        print(f"  Batch size     {batch_size}")

    # Dataset display
    ds = config.dataset
    dataset_str = f"{ds.source} ({ds.n_prompts} prompts)"
    print(f"  Dataset        {dataset_str}")

    output_dir = config.output_dir or "results/ (default)"
    print(f"  Output         {output_dir}")
    print()

    # --- VRAM estimate ---
    print("VRAM estimate")
    if vram is None:
        print("  (unavailable)")
    else:
        weights_line = f"  Weights        {_sig3(vram['weights_gb'])} GB ({config.dtype})"
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

    For DockerError subclasses, appends fix_suggestion and stderr_snippet
    so the user sees actionable guidance without needing to dig into logs.
    """
    class_name = type(error).__name__
    message = f"{class_name}: {error}"
    if verbose:
        tb = traceback.format_exc()
        if tb and tb.strip() != "NoneType: None":
            message = f"{tb}\n{message}"

    # Append Docker-specific details when available
    if isinstance(error, DockerError):
        if error.fix_suggestion:
            message += f"\n\nSuggestion: {error.fix_suggestion}"
        if error.stderr_snippet:
            message += f"\n\nContainer stderr (last 20 lines):\n{error.stderr_snippet}"

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
    valid_dtypes = list({p for precs in PRECISION_SUPPORT.values() for p in precs})

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
                elif last_loc == "dtype":
                    pool = valid_dtypes
                else:
                    pool = valid_backends + valid_dtypes

                suggestions = difflib.get_close_matches(bad_value, pool, n=3, cutoff=0.6)
                if suggestions:
                    lines.append(f"    Did you mean: {', '.join(suggestions)}?")
                    lines.append(f"    Valid values: {', '.join(pool)}")

    return "\n".join(lines)


def print_study_dry_run(
    study_config: object,
    verbose: bool = False,
    runner_specs: dict[str, RunnerSpec] | None = None,
    study_dir: Path | None = None,
) -> None:
    """Print dry-run output for a study to stdout.

    Shows grid summary, per-experiment configs, and VRAM estimate for the
    largest model. Mirrors the single-experiment dry-run format.
    """
    from rich.console import Console as RichConsole

    from llenergymeasure.api import probe_energy_sampler
    from llenergymeasure.cli._vram import estimate_vram, get_gpu_vram_gb
    from llenergymeasure.config.grid import build_preflight_panel
    from llenergymeasure.config.models import StudyConfig
    from llenergymeasure.utils.formatting import format_experiment_header

    assert isinstance(study_config, StudyConfig)

    # Pre-flight panel — same args as actual run so both show identical output
    _stdout_console = RichConsole()
    panel = build_preflight_panel(
        study_config,
        runner_specs=runner_specs,
        study_dir=study_dir,
        probed_energy_sampler=probe_energy_sampler(),
    )
    _stdout_console.print(panel)

    if study_config.skipped_configs:
        skip_lines = [f"Skipping {len(study_config.skipped_configs)} config(s):"]
        for s in study_config.skipped_configs:
            label = s.get("short_label", "unknown")
            reason = s.get("reason", "unknown error")
            skip_lines.append(f"  - {label}: {reason}")
        _stdout_console.print("\n".join(skip_lines))
        _stdout_console.print()

    # Per-experiment list using the same header format as the live run
    n = len(study_config.experiments)
    width = len(str(n))
    for i, exp in enumerate(study_config.experiments, 1):
        print(f"  {i:>{width}}  {format_experiment_header(exp)}")
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
        print(f"  Weights        {_sig3(peak_vram['weights_gb'])} GB ({peak_config.dtype})")
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

    Uses ``format_experiment_header()`` for consistent formatting across
    single-experiment and study modes.
    """
    from llenergymeasure.utils.formatting import format_experiment_header

    print(f"Experiment: {format_experiment_header(config)}", file=sys.stderr)


def print_study_summary(result: StudyResult) -> None:
    """Print study summary table to stdout.

    Columns: #, Config, Status, Time, Energy, tok/s
    Failed experiments show error type instead of metrics.
    Footer with totals.

    Args:
        result: Completed StudyResult.
    """
    print()
    print(f"Study: {result.study_name or 'unnamed'}")
    if result.study_design_hash:
        print(f"Hash:  {result.study_design_hash}")
    print()

    # Table header
    header = (
        f"{'#':>3}  {'':>2}  {'Config':<40}  {'Total':>8}  {'Infer':>8}"
        f"  {'Energy':>10}  {'tok/s':>8}  {'mJ/tok':>8}"
    )
    print(header)
    print("-" * len(header))

    # Table rows
    for i, exp in enumerate(result.experiments, 1):
        # Build compact config string: model_short / backend / non-default params
        model_raw = exp.effective_config.get("model", "unknown")
        model_short = model_short_name(model_raw)
        if len(model_short) > 20:
            model_short = "..." + model_short[-17:]
        backend = exp.backend
        dtype_val = (
            exp.effective_config.get("dtype", "?") if hasattr(exp, "effective_config") else "?"
        )
        config_str = f"{model_short} / {backend} / {dtype_val}"
        if len(config_str) > 40:
            config_str = config_str[:37] + "..."

        # Status icon
        is_ok = exp.total_energy_j is not None and exp.total_energy_j > 0
        status_icon = "\u2713" if is_ok else "\u2717"

        total_str = _format_duration(exp.duration_sec)
        infer_str = (
            _format_duration(exp.total_inference_time_sec)
            if exp.total_inference_time_sec > 0
            else "-"
        )
        energy_str = f"{_sig3(exp.total_energy_j)} J" if exp.total_energy_j else "-"
        toks_str = _sig3(exp.avg_tokens_per_second) if exp.avg_tokens_per_second else "-"

        mj_tok = _compute_mj_per_tok(
            exp.total_energy_j, exp.avg_tokens_per_second, exp.duration_sec
        )
        mj_str = _sig3(mj_tok) if mj_tok is not None else "-"

        print(
            f"{i:>3}  {status_icon:>2}  {config_str:<40}  {total_str:>8}  {infer_str:>8}"
            f"  {energy_str:>10}  {toks_str:>8}  {mj_str:>8}"
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
