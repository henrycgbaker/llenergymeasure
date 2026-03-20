"""llem run — primary command for running LLM efficiency experiments."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from llenergymeasure.api import run_experiment
from llenergymeasure.cli._display import (
    format_error,
    format_validation_error,
    print_dry_run,
    print_result_summary,
)
from llenergymeasure.cli._vram import estimate_vram, get_gpu_vram_gb
from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.utils.exceptions import (
    BackendError,
    ConfigError,
    ExperimentError,
    PreFlightError,
    StudyError,
)

# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def run(
    config: Annotated[
        Path | None,
        typer.Argument(help="Path to experiment YAML config"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name or HuggingFace path"),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", "-b", help="Inference backend (pytorch, vllm, tensorrt)"),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="Dataset name"),
    ] = None,
    n: Annotated[
        int | None,
        typer.Option("-n", help="Number of prompts to run"),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Batch size (PyTorch backend)"),
    ] = None,
    precision: Annotated[
        str | None,
        typer.Option(
            "--precision", "-p", help="Floating point precision (fp32, fp16, bf16, int8, int4)"
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output directory for results"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config and estimate VRAM without running"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress bars"),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v=INFO, -vv=DEBUG)"),
    ] = 0,
    cycles: Annotated[
        int | None,
        typer.Option("--cycles", help="Number of cycles (study mode)"),
    ] = None,
    order: Annotated[
        str | None,
        typer.Option(
            "--order", help="Cycle ordering: sequential, interleaved, shuffled (study mode)"
        ),
    ] = None,
    no_gaps: Annotated[
        bool,
        typer.Option("--no-gaps", help="Disable thermal gaps between experiments (study mode)"),
    ] = False,
    skip_preflight: Annotated[
        bool,
        typer.Option(
            "--skip-preflight",
            help="Skip Docker pre-flight checks (GPU visibility, CUDA/driver compatibility)",
        ),
    ] = False,
) -> None:
    """Run an LLM efficiency experiment."""

    from llenergymeasure.cli import _setup_logging

    _setup_logging(verbose)

    # Install SIGINT handler so Ctrl-C exits with code 130
    def _handle_sigint(signum: int, frame: Any) -> None:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        _run_impl(
            config=config,
            model=model,
            backend=backend,
            dataset=dataset,
            n=n,
            batch_size=batch_size,
            precision=precision,
            output=output,
            dry_run=dry_run,
            quiet=quiet,
            verbose=verbose > 0,
            cycles=cycles,
            order=order,
            no_gaps=no_gaps,
            skip_preflight=skip_preflight,
        )
    except ConfigError as e:
        print(format_error(e, verbose=verbose > 0), file=sys.stderr)
        raise typer.Exit(code=2) from None
    except (PreFlightError, ExperimentError, BackendError, StudyError) as e:
        print(format_error(e, verbose=verbose > 0), file=sys.stderr)
        raise typer.Exit(code=1) from None
    except ValidationError as e:
        print(format_validation_error(e), file=sys.stderr)
        raise typer.Exit(code=2) from None
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _run_impl(
    config: Path | None,
    model: str | None,
    backend: str | None,
    dataset: str | None,
    n: int | None,
    batch_size: int | None,
    precision: str | None,
    output: str | None,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    cycles: int | None = None,
    order: str | None = None,
    no_gaps: bool = False,
    skip_preflight: bool = False,
) -> None:
    """Core implementation — separated for clean error handling in run()."""
    # Build CLI overrides dict — only include flags the user explicitly passed
    cli_overrides: dict[str, Any] = {}
    if model is not None:
        cli_overrides["model"] = model
    if backend is not None:
        cli_overrides["backend"] = backend
    if dataset is not None:
        cli_overrides["dataset"] = dataset
    if n is not None:
        cli_overrides["n"] = n
    if batch_size is not None:
        # Dotted key for _unflatten() in loader — maps to pytorch.batch_size
        cli_overrides["pytorch.batch_size"] = batch_size
    if precision is not None:
        cli_overrides["precision"] = precision
    if output is not None:
        cli_overrides["output_dir"] = output

    # Validate we have enough information to resolve a config
    if config is None and model is None:
        raise ConfigError(
            "Provide a config file or --model flag.\n"
            "  Examples:\n"
            "    llem run experiment.yaml\n"
            "    llem run --model gpt2 --backend pytorch"
        )

    # Study detection: YAML with sweep: or experiments: keys is a study
    is_study = False
    if config is not None:
        import yaml

        try:
            raw = yaml.safe_load(config.read_text())
            if isinstance(raw, dict) and ("sweep" in raw or "experiments" in raw):
                is_study = True
        except Exception:
            pass  # Fall through to normal experiment path — loader will raise if invalid

    # Route to study execution path
    if is_study:
        assert config is not None  # Guarded by study detection above
        _run_study_impl(
            config=config,
            cli_overrides=cli_overrides,
            cycles=cycles,
            order=order,
            no_gaps=no_gaps,
            quiet=quiet,
            verbose=verbose,
            skip_preflight=skip_preflight,
            dry_run=dry_run,
        )
        return

    # Load/resolve the experiment config
    experiment_config = load_experiment_config(
        path=config,
        cli_overrides=cli_overrides if cli_overrides else None,
    )

    # --- Dry-run branch ---
    if dry_run:
        vram = estimate_vram(experiment_config)
        gpu_vram_gb = get_gpu_vram_gb()
        print_dry_run(experiment_config, vram, gpu_vram_gb, verbose=verbose)
        return

    # --- Run branch ---
    # Build experiment header string
    header = _build_header(experiment_config)

    # Create progress display (None in quiet mode).
    # Steps are pre-registered with a fixed count so [x/y] counters are
    # stable. Steps that don't apply are shown as SKIP.
    progress = None
    if not quiet:
        from llenergymeasure.cli._step_display import StepDisplay
        from llenergymeasure.domain.progress import STEPS_DOCKER

        display = StepDisplay(header=f"Experiment: {header}")
        # Pre-register: Docker path is the common case (auto-elevation).
        # Local path is rare (only when runner explicitly set to local).
        display.register_steps(STEPS_DOCKER)
        display.start()
        progress = display

    try:
        result = run_experiment(experiment_config, skip_preflight=skip_preflight, progress=progress)
    finally:
        if progress is not None:
            display.finish()

    print_result_summary(result)

    # Save output if output_dir specified
    if experiment_config.output_dir:
        from llenergymeasure.api import save_result

        output_dir = Path(experiment_config.output_dir)
        ts_source = output_dir / result.timeseries if result.timeseries else None
        save_result(result, output_dir, timeseries_source=ts_source)
        # Clean up stale flat timeseries file after copy into subdirectory
        if ts_source is not None:
            ts_source.unlink(missing_ok=True)
        print(f"Saved: {experiment_config.output_dir}", file=sys.stderr)


def _build_header(config: Any) -> str:
    """Build a compact experiment header string from config."""
    parts = [config.model, config.backend, config.precision]
    if config.n != 100:
        parts.append(f"n={config.n}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Study execution path
# ---------------------------------------------------------------------------


def _run_study_impl(
    config: Path,
    cli_overrides: dict[str, Any],
    cycles: int | None,
    order: str | None,
    no_gaps: bool,
    quiet: bool,
    verbose: bool,
    skip_preflight: bool = False,
    dry_run: bool = False,
) -> None:
    """Study execution path — separated for clean error handling."""
    import yaml

    from llenergymeasure.cli._display import print_study_dry_run, print_study_summary
    from llenergymeasure.config.grid import format_preflight_summary
    from llenergymeasure.config.loader import load_study_config

    # Check what the YAML execution block specifies (to apply CLI effective defaults)
    raw = yaml.safe_load(config.read_text()) or {}
    yaml_execution = raw.get("execution", {}) or {}

    # Build execution overrides from CLI flags
    exec_overrides: dict[str, Any] = {}

    # CLI effective defaults: n_cycles=3, cycle_order="shuffled" when neither YAML nor CLI specifies
    # These are applied at CLI layer; Pydantic defaults are conservative (n_cycles=1)
    if cycles is not None:
        exec_overrides["n_cycles"] = cycles
    elif "n_cycles" not in yaml_execution:
        exec_overrides["n_cycles"] = 3  # CLI effective default

    if order is not None:
        exec_overrides["cycle_order"] = order
    elif "cycle_order" not in yaml_execution:
        exec_overrides["cycle_order"] = "shuffled"  # CLI effective default

    if no_gaps:
        exec_overrides["experiment_gap_seconds"] = 0
        exec_overrides["cycle_gap_seconds"] = 0

    # Build full CLI overrides dict
    study_cli_overrides: dict[str, Any] = {}
    if cli_overrides:
        study_cli_overrides.update(cli_overrides)
    if exec_overrides:
        study_cli_overrides["execution"] = exec_overrides

    # Load study config with overrides
    study_config = load_study_config(
        path=config,
        cli_overrides=study_cli_overrides if study_cli_overrides else None,
    )

    # --- Dry-run branch ---
    if dry_run:
        print_study_dry_run(study_config, verbose=verbose)
        return

    # Pre-flight summary display
    if not quiet:
        summary = format_preflight_summary(study_config)
        print(summary, file=sys.stderr)
        print(file=sys.stderr)

    # Run the study — pass skip_preflight so CLI flag overrides YAML config
    from llenergymeasure import run_study

    result = run_study(study_config, skip_preflight=skip_preflight)

    # Display summary
    if not quiet:
        print_study_summary(result)

    # Show output path
    if result.result_files:
        first = result.result_files[0]
        print(f"Study results: {first}", file=sys.stderr)
