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
from llenergymeasure.config.ssot import (
    BACKEND_PYTORCH,
    RUNNER_DOCKER,
    RUNNER_LOCAL,
)
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
    n_prompts: Annotated[
        int | None,
        typer.Option("--n-prompts", "-n", help="Number of prompts to run"),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Batch size (PyTorch backend)"),
    ] = None,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", "-p", help="Model dtype (float32, float16, bfloat16)"),
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
            "--order",
            help="Experiment ordering: sequential, interleave, shuffle, reverse, latin_square (study mode)",
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
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume most recent interrupted study"),
    ] = False,
    resume_dir: Annotated[
        Path | None,
        typer.Option("--resume-dir", help="Resume a specific study directory"),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast", help="Abort study on first failure (circuit breaker threshold=1)"
        ),
    ] = False,
    no_circuit_breaker: Annotated[
        bool,
        typer.Option("--no-circuit-breaker", help="Disable circuit breaker entirely"),
    ] = False,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Study wall-clock timeout in hours (e.g. 24, 1.5)"),
    ] = None,
    no_lock: Annotated[
        bool,
        typer.Option("--no-lock", help="Disable GPU lock files (advanced)"),
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
            n_prompts=n_prompts,
            batch_size=batch_size,
            dtype=dtype,
            output=output,
            dry_run=dry_run,
            quiet=quiet,
            verbose=verbose > 0,
            cycles=cycles,
            order=order,
            no_gaps=no_gaps,
            skip_preflight=skip_preflight,
            resume=resume,
            resume_dir=resume_dir,
            fail_fast=fail_fast,
            no_circuit_breaker=no_circuit_breaker,
            timeout=timeout,
            no_lock=no_lock,
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
    n_prompts: int | None,
    batch_size: int | None,
    dtype: str | None,
    output: str | None,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    cycles: int | None = None,
    order: str | None = None,
    no_gaps: bool = False,
    skip_preflight: bool = False,
    resume: bool = False,
    resume_dir: Path | None = None,
    fail_fast: bool = False,
    no_circuit_breaker: bool = False,
    timeout: float | None = None,
    no_lock: bool = False,
) -> None:
    """Core implementation — separated for clean error handling in run()."""
    # Build CLI overrides dict — only include flags the user explicitly passed
    cli_overrides: dict[str, Any] = {}
    if model is not None:
        cli_overrides["model"] = model
    if backend is not None:
        cli_overrides["backend"] = backend
    if dataset is not None:
        cli_overrides["dataset.source"] = dataset
    if n_prompts is not None:
        cli_overrides["dataset.n_prompts"] = n_prompts
    if batch_size is not None:
        # Dotted key for _unflatten() in loader — maps to pytorch.batch_size
        cli_overrides["pytorch.batch_size"] = batch_size
    if dtype is not None:
        cli_overrides["dtype"] = dtype

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
            output=output,
            resume=resume,
            resume_dir=resume_dir,
            fail_fast=fail_fast,
            no_circuit_breaker=no_circuit_breaker,
            timeout=timeout,
            no_lock=no_lock,
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
        print_dry_run(experiment_config, vram, gpu_vram_gb, verbose=verbose, output_dir=output)
        return

    # --- Run branch ---
    # Build experiment header string
    runner_tag = _resolve_runner_tag(experiment_config)
    header = _build_header(experiment_config, runner_tag=runner_tag)

    effective_mode = _resolve_progress_mode(quiet, verbose)

    # Create progress display (None in quiet mode).
    # Steps are pre-registered with a fixed count so [x/y] counters are
    # stable. Steps that don't apply are shown as SKIP.
    progress = None
    display = None
    if effective_mode != "quiet":
        from llenergymeasure.cli._step_display import StepDisplay
        from llenergymeasure.domain.progress import STEPS_DOCKER

        display = StepDisplay(
            header=f"Experiment: {header}",
            force_plain=effective_mode == "plain",
        )
        # Pre-register: Docker path is the common case (auto-elevation).
        # Local path is rare (only when runner explicitly set to local).
        display.register_steps(STEPS_DOCKER)
        display.start()
        progress = display

    result = None
    try:
        result = run_experiment(experiment_config, skip_preflight=skip_preflight, progress=progress)
    finally:
        if display is not None:
            energy = getattr(result, "total_energy_j", None) if result is not None else None
            throughput = (
                getattr(result, "avg_tokens_per_second", None) if result is not None else None
            )
            display.finish(energy_j=energy, throughput_tok_s=throughput)

    print_result_summary(result)

    # Save output if --output flag specified (runtime param, not config field)
    if output:
        from llenergymeasure.api import save_result

        output_path = Path(output)
        ts_source = output_path / result.timeseries if result.timeseries else None
        save_result(result, output_path, timeseries_source=ts_source)
        # Clean up stale flat timeseries file after copy into subdirectory
        if ts_source is not None:
            ts_source.unlink(missing_ok=True)
        print(f"Saved: {output}", file=sys.stderr)


def _resolve_progress_mode(quiet: bool, verbose: bool) -> str:
    """Resolve effective progress mode: CLI flags > user config > default."""
    if quiet:
        return "quiet"
    if verbose:
        return "plain"
    from llenergymeasure.config.user_config import load_user_config

    return load_user_config().ui.progress_mode


def _resolve_runner_tag(config: Any) -> str:
    """Determine the runner tag string for display from config.runner.

    Returns "local" or "docker" based on the runner field.
    """
    runner = getattr(config, "runner", "auto")
    if runner == RUNNER_LOCAL:
        return RUNNER_LOCAL
    if runner == RUNNER_DOCKER or (isinstance(runner, str) and runner.startswith("docker:")):
        return RUNNER_DOCKER
    # auto: pytorch defaults to local, vllm/tensorrt default to docker
    backend = getattr(config, "backend", BACKEND_PYTORCH)
    return RUNNER_LOCAL if backend == BACKEND_PYTORCH else RUNNER_DOCKER


def _build_header(config: Any, runner_tag: str = RUNNER_LOCAL) -> str:
    """Build compact experiment header: model | backend [runner] + deviation fields.

    Args:
        config: ExperimentConfig with model, backend, dtype, dataset fields.
        runner_tag: Runner tag string ("local" or "docker").
    """
    from llenergymeasure.config.models import DatasetConfig, ExperimentConfig

    _fields = ExperimentConfig.model_fields
    _ds_fields = DatasetConfig.model_fields
    default_dtype = _fields["dtype"].default
    default_n = _ds_fields["n_prompts"].default
    default_source = _ds_fields["source"].default

    # Strip HuggingFace org prefix (meta-llama/Llama-3.2-1B-Instruct -> Llama-3.2-1B-Instruct)
    model = config.model.split("/")[-1] if "/" in config.model else config.model
    parts = [f"{model} | {config.backend}"]
    # Deviation fields (only when non-default)
    if config.dtype != default_dtype:
        parts.append(config.dtype)
    if config.dataset.n_prompts != default_n:
        parts.append(f"n_prompts={config.dataset.n_prompts}")
    if config.dataset.source != default_source:
        parts.append(config.dataset.source)
    return f"{' | '.join(parts)} [{runner_tag}]"


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
    output: str | None = None,
    resume: bool = False,
    resume_dir: Path | None = None,
    fail_fast: bool = False,
    no_circuit_breaker: bool = False,
    timeout: float | None = None,
    no_lock: bool = False,
) -> None:
    """Study execution path — separated for clean error handling."""
    import yaml

    from llenergymeasure.cli._display import print_study_dry_run
    from llenergymeasure.config.grid import build_preflight_panel
    from llenergymeasure.config.loader import load_study_config

    # Fast-fail: verify resume target exists before expensive grid expansion.
    if resume_dir is not None:
        if not (resume_dir / "manifest.json").exists():
            raise typer.BadParameter(
                f"No manifest.json in {resume_dir} — not a valid study directory.",
                param_hint="--resume-dir",
            )
    elif resume:
        from llenergymeasure.api import find_resumable_study

        _output = Path(output or "./results")
        if not find_resumable_study(_output):
            raise typer.BadParameter(
                f"No resumable study found in {_output}. Run a study first or use --resume-dir.",
                param_hint="--resume",
            )

    # Check what the YAML execution block specifies (to apply CLI effective defaults)
    raw = yaml.safe_load(config.read_text()) or {}
    yaml_execution = raw.get("study_execution", {}) or {}

    # Build execution overrides from CLI flags
    exec_overrides: dict[str, Any] = {}

    # CLI effective defaults: n_cycles=3, experiment_order="shuffle" when neither YAML nor CLI specifies
    # These are applied at CLI layer; Pydantic defaults are conservative (n_cycles=1)
    if cycles is not None:
        exec_overrides["n_cycles"] = cycles
    elif "n_cycles" not in yaml_execution:
        exec_overrides["n_cycles"] = 3  # CLI effective default

    if order is not None:
        exec_overrides["experiment_order"] = order
    elif "experiment_order" not in yaml_execution:
        exec_overrides["experiment_order"] = "shuffle"  # CLI effective default

    if no_gaps:
        exec_overrides["experiment_gap_seconds"] = 0
        exec_overrides["cycle_gap_seconds"] = 0

    # Robustness overrides: circuit breaker, timeout
    if fail_fast:
        exec_overrides["max_consecutive_failures"] = 1
        exec_overrides["circuit_breaker_cooldown_seconds"] = 0
    if no_circuit_breaker:
        exec_overrides["max_consecutive_failures"] = 0
    if timeout is not None:
        exec_overrides["wall_clock_timeout_hours"] = timeout

    # Build full CLI overrides dict
    study_cli_overrides: dict[str, Any] = {}
    if cli_overrides:
        study_cli_overrides.update(cli_overrides)
    if exec_overrides:
        study_cli_overrides["study_execution"] = exec_overrides

    # Load study config with overrides
    study_config = load_study_config(
        path=config,
        cli_overrides=study_cli_overrides if study_cli_overrides else None,
    )

    # ---------------------------------------------------------------
    # Resolve runners and compute study dir preview — shared by both
    # dry-run and actual-run so both show the same preflight panel.
    # ---------------------------------------------------------------
    from datetime import datetime, timezone

    from llenergymeasure.api import probe_energy_sampler, run_study_preflight
    from llenergymeasure.config.user_config import load_user_config

    user_config = load_user_config()
    try:
        runner_specs = run_study_preflight(
            study_config,
            # Dry-run: skip Docker binary checks (just resolve runner modes).
            skip_preflight=skip_preflight or dry_run,
            yaml_runners=study_config.runners,
            user_config=user_config.runners,
            yaml_images=study_config.images,
            user_config_images=user_config.images or None,
        )
    except Exception:
        runner_specs = None  # graceful: Docker unavailable, show YAML runners

    prefix = study_config.study_name or "study"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    study_dir_preview = Path("results") / f"{prefix}_{ts}"

    # --- Dry-run branch ---
    if dry_run:
        print_study_dry_run(
            study_config,
            verbose=verbose,
            runner_specs=runner_specs,
            study_dir=study_dir_preview,
        )
        return

    effective_mode = _resolve_progress_mode(quiet, verbose)

    # Create live study display before the run so per-experiment progress is shown
    study_display = None
    if effective_mode != "quiet":
        from rich.console import Console as RichConsole

        from llenergymeasure.cli._step_display import StudyStepDisplay

        n_exp = len(study_config.experiments)
        n_cycles = study_config.study_execution.n_cycles
        name = study_config.study_name or "unnamed"

        _stderr_console = RichConsole(stderr=True)
        panel = build_preflight_panel(
            study_config,
            runner_specs=runner_specs,
            study_dir=study_dir_preview,
            probed_energy_sampler=probe_energy_sampler(),
        )
        _stderr_console.print(panel)

        if study_config.skipped_configs:
            n_skip = len(study_config.skipped_configs)
            _stderr_console.print(
                f"Skipped {n_skip} invalid config(s) — details in skipped_configs.log"
            )

        study_display = StudyStepDisplay(
            total_experiments=n_exp,
            study_name=name,
            n_cycles=n_cycles,
            force_plain=effective_mode == "plain",
        )
        # Header already printed above; start Live without repeating it
        study_display.start(print_header=False)

    # Track elapsed time around the study run
    import time as _time

    _study_start = _time.monotonic()

    # Run the study with live progress display.
    # skip_preflight=True because we already ran preflight above.
    from llenergymeasure import run_study

    try:
        result = run_study(
            study_config,
            skip_preflight=True,
            progress=study_display,
            resume=resume,
            resume_dir=resume_dir,
            output_dir=Path(output) if output else None,
            no_lock=no_lock,
        )
    finally:
        # Safety stop — ensures Rich Live is torn down even on exceptions
        if study_display is not None:
            study_display.stop()

    _study_elapsed = _time.monotonic() - _study_start

    # Study completion footer
    if study_display is not None:
        save_path = str(result.result_files[0]) if result.result_files else None
        study_display.finish(save_path=save_path, total_elapsed=_study_elapsed)
    elif effective_mode != "quiet":
        from llenergymeasure.cli._display import print_study_summary

        print_study_summary(result)
