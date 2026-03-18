"""Internal API implementation for llenergymeasure.

This module is internal (underscore prefix). Import via llenergymeasure.__init__ only.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# GPU index resolution
# ---------------------------------------------------------------------------
from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.device.gpu_info import _resolve_gpu_indices
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult
from llenergymeasure.utils.exceptions import ConfigError

# ---------------------------------------------------------------------------
# run_experiment — three overloaded forms
# ---------------------------------------------------------------------------


@overload
def run_experiment(config: str | Path, *, skip_preflight: bool = ...) -> ExperimentResult: ...


@overload
def run_experiment(config: ExperimentConfig, *, skip_preflight: bool = ...) -> ExperimentResult: ...


@overload
def run_experiment(
    config: None = None,
    *,
    model: str,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    skip_preflight: bool = ...,
    **kwargs: Any,
) -> ExperimentResult: ...


def run_experiment(
    config: str | Path | ExperimentConfig | None = None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    skip_preflight: bool = False,
    **kwargs: Any,
) -> ExperimentResult:
    """Run a single LLM inference efficiency experiment.

    Side-effect free: no disk writes unless output_dir is specified in the config.

    Three call forms:
        run_experiment("config.yaml")              # YAML path
        run_experiment(ExperimentConfig(...))       # config object
        run_experiment(model="gpt2", backend="Y")  # kwargs convenience

    Args:
        config: YAML file path, ExperimentConfig object, or None (use kwargs).
        model: Model name/path (kwargs form only).
        backend: Inference backend (kwargs form only, defaults to ExperimentConfig default).
        n: Number of prompts (kwargs form only, default 100).
        dataset: Dataset name (kwargs form only, default "aienergyscore").
        skip_preflight: Skip Docker pre-flight checks (GPU visibility, CUDA/driver compat).
        **kwargs: Additional ExperimentConfig fields (kwargs form only).

    Returns:
        ExperimentResult: Experiment measurements and metadata.

    Raises:
        ConfigError: Invalid config path, missing model in kwargs form.
        pydantic.ValidationError: Invalid field values (passes through unchanged).
    """
    study = _to_study_config(config, model=model, backend=backend, n=n, dataset=dataset, **kwargs)
    study_result = _run(study, skip_preflight=skip_preflight)
    return study_result.experiments[0]


# ---------------------------------------------------------------------------
# run_study — M2 implementation
# ---------------------------------------------------------------------------


def run_study(config: str | Path | StudyConfig, *, skip_preflight: bool = False) -> StudyResult:
    """Run a multi-experiment study.

    Always writes manifest.json to disk (LA-05 — documented side-effect).
    Returns StudyResult with full schema (RES-13).

    Args:
        config: YAML file path or resolved StudyConfig.
        skip_preflight: Skip Docker pre-flight checks (GPU visibility, CUDA/driver compat).
            CLI --skip-preflight flag and YAML execution.skip_preflight: true also bypass.

    Returns:
        StudyResult with experiments, result_files, measurement_protocol, summary.

    Raises:
        ConfigError: Invalid config path or parse error.
        PreFlightError: Multi-backend study without Docker (CM-10).
        pydantic.ValidationError: Invalid field values (passes through unchanged).
    """
    if isinstance(config, (str, Path)):
        from llenergymeasure.config.loader import load_study_config

        study = load_study_config(path=Path(config))
    elif isinstance(config, StudyConfig):
        study = config
    else:
        raise ConfigError(f"Expected str, Path, or StudyConfig; got {type(config).__name__}")
    return _run(study, skip_preflight=skip_preflight)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_study_config(
    config: str | Path | ExperimentConfig | None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs: Any,
) -> StudyConfig:
    """Convert any run_experiment() input form to a degenerate StudyConfig."""
    if isinstance(config, ExperimentConfig):
        experiment = config
    elif isinstance(config, (str, Path)):
        experiment = load_experiment_config(path=Path(config))
    elif config is None:
        if model is None:
            raise ConfigError(
                "run_experiment() requires either a config argument or model= keyword.\n"
                "Example: run_experiment(model='meta-llama/Llama-3.1-8B')"
            )
        # Build kwargs dict for ExperimentConfig — only include non-default values
        # to let Pydantic defaults apply for omitted fields.
        ec_kwargs: dict[str, Any] = {"model": model, "n": n, "dataset": dataset}
        if backend is not None:
            ec_kwargs["backend"] = backend
        ec_kwargs.update(kwargs)
        experiment = ExperimentConfig(**ec_kwargs)
    else:
        raise ConfigError(
            f"Expected str, Path, ExperimentConfig, or None; got {type(config).__name__}"
        )
    return StudyConfig(experiments=[experiment])


def _run(study: StudyConfig, skip_preflight: bool = False) -> StudyResult:
    """Dispatcher: single experiment runs in-process; multi-experiment uses StudyRunner.

    Always:
    - Calls run_study_preflight(study, skip_preflight) first (CM-10 multi-backend guard
      and Docker pre-flight checks)
    - Resolves runner specs for all backends in the study
    - Creates study output directory and ManifestWriter (LA-05)
    - Returns fully populated StudyResult (RES-13 + RES-15)

    Single-experiment / n_cycles=1:  runs in-process or via DockerRunner directly.
    Otherwise:                         delegates to StudyRunner.
    """
    import logging

    from llenergymeasure.config.user_config import load_user_config
    from llenergymeasure.domain.experiment import StudySummary
    from llenergymeasure.infra.runner_resolution import resolve_study_runners
    from llenergymeasure.study.manifest import ManifestWriter, create_study_dir
    from llenergymeasure.study.preflight import run_study_preflight

    _api_logger = logging.getLogger(__name__)

    # Load user config first so runner context can be forwarded to preflight,
    # ensuring preflight uses the same runner resolution as the actual dispatch path.
    backends: list[str] = list({exp.backend for exp in study.experiments})
    user_config = load_user_config()

    # Multi-backend guard — raises PreFlightError for multi-backend studies (CM-10)
    # or auto-elevates to Docker when available (DOCK-05). Also runs Docker pre-flight
    # checks when any backend resolves to a Docker runner.
    run_study_preflight(
        study,
        skip_preflight=skip_preflight,
        yaml_runners=study.runners,
        user_config=user_config.runners,
    )

    # Resolve runner specs for all backends in the study
    runner_specs = resolve_study_runners(
        backends,
        yaml_runners=study.runners,
        user_config=user_config.runners,
    )

    # Warn on mixed runners (some local, some docker)
    modes = {spec.mode for spec in runner_specs.values()}
    if len(modes) > 1:
        _api_logger.warning(
            "Mixed runners detected. For consistent measurements, "
            "consider running all backends in Docker."
        )

    # Always create study dir + manifest (LA-05)
    study_dir = create_study_dir(study.name, Path("results"))
    manifest = ManifestWriter(study, study_dir)

    wall_start = time.monotonic()
    is_single = len(study.experiments) == 1 and study.execution.n_cycles == 1

    if is_single:
        result_files, experiment_results, warnings = _run_in_process(
            study, manifest, study_dir, runner_specs=runner_specs
        )
    else:
        result_files, experiment_results, warnings = _run_via_runner(
            study, manifest, study_dir, runner_specs=runner_specs
        )

    wall_time = time.monotonic() - wall_start

    # Mark manifest as completed — only reached on success (SIGINT path calls
    # manifest.mark_interrupted() then sys.exit(130) before returning here).
    manifest.mark_study_completed()

    completed = sum(1 for r in experiment_results if r is not None)
    failed = len(experiment_results) - completed
    total_energy = sum(r.total_energy_j for r in experiment_results if r is not None)

    # study.experiments is already cycle-expanded by apply_cycles(), so len() is the true total
    n_cycles = study.execution.n_cycles
    unique_configs = len(study.experiments) // n_cycles if n_cycles > 0 else len(study.experiments)

    summary = StudySummary(
        total_experiments=len(study.experiments),
        completed=completed,
        failed=failed,
        total_wall_time_s=wall_time,
        total_energy_j=total_energy,
        unique_configurations=unique_configs,
        warnings=warnings,
    )

    measurement_protocol: dict[str, Any] = {
        "n_cycles": study.execution.n_cycles,
        "cycle_order": study.execution.cycle_order,
        "experiment_gap_seconds": study.execution.experiment_gap_seconds,
        "cycle_gap_seconds": study.execution.cycle_gap_seconds,
        "shuffle_seed": study.execution.shuffle_seed,
    }

    return StudyResult(
        experiments=[r for r in experiment_results if r is not None],
        name=study.name,
        study_design_hash=study.study_design_hash,
        measurement_protocol=measurement_protocol,
        result_files=result_files,
        summary=summary,
    )


def _run_in_process(
    study: StudyConfig,
    manifest: Any,
    study_dir: Path,
    runner_specs: Any = None,
) -> tuple[list[str], list[ExperimentResult | None], list[str]]:
    """Run a single experiment in-process or via DockerRunner directly.

    When runner_specs resolves the backend to "docker", uses DockerRunner directly
    (no subprocess spawning). Otherwise runs in-process via the backend.

    Errors from run_preflight() and harness.run(backend, config) propagate unchanged (PreFlightError,
    BackendError). Only result-saving errors are caught so a save failure does not
    discard a completed measurement.
    """
    from llenergymeasure.domain.experiment import compute_measurement_config_hash
    from llenergymeasure.study.runner import _save_and_record

    config = study.experiments[0]
    config_hash = compute_measurement_config_hash(config)
    cycle = 1

    # Pre-dispatch GPU memory residual check (MEAS-01, MEAS-02)
    # Mirrors the pattern used in StudyRunner._run_one() and _run_one_docker().
    from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

    check_gpu_memory_residual()

    # Check runner spec for this backend
    spec = runner_specs.get(config.backend) if runner_specs else None

    manifest.mark_running(config_hash, cycle)

    if spec is not None and spec.mode == "docker":
        # Docker path: dispatch to container directly (no subprocess)
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.utils.exceptions import DockerError

        image = spec.image if spec.image is not None else get_default_image(config.backend)
        from llenergymeasure.study.runner import _calculate_timeout

        docker_runner = DockerRunner(
            image=image,
            timeout=_calculate_timeout(config),
            source=spec.source,
            extra_mounts=spec.extra_mounts,
        )
        try:
            result = docker_runner.run(config)
        except DockerError as exc:
            # Convert to failure dict — manifest marks failed, study continues
            error_payload: dict[str, Any] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "config_hash": config_hash,
            }
            manifest.mark_failed(
                config_hash, cycle, error_payload["type"], error_payload["message"]
            )
            return [], [None], [error_payload["message"]]
    else:
        # Local in-process path — errors propagate naturally (PreFlightError, BackendError)
        from llenergymeasure.backends import get_backend
        from llenergymeasure.harness import MeasurementHarness
        from llenergymeasure.harness.preflight import run_preflight

        run_preflight(config)
        backend = get_backend(config.backend)
        harness = MeasurementHarness()
        gpu_indices = _resolve_gpu_indices(config)
        result = harness.run(backend, config, gpu_indices=gpu_indices)

    # Handle error payload returned from Docker container (exit 0 but wrote error JSON)
    if isinstance(result, dict) and "type" in result:
        error_type = result.get("type", "UnknownError")
        error_message = result.get("message", "")
        manifest.mark_failed(config_hash, cycle, error_type, error_message)
        return [], [None], [error_message]

    result_files: list[str] = []
    warnings: list[str] = []
    _save_and_record(result, study_dir, manifest, config_hash, cycle, result_files)

    return result_files, [result], warnings


def _run_via_runner(
    study: StudyConfig,
    manifest: Any,
    study_dir: Path,
    runner_specs: Any = None,
) -> tuple[list[str], list[ExperimentResult | None], list[str]]:
    """Delegate to StudyRunner for multi-experiment / multi-cycle runs."""
    from llenergymeasure.study.runner import StudyRunner

    runner = StudyRunner(study, manifest, study_dir, runner_specs=runner_specs)
    raw_results = runner.run()

    warnings: list[str] = []
    experiment_results: list[ExperimentResult | None] = []
    for r in raw_results:
        if isinstance(r, dict):
            warnings.append(r.get("message", "Unknown error"))
            experiment_results.append(None)
        else:
            experiment_results.append(r)

    return runner.result_files, experiment_results, warnings
