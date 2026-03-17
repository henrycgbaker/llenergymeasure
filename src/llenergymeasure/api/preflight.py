"""Pre-flight validation module.

Runs before any GPU allocation or model loading. Collects all failures into a
single PreFlightError so the user sees all problems at once.

Boundary:
    Pydantic handles schema validation (types, enums, missing fields).
    Pre-flight handles runtime checks: backend installed? model accessible? CUDA available?
"""

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.exceptions import PreFlightError

if TYPE_CHECKING:
    from llenergymeasure.config.user_config import UserRunnersConfig

logger = logging.getLogger(__name__)

# Map from backend name to the package that provides it.
_BACKEND_PACKAGES: dict[str, str] = {
    "pytorch": "transformers",
    "vllm": "vllm",
    "tensorrt": "tensorrt_llm",
}


# ---------------------------------------------------------------------------
# Internal check helpers
# ---------------------------------------------------------------------------


def _check_cuda_available() -> bool:
    """Return True if CUDA is available via torch.

    Uses importlib.util.find_spec() first to avoid importing torch when it is
    not installed (heavy module init).
    """
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _check_backend_installed(backend: str) -> bool:
    """Return True if the package that provides *backend* is importable."""
    package = _BACKEND_PACKAGES.get(backend)
    if package is None:
        # Unknown backend — Pydantic already blocked invalid values; treat as missing.
        return False
    return importlib.util.find_spec(package) is not None


def _check_model_accessible(model_id: str) -> str | None:
    """Check whether *model_id* is reachable.

    Returns an error string if a definitive failure is detected, None otherwise
    (including when we cannot determine reachability).
    """
    # Local path — starts with /, ./, or ~
    if model_id.startswith("/") or model_id.startswith("./") or model_id.startswith("~"):
        path = Path(model_id).expanduser()
        if not path.exists():
            return f"{model_id} not found — path does not exist"
        return None

    # Hub model — use huggingface_hub if available
    if importlib.util.find_spec("huggingface_hub") is None:
        return None  # Cannot check — skip rather than block

    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
        return None  # Accessible
    except Exception as exc:
        exc_str = str(exc)
        if "401" in exc_str or "403" in exc_str or "gated" in exc_str.lower():
            return f"{model_id} gated model — no HF_TOKEN → export HF_TOKEN=<your_token>"
        if "404" in exc_str or "not found" in exc_str.lower():
            return f"{model_id} not found on HuggingFace Hub"
        # Network error, timeout, etc. — don't block
        logger.debug("Model accessibility check skipped (network error): %s", exc)
        return None


def _warn_if_persistence_mode_off(gpu_indices: list[int] | None = None) -> None:
    """Log a warning if GPU persistence mode is disabled on any specified GPU.

    Checks all GPUs in gpu_indices (defaults to [0] when None). Warns once if
    any GPU has persistence mode off.

    Never raises — always wrapped in a broad except.

    Args:
        gpu_indices: GPU device indices to check. Defaults to [0] when None.
    """
    indices = gpu_indices if gpu_indices is not None else [0]
    try:
        import pynvml

        from llenergymeasure.core.gpu_info import nvml_context

        with nvml_context():
            for idx in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
                if mode == pynvml.NVML_FEATURE_DISABLED:
                    logger.warning(
                        "GPU persistence mode is off on GPU %d. First experiment may have higher "
                        "latency. Enable: sudo nvidia-smi -pm 1",
                        idx,
                    )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_preflight(config: ExperimentConfig) -> None:
    """Run all pre-flight checks for *config*.

    Collects every failure into a single ``PreFlightError`` so the user sees
    all problems at once. Raises nothing on success.

    Raises:
        PreFlightError: One or more checks failed.
    """
    failures: list[str] = []

    # Check 1: CUDA available
    if not _check_cuda_available():
        failures.append("CUDA not available — is a GPU present and CUDA installed?")

    # Check 2: Backend installed
    if not _check_backend_installed(config.backend):
        package = _BACKEND_PACKAGES.get(config.backend, config.backend)
        failures.append(
            f"{config.backend} not installed — pip install llenergymeasure[{config.backend}]"
            f" (missing: {package})"
        )

    # Check 3: Model accessible
    model_error = _check_model_accessible(config.model)
    if model_error is not None:
        failures.append(model_error)

    # Check 4: Backend config validation (hardware x config cross-checks)
    try:
        from llenergymeasure.core.backends import get_backend

        backend = get_backend(config.backend)
        backend_errors = backend.validate_config(config)
        failures.extend(backend_errors)
    except Exception:
        pass  # get_backend may fail if backend not installed — already caught by Check 2

    if failures:
        n = len(failures)
        lines = "\n".join(f"  \u2717 {f}" for f in failures)
        raise PreFlightError(f"Pre-flight failed: {n} issue(s) found\n{lines}")

    # Non-blocking warning (after all checks pass)
    _warn_if_persistence_mode_off()


def run_study_preflight(
    study: StudyConfig,
    skip_preflight: bool = False,
    yaml_runners: dict[str, str] | None = None,
    user_config: "UserRunnersConfig | None" = None,
) -> None:
    """Pre-flight checks for a study configuration.

    Single-backend studies pass through — per-experiment pre-flight runs later
    in the subprocess. Multi-backend studies auto-elevate to Docker when Docker
    is available (DOCK-05); raise PreFlightError otherwise.

    When any experiment in the study will use a Docker runner, runs Docker
    pre-flight checks (GPU visibility, CUDA/driver compat) unless skipped.

    Args:
        study: Resolved StudyConfig.
        skip_preflight: Skip Docker pre-flight checks. The effective skip value
            is ``skip_preflight OR study.execution.skip_preflight`` — CLI flag
            takes priority, then YAML config.
        yaml_runners: Runner config from the study YAML ``runners:`` section.
            Forwarded to ``resolve_study_runners()`` so pre-flight uses the same
            runner resolution as the actual dispatch path.
        user_config: Loaded UserRunnersConfig. Forwarded to
            ``resolve_study_runners()`` to match actual dispatch precedence.

    Raises:
        PreFlightError: Multi-backend study and Docker is not available.
        DockerPreFlightError: Docker pre-flight check failed (inherits PreFlightError).
    """
    from llenergymeasure.infra.runner_resolution import is_docker_available, resolve_study_runners

    backends = {exp.backend for exp in study.experiments}
    if len(backends) > 1:
        backend_list = ", ".join(sorted(backends))
        if is_docker_available():
            logger.info(
                "Multi-backend study detected (%s). Auto-elevating all backends to Docker "
                "for isolation.",
                backend_list,
            )
        else:
            raise PreFlightError(
                f"Multi-backend study requires Docker isolation. "
                f"Found backends: {backend_list}. "
                "Install Docker + NVIDIA Container Toolkit, or use a single backend."
            )

    # Docker pre-flight: run once if any backend resolves to a Docker runner.
    # Effective skip = CLI flag (skip_preflight param) OR YAML config value.
    effective_skip = skip_preflight or getattr(study.execution, "skip_preflight", False)
    runner_specs = resolve_study_runners(
        list(backends), yaml_runners=yaml_runners, user_config=user_config
    )
    if any(spec.mode == "docker" for spec in runner_specs.values()):
        from llenergymeasure.infra.docker_preflight import run_docker_preflight

        run_docker_preflight(skip=effective_skip)
