"""Pytest configuration and shared fixtures for v2.0 tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    RawProcessResult,
    StudyResult,
    Timestamps,
)
from llenergymeasure.domain.metrics import (
    ComputeMetrics,
    EnergyMetrics,
    InferenceMetrics,
)

_REPLAY_DIR = Path(__file__).parent / "fixtures" / "replay"

# ---------------------------------------------------------------------------
# Shared test constants - single source of truth for magic values
# ---------------------------------------------------------------------------

TEST_MODEL = "gpt2"
TEST_ENGINE = "pytorch"
TEST_EXPERIMENT_ID = "test-001"
TEST_CONFIG_HASH = "deadbeef12345678"
TEST_MEASUREMENT_HASH = "abc123def4567890"
TEST_POWER_MW = 200_000  # 200 W in milliwatts (pynvml convention)
TEST_POWER_W = 200.0

# Derived from model defaults - single source of truth for schema assertions
EXPERIMENT_SCHEMA_VERSION = ExperimentResult.model_fields["schema_version"].default
RAW_PROCESS_SCHEMA_VERSION = RawProcessResult.model_fields["schema_version"].default

_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_EPOCH_END = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def make_config(**overrides) -> ExperimentConfig:
    """Return a valid ExperimentConfig with sensible defaults.

    Tests override only what they care about.
    """
    defaults: dict = {
        "model": TEST_MODEL,
        "engine": TEST_ENGINE,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def make_result(**overrides) -> ExperimentResult:
    """Return a valid ExperimentResult with sensible defaults.

    Includes all required fields (measurement_config_hash,
    measurement_methodology, start_time, end_time) to prevent ValidationError.
    """
    defaults: dict = {
        "experiment_id": TEST_EXPERIMENT_ID,
        "measurement_config_hash": TEST_MEASUREMENT_HASH,
        "measurement_methodology": "total",
        "aggregation": AggregationMetadata(num_processes=1),
        "total_tokens": 1000,
        "total_energy_j": 10.0,
        "total_inference_time_sec": 5.0,
        "avg_tokens_per_second": 200.0,
        "avg_energy_per_token_j": 0.01,
        "total_flops": 1e9,
        "start_time": _EPOCH,
        "end_time": _EPOCH_END,
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


def make_study_result(**overrides) -> StudyResult:
    """Return a valid StudyResult with sensible defaults.

    Needed by CLI tests (Plan 03) and E2E tests (Plan 04).
    Tests override only what they care about.
    """
    from llenergymeasure.domain.experiment import StudySummary

    one_result = make_result()
    defaults: dict = {
        "study_name": "test-study",
        "experiments": [one_result],
        "summary": StudySummary(
            total_experiments=1,
            completed=1,
            failed=0,
            total_wall_time_s=5.0,
            total_energy_j=10.0,
        ),
        "result_files": [],
    }
    defaults.update(overrides)
    return StudyResult(**defaults)


def make_energy_metrics(**overrides) -> EnergyMetrics:
    """Return a valid EnergyMetrics with sensible defaults."""
    defaults: dict = {
        "total_energy_j": 10.0,
        "duration_sec": 5.0,
    }
    defaults.update(overrides)
    return EnergyMetrics(**defaults)


def make_inference_metrics(**overrides) -> InferenceMetrics:
    """Return a valid InferenceMetrics with sensible defaults."""
    defaults: dict = {
        "total_tokens": 500,
        "input_tokens": 100,
        "output_tokens": 400,
        "inference_time_sec": 10.0,
        "tokens_per_second": 50.0,
        "latency_per_token_ms": 2.0,
    }
    defaults.update(overrides)
    return InferenceMetrics(**defaults)


def make_compute_metrics(**overrides) -> ComputeMetrics:
    """Return a valid ComputeMetrics with sensible defaults."""
    defaults: dict = {
        "flops_total": 5e11,
    }
    defaults.update(overrides)
    return ComputeMetrics(**defaults)


def make_raw_process_result(**overrides) -> RawProcessResult:
    """Return a valid RawProcessResult with sensible defaults.

    Builds on make_energy_metrics, make_inference_metrics, and
    make_compute_metrics factories for nested fields.
    """
    defaults: dict = {
        "experiment_id": TEST_EXPERIMENT_ID,
        "process_index": 0,
        "gpu_id": 0,
        "model_name": TEST_MODEL,
        "timestamps": Timestamps.from_times(
            datetime(2026, 2, 26, 14, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 2, 26, 14, 0, 10, tzinfo=timezone.utc),
        ),
        "inference_metrics": make_inference_metrics(),
        "energy_metrics": make_energy_metrics(),
        "compute_metrics": make_compute_metrics(),
    }
    defaults.update(overrides)
    return RawProcessResult(**defaults)


def make_user_config(**overrides):
    """Return a minimal mock UserConfig for tests that need load_user_config.

    Uses real Pydantic models to avoid fragile anonymous-type hacks.
    """
    from llenergymeasure.config.user_config import UserConfig

    defaults: dict = {}
    defaults.update(overrides)
    return UserConfig(**defaults)


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return make_config()


@pytest.fixture
def sample_result() -> ExperimentResult:
    return make_result()


@pytest.fixture
def tmp_results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def replay_results() -> list[ExperimentResult]:
    """Load GPU-produced ExperimentResult fixtures from tests/fixtures/replay/.

    Returns an empty list when no fixtures exist (safe for GPU-free CI).
    Uses model_validate_json directly (not from_json) because replay fixtures
    are standalone JSON files without timeseries sidecars.
    """
    results = []
    if _REPLAY_DIR.is_dir():
        for json_file in sorted(_REPLAY_DIR.glob("*.json")):
            content = json_file.read_text(encoding="utf-8")
            results.append(ExperimentResult.model_validate_json(content))
    return results


# ---------------------------------------------------------------------------
# Autouse fixtures: module-level singleton cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_gpu_locks(monkeypatch):
    """Prevent real GPU advisory locks from interfering with tests.

    Tests that specifically exercise gpu_locks can override this by
    importing and calling the real functions directly.
    """
    monkeypatch.setattr("llenergymeasure.study.gpu_locks.acquire_gpu_locks", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_locks.release_gpu_locks", lambda *_a, **_kw: None
    )


@pytest.fixture(autouse=True)
def clear_baseline_cache():
    """Clear _baseline_cache before and after each test.

    Prevents baseline measurement state from leaking between tests, which is
    especially important when pytest-randomly changes execution order.
    """
    from llenergymeasure.harness.baseline import _baseline_cache

    _baseline_cache.clear()
    yield
    _baseline_cache.clear()


@pytest.fixture(autouse=True)
def reset_lru_caches():
    """Clear lru_cache / functools.cache decorated functions between tests.

    Only clears caches that are known to produce order-dependent results
    (i.e. caches that depend on host environment state).
    """
    try:
        from llenergymeasure.infra.runner_resolution import is_docker_available

        if hasattr(is_docker_available, "cache_clear"):
            is_docker_available.cache_clear()
    except ImportError:
        pass

    try:
        from llenergymeasure.infra.image_registry import get_cuda_major_version

        if hasattr(get_cuda_major_version, "cache_clear"):
            get_cuda_major_version.cache_clear()
    except ImportError:
        pass

    yield

    try:
        from llenergymeasure.infra.runner_resolution import is_docker_available

        if hasattr(is_docker_available, "cache_clear"):
            is_docker_available.cache_clear()
    except ImportError:
        pass

    try:
        from llenergymeasure.infra.image_registry import get_cuda_major_version

        if hasattr(get_cuda_major_version, "cache_clear"):
            get_cuda_major_version.cache_clear()
    except ImportError:
        pass
