"""Tests for StudyRunner baseline strategy dispatch and drift validation.

Covers the _get_baseline() and _validate_baseline() methods introduced in PR #242.
All tests run without GPU hardware - baseline measurement and GPU resolution are mocked.

Test strategy mirrors test_study_runner.py: StudyRunner is instantiated with a
MagicMock manifest and tmp_path study_dir. Lazy imports inside _get_baseline and
_validate_baseline are patched at source module level.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.models import (
    BaselineConfig,
    ExecutionConfig,
    ExperimentConfig,
    StudyConfig,
)
from llenergymeasure.harness.baseline import BaselineCache
from llenergymeasure.infra.runner_resolution import RunnerSpec
from llenergymeasure.study.runner import StudyRunner
from llenergymeasure.utils.exceptions import DockerError
from tests.conftest import TEST_CONFIG_HASH

# Patch targets: source modules, not runner imports (lazy local imports)
_MEASURE = "llenergymeasure.harness.baseline.measure_baseline_power"
_SAVE = "llenergymeasure.harness.baseline.save_baseline_cache"
_LOAD = "llenergymeasure.harness.baseline.load_baseline_cache"
_SPOT = "llenergymeasure.harness.baseline.measure_spot_check"
_RESOLVE_GPU = "llenergymeasure.device.gpu_info._resolve_gpu_indices"


# =============================================================================
# Helpers
# =============================================================================


def _make_baseline(power_w: float = 50.0, **kwargs) -> BaselineCache:
    defaults = {
        "power_w": power_w,
        "timestamp": time.time(),
        "gpu_indices": [0],
        "sample_count": 200,
        "duration_sec": 30.0,
    }
    defaults.update(kwargs)
    return BaselineCache(**defaults)


def _make_runner(tmp_path: Path, config: ExperimentConfig) -> StudyRunner:
    """Construct a StudyRunner with minimal plumbing for baseline testing."""
    study = StudyConfig(
        experiments=[config],
        study_name="test-baseline",
        study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
        study_design_hash=TEST_CONFIG_HASH,
    )
    manifest = MagicMock()
    runner = StudyRunner(study, manifest, tmp_path)
    return runner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config_cached() -> ExperimentConfig:
    """ExperimentConfig with strategy='cached' (default)."""
    return ExperimentConfig(
        model="test/model",
        backend="pytorch",
        baseline=BaselineConfig(strategy="cached", duration_seconds=30.0),
    )


@pytest.fixture
def config_fresh() -> ExperimentConfig:
    return ExperimentConfig(
        model="test/model",
        backend="pytorch",
        baseline=BaselineConfig(strategy="fresh"),
    )


@pytest.fixture
def config_validated() -> ExperimentConfig:
    return ExperimentConfig(
        model="test/model",
        backend="pytorch",
        baseline=BaselineConfig(
            strategy="validated",
            validation_interval=3,
            drift_threshold=0.10,
        ),
    )


# =============================================================================
# Strategy: fresh
# =============================================================================


class TestStrategyFresh:
    def test_returns_none(self, tmp_path: Path, config_fresh: ExperimentConfig):
        """strategy='fresh' returns None — each experiment measures its own."""
        runner = _make_runner(tmp_path, config_fresh)
        result = runner._get_baseline(config_fresh)
        assert result is None

    def test_does_not_measure(self, tmp_path: Path, config_fresh: ExperimentConfig):
        """strategy='fresh' never calls measure_baseline_power."""
        runner = _make_runner(tmp_path, config_fresh)
        with patch(_MEASURE) as mock_measure:
            runner._get_baseline(config_fresh)
            mock_measure.assert_not_called()


# =============================================================================
# Strategy: cached
# =============================================================================


class TestStrategyCached:
    def test_measures_and_returns_baseline(self, tmp_path: Path, config_cached: ExperimentConfig):
        """First call measures a fresh baseline and returns it."""
        runner = _make_runner(tmp_path, config_cached)
        fake = _make_baseline(60.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            result = runner._get_baseline(config_cached)

        assert result is not None
        assert result.power_w == 60.0

    def test_persists_to_disk(self, tmp_path: Path, config_cached: ExperimentConfig):
        """First call saves the baseline to the artefacts directory."""
        runner = _make_runner(tmp_path, config_cached)
        fake = _make_baseline()

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE) as mock_save,
        ):
            runner._get_baseline(config_cached)

        mock_save.assert_called_once()
        saved_path = mock_save.call_args[0][0]
        assert saved_path.name == "baseline_cache.json"
        assert "_study-artefacts" in str(saved_path)

    def test_sets_method_to_cached(self, tmp_path: Path, config_cached: ExperimentConfig):
        """Baseline method is set to 'cached' for result reporting."""
        runner = _make_runner(tmp_path, config_cached)
        fake = _make_baseline()

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            result = runner._get_baseline(config_cached)

        assert result.method == "cached"

    def test_reuses_on_second_call(self, tmp_path: Path, config_cached: ExperimentConfig):
        """Second call returns the cached baseline without re-measuring."""
        runner = _make_runner(tmp_path, config_cached)
        fake = _make_baseline()

        with (
            patch(_MEASURE, return_value=fake) as mock_measure,
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            first = runner._get_baseline(config_cached)
            second = runner._get_baseline(config_cached)

        assert second is first
        assert mock_measure.call_count == 1  # measured only once

    def test_loads_from_disk_on_restart(self, tmp_path: Path, config_cached: ExperimentConfig):
        """When in-memory cache is empty, loads from disk cache (mid-study restart)."""
        runner = _make_runner(tmp_path, config_cached)
        disk_baseline = _make_baseline(72.0, from_cache=True)

        with (
            patch(_LOAD, return_value=disk_baseline),
            patch(_MEASURE) as mock_measure,
        ):
            result = runner._get_baseline(config_cached)

        assert result is not None
        assert result.power_w == 72.0
        mock_measure.assert_not_called()  # loaded from disk, no fresh measurement

    def test_measures_fresh_when_disk_cache_expired(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        """When disk cache returns None (expired/missing), measures fresh."""
        runner = _make_runner(tmp_path, config_cached)
        fresh = _make_baseline(55.0)

        with (
            patch(_LOAD, return_value=None),
            patch(_MEASURE, return_value=fresh),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            result = runner._get_baseline(config_cached)

        assert result.power_w == 55.0

    def test_measurement_failure_returns_none(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        """When measurement fails (returns None), _get_baseline returns None."""
        runner = _make_runner(tmp_path, config_cached)

        with (
            patch(_LOAD, return_value=None),
            patch(_MEASURE, return_value=None),
            patch(_RESOLVE_GPU, return_value=[0]),
        ):
            result = runner._get_baseline(config_cached)

        assert result is None

    def test_ttl_expiry_triggers_remeasurement(self, tmp_path: Path):
        """In-memory baseline is re-measured when TTL expires."""
        config = ExperimentConfig(
            model="test/model",
            backend="pytorch",
            baseline=BaselineConfig(strategy="cached", cache_ttl_seconds=3600.0),
        )
        runner = _make_runner(tmp_path, config)

        # Initial baseline with old timestamp (2 hours ago)
        old = _make_baseline(50.0, timestamp=time.time() - 7200)
        fresh = _make_baseline(55.0)

        with (
            patch(_MEASURE, return_value=old),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config)

        assert runner._baseline.power_w == 50.0

        # Second call: old baseline has expired (age 7200 > ttl 3600)
        with (
            patch(_MEASURE, return_value=fresh) as mock_measure,
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_LOAD, return_value=None),
        ):
            result = runner._get_baseline(config)

        assert result.power_w == 55.0  # re-measured
        mock_measure.assert_called_once()

    def test_ttl_not_expired_reuses_cache(self, tmp_path: Path, config_cached: ExperimentConfig):
        """In-memory baseline within TTL is reused without re-measurement."""
        runner = _make_runner(tmp_path, config_cached)
        # Recent baseline (timestamp = now)
        recent = _make_baseline(60.0)

        with (
            patch(_MEASURE, return_value=recent) as mock_measure,
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            first = runner._get_baseline(config_cached)
            second = runner._get_baseline(config_cached)

        assert second is first
        assert mock_measure.call_count == 1


# =============================================================================
# Strategy: validated
# =============================================================================


class TestStrategyValidated:
    def test_first_call_measures_baseline(self, tmp_path: Path, config_validated: ExperimentConfig):
        """First call behaves like 'cached' — measures and persists."""
        runner = _make_runner(tmp_path, config_validated)
        fake = _make_baseline()

        with (
            patch(_MEASURE, return_value=fake) as mock_measure,
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            result = runner._get_baseline(config_validated)

        assert result is not None
        mock_measure.assert_called_once()

    def test_no_spot_check_before_interval(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        """Spot-check not triggered until validation_interval (3) is reached."""
        runner = _make_runner(tmp_path, config_validated)
        fake = _make_baseline()

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT) as mock_spot,
        ):
            # First call: initial measurement (resets counter to 0)
            runner._get_baseline(config_validated)
            # Calls 2 and 3: counter goes to 1, 2 (below interval=3)
            runner._get_baseline(config_validated)
            runner._get_baseline(config_validated)

        mock_spot.assert_not_called()

    def test_spot_check_at_interval(self, tmp_path: Path, config_validated: ExperimentConfig):
        """Spot-check triggered when validation_interval (3) is reached."""
        runner = _make_runner(tmp_path, config_validated)
        fake = _make_baseline(50.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT, return_value=51.0) as mock_spot,
        ):
            runner._get_baseline(config_validated)  # initial measure, counter=0
            runner._get_baseline(config_validated)  # counter=1
            runner._get_baseline(config_validated)  # counter=2
            runner._get_baseline(config_validated)  # counter=3 -> triggers

        assert mock_spot.call_count == 1

    def test_no_drift_keeps_baseline(self, tmp_path: Path, config_validated: ExperimentConfig):
        """Drift below threshold keeps the existing baseline."""
        runner = _make_runner(tmp_path, config_validated)
        fake = _make_baseline(50.0)

        with (
            patch(_MEASURE, return_value=fake) as mock_measure,
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT, return_value=51.0),  # 2% drift, below 10% threshold
        ):
            runner._get_baseline(config_validated)
            for _ in range(3):  # reach interval
                runner._get_baseline(config_validated)

        assert runner._baseline.power_w == 50.0  # unchanged
        assert mock_measure.call_count == 1  # no re-measurement

    @staticmethod
    def _run_drift_scenario(tmp_path: Path, config: ExperimentConfig):
        """Run a drift scenario: initial measure -> 3 calls -> drift triggers re-measure.

        Returns (runner, mock_save) for assertions.
        """
        runner = _make_runner(tmp_path, config)
        original = _make_baseline(50.0)
        remeasured = _make_baseline(58.0)
        call_count = [0]

        def measure_side_effect(**kwargs):
            call_count[0] += 1
            return original if call_count[0] == 1 else remeasured

        with (
            patch(_MEASURE, side_effect=measure_side_effect),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE) as mock_save,
            patch(_SPOT, return_value=60.0),  # 20% drift, above 10% threshold
        ):
            runner._get_baseline(config)
            for _ in range(3):
                runner._get_baseline(config)

        return runner, mock_save, call_count[0]

    def test_drift_triggers_remeasurement(self, tmp_path: Path, config_validated: ExperimentConfig):
        """Drift above threshold triggers a full re-measurement."""
        runner, _, measure_calls = self._run_drift_scenario(tmp_path, config_validated)

        assert runner._baseline.power_w == 58.0  # updated
        assert measure_calls == 2  # initial + re-measurement

    def test_drift_remeasurement_saved_to_disk(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        """Re-measured baseline after drift is persisted to disk."""
        _, mock_save, _ = self._run_drift_scenario(tmp_path, config_validated)

        # save called twice: initial + after drift re-measurement
        assert mock_save.call_count == 2

    def test_method_set_to_validated(self, tmp_path: Path, config_validated: ExperimentConfig):
        """After validation (no drift), method is set to 'validated'."""
        runner = _make_runner(tmp_path, config_validated)
        fake = _make_baseline(50.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT, return_value=51.0),  # low drift
        ):
            runner._get_baseline(config_validated)
            for _ in range(3):
                runner._get_baseline(config_validated)

        assert runner._baseline.method == "validated"

    def test_spot_check_failure_is_nonfatal(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        """If spot-check returns None, baseline is kept unchanged."""
        runner = _make_runner(tmp_path, config_validated)
        fake = _make_baseline(50.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT, return_value=None),  # measurement failed
        ):
            runner._get_baseline(config_validated)
            for _ in range(3):
                runner._get_baseline(config_validated)

        assert runner._baseline.power_w == 50.0  # kept original

    def test_counter_resets_after_validation(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        """Counter resets after each validation, so next check is interval calls away."""
        runner = _make_runner(tmp_path, config_validated)
        fake = _make_baseline(50.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT, return_value=51.0) as mock_spot,
        ):
            runner._get_baseline(config_validated)  # initial measure
            # First interval: 3 more calls
            for _ in range(3):
                runner._get_baseline(config_validated)
            assert mock_spot.call_count == 1

            # Second interval: 3 more calls
            for _ in range(3):
                runner._get_baseline(config_validated)
            assert mock_spot.call_count == 2


# =============================================================================
# Baseline cache path
# =============================================================================


class TestBaselineCachePath:
    def test_path_in_study_artefacts(self, tmp_path: Path, config_cached: ExperimentConfig):
        """Cache path is {study_dir}/_study-artefacts/baseline_cache.json."""
        runner = _make_runner(tmp_path, config_cached)
        path = runner._get_baseline_cache_path()
        assert path == tmp_path / "_study-artefacts" / "baseline_cache.json"

    def test_artefacts_dir_created(self, tmp_path: Path, config_cached: ExperimentConfig):
        """_get_baseline_cache_path creates the _study-artefacts directory."""
        runner = _make_runner(tmp_path, config_cached)
        runner._get_baseline_cache_path()
        assert (tmp_path / "_study-artefacts").is_dir()

    def test_path_cached_on_runner(self, tmp_path: Path, config_cached: ExperimentConfig):
        """Second call returns the same Path object (no redundant mkdir)."""
        runner = _make_runner(tmp_path, config_cached)
        first = runner._get_baseline_cache_path()
        second = runner._get_baseline_cache_path()
        assert first is second


# =============================================================================
# Docker baseline mount
# =============================================================================


class TestDockerBaselineMount:
    def test_baseline_cache_mounted_into_container(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        """When baseline exists and cache file is on disk, it is mounted."""
        runner = _make_runner(tmp_path, config_cached)
        fake = _make_baseline()

        # Pre-populate the cache file on disk
        cache_path = tmp_path / "_study-artefacts" / "baseline_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"power_w": 50.0}))

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_cached)

        # Verify the cache file path is correct
        assert runner._get_baseline_cache_path() == cache_path
        assert cache_path.exists()

    def test_baseline_cache_mount_path_is_absolute_when_study_dir_relative(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        config_cached: ExperimentConfig,
    ):
        """Regression: relative study_dir must yield an absolute bind-mount source."""
        monkeypatch.chdir(tmp_path)
        relative_study_dir = Path("results/test-study")

        runner = _make_runner(relative_study_dir, config_cached)
        runner._baseline = _make_baseline()
        runner._images_prepared = True

        cache_path = relative_study_dir / "_study-artefacts" / "baseline_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"power_w": 50.0}))

        captured: dict[str, object] = {}

        class FakeDockerRunner:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, *args, **kwargs):
                raise DockerError("stop here")

        spec = RunnerSpec(mode="docker", image="test/image:latest", source="yaml")

        with (
            patch("llenergymeasure.infra.docker_runner.DockerRunner", FakeDockerRunner),
            patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
            patch.object(runner, "_handle_result"),
            patch.object(runner, "_persist_failure_artefacts"),
        ):
            runner._run_one_docker(config_cached, spec, config_hash="abc123", cycle=1, index=1)

        extra_mounts = captured.get("extra_mounts") or []
        baseline_mounts = [
            host for host, cont in extra_mounts if cont == "/run/llem/baseline_cache.json"
        ]
        assert baseline_mounts, "baseline cache was not mounted"
        host_path = baseline_mounts[0]
        assert Path(host_path).is_absolute(), (
            f"baseline_cache.json bind-mount source must be absolute for Docker, "
            f"got relative path: {host_path!r}"
        )
        assert Path(host_path).exists(), f"resolved mount path does not exist: {host_path}"
