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
from llenergymeasure.config.ssot import CONTAINER_EXCHANGE_DIR
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
_RUN_BASELINE_CONTAINER = "llenergymeasure.study.baseline_container.run_baseline_container"

# Cache key for purely-local runner targets — all existing tests construct
# a StudyRunner with runner_specs=None, which maps to "local".
_LOCAL_KEY = "local"


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
        task={"model": "test/model"},
        engine="transformers",
        measurement={"baseline": BaselineConfig(strategy="cached", duration_seconds=30.0)},
    )


@pytest.fixture
def config_fresh() -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "test/model"},
        engine="transformers",
        measurement={"baseline": BaselineConfig(strategy="fresh")},
    )


@pytest.fixture
def config_validated() -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "test/model"},
        measurement={
            "baseline": BaselineConfig(
                strategy="validated",
                validation_interval=3,
                drift_threshold=0.10,
            )
        },
        engine="transformers",
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
        assert saved_path.name == "baseline_cache_local.json"
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

        # The runner now checks disk_path.exists() before calling load — create
        # a real file on disk so the load path is taken.
        disk_path = runner._get_baseline_cache_path(_LOCAL_KEY)
        disk_path.write_text("{}", encoding="utf-8")

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
            task={"model": "test/model"},
            measurement={"baseline": BaselineConfig(strategy="cached", cache_ttl_seconds=3600.0)},
            engine="transformers",
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

        assert runner._baselines[_LOCAL_KEY].power_w == 50.0

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

        assert runner._baselines[_LOCAL_KEY].power_w == 50.0  # unchanged
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

        assert runner._baselines[_LOCAL_KEY].power_w == 58.0  # updated
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

        assert runner._baselines[_LOCAL_KEY].method == "validated"

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

        assert runner._baselines[_LOCAL_KEY].power_w == 50.0  # kept original

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
        """Cache path is {study_dir}/_study-artefacts/baseline_cache_{key}.json."""
        runner = _make_runner(tmp_path, config_cached)
        path = runner._get_baseline_cache_path(_LOCAL_KEY)
        assert path == tmp_path / "_study-artefacts" / "baseline_cache_local.json"

    def test_artefacts_dir_created(self, tmp_path: Path, config_cached: ExperimentConfig):
        """_get_baseline_cache_path creates the _study-artefacts directory."""
        runner = _make_runner(tmp_path, config_cached)
        runner._get_baseline_cache_path(_LOCAL_KEY)
        assert (tmp_path / "_study-artefacts").is_dir()


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

        # Pre-populate the cache file on disk under the local cache key.
        cache_path = tmp_path / "_study-artefacts" / "baseline_cache_local.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"power_w": 50.0}))

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_cached)

        # Verify the cache file path is correct
        assert runner._get_baseline_cache_path(_LOCAL_KEY) == cache_path
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

        # Construct runner with a docker spec registered for pytorch so cache_key
        # resolves to the docker image key (what _run_one_docker will see).
        study = StudyConfig(
            experiments=[config_cached],
            study_name="test-baseline",
            study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
            study_design_hash=TEST_CONFIG_HASH,
        )
        manifest = MagicMock()
        spec = RunnerSpec(mode="docker", image="test/image:latest", source="yaml")
        runner = StudyRunner(
            study,
            manifest,
            relative_study_dir,
            runner_specs={"transformers": spec},
        )

        docker_key = runner._baseline_cache_key(config_cached)
        runner._baselines[docker_key] = _make_baseline()
        runner._images_prepared = True

        cache_path = relative_study_dir / "_study-artefacts" / f"baseline_cache_{docker_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"power_w": 50.0}))

        captured: dict[str, object] = {}

        class FakeDockerRunner:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, *args, **kwargs):
                raise DockerError("stop here")

        with (
            patch("llenergymeasure.infra.docker_runner.DockerRunner", FakeDockerRunner),
            patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
            patch.object(runner, "_handle_result"),
            patch.object(runner, "_persist_failure_artefacts"),
        ):
            runner._run_one_docker(config_cached, spec, config_hash="abc123", cycle=1, index=1)

        extra_mounts = captured.get("extra_mounts") or []
        baseline_container_path = f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json"
        baseline_mounts = [host for host, cont in extra_mounts if cont == baseline_container_path]
        assert baseline_mounts, "baseline cache was not mounted"
        host_path = baseline_mounts[0]
        assert Path(host_path).is_absolute(), (
            f"baseline_cache.json bind-mount source must be absolute for Docker, "
            f"got relative path: {host_path!r}"
        )
        assert Path(host_path).exists(), f"resolved mount path does not exist: {host_path}"


# =============================================================================
# Progress events (host-side baseline lifecycle)
# =============================================================================


def _make_runner_with_progress(
    tmp_path: Path,
    config: ExperimentConfig,
    runner_specs: dict | None = None,
) -> tuple[StudyRunner, MagicMock]:
    """Construct a StudyRunner with a MagicMock progress callback attached."""
    study = StudyConfig(
        experiments=[config],
        study_name="test-baseline-progress",
        study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
        study_design_hash=TEST_CONFIG_HASH,
    )
    manifest = MagicMock()
    progress = MagicMock()
    runner = StudyRunner(
        study,
        manifest,
        tmp_path,
        progress=progress,
        runner_specs=runner_specs,
    )
    return runner, progress


def _baseline_step_events(progress: MagicMock) -> list[tuple[str, tuple, dict]]:
    """Return all on_step_* calls targeting the 'baseline' step, in order."""
    out: list[tuple[str, tuple, dict]] = []
    for name, args, kwargs in progress.mock_calls:
        if name.startswith("on_step_") and args and args[0] == "baseline":
            out.append((name, args, kwargs))
    return out


class TestBaselineHostProgressEvents:
    """Verify host-side baseline events fire for cached/validated strategies."""

    def test_local_fresh_measure_emits_calibrating_start_and_done(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        runner, progress = _make_runner_with_progress(tmp_path, config_cached)
        fake = _make_baseline(60.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_cached)

        events = _baseline_step_events(progress)
        names = [e[0] for e in events]
        assert "on_step_start" in names
        assert "on_step_done" in names
        # Fresh measurement uses "Calibrating" verb + "sampling idle GPU draw"
        # detail. The label was refactored away from "Measuring ... idle power
        # (30s)" to keep the parent line concise while the sub-bullets carry
        # the breakdown.
        start_event = next(e for e in events if e[0] == "on_step_start")
        assert start_event[1][1] == "Calibrating"
        assert start_event[1][2] == "sampling idle GPU draw"

    def test_disk_load_emits_loading_event_when_file_exists(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        runner, progress = _make_runner_with_progress(tmp_path, config_cached)
        loaded = _make_baseline(72.0, from_cache=True, method="cached")

        disk_path = runner._get_baseline_cache_path(_LOCAL_KEY)
        disk_path.write_text("{}", encoding="utf-8")

        with (
            patch(_LOAD, return_value=loaded),
            patch(_MEASURE) as mock_measure,
        ):
            runner._get_baseline(config_cached)

        events = _baseline_step_events(progress)
        start_events = [e for e in events if e[0] == "on_step_start"]
        assert start_events, "expected at least one on_step_start for baseline"
        # First start uses the Loading verb
        assert start_events[0][1][1] == "Loading"
        mock_measure.assert_not_called()

    def test_disk_load_silent_when_no_cache_file(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        runner, progress = _make_runner_with_progress(tmp_path, config_cached)
        fake = _make_baseline(55.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_cached)

        events = _baseline_step_events(progress)
        verbs = [e[1][1] for e in events if e[0] == "on_step_start"]
        assert "Loading" not in verbs
        assert "Calibrating" in verbs

    def test_disk_load_sets_method_for_backward_compat(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        """Old cache files with method=None get strategy-default applied on load."""
        runner, _ = _make_runner_with_progress(tmp_path, config_cached)
        loaded = _make_baseline(72.0, from_cache=True, method=None)

        disk_path = runner._get_baseline_cache_path(_LOCAL_KEY)
        disk_path.write_text("{}", encoding="utf-8")

        with patch(_LOAD, return_value=loaded):
            runner._get_baseline(config_cached)

        assert runner._baselines[_LOCAL_KEY].method == "cached"

    def test_in_memory_hit_emits_reusing_event(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        runner, progress = _make_runner_with_progress(tmp_path, config_cached)
        fake = _make_baseline(60.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_cached)
            progress.reset_mock()
            runner._get_baseline(config_cached)  # second call: in-memory hit

        events = _baseline_step_events(progress)
        start_events = [e for e in events if e[0] == "on_step_start"]
        assert start_events, "expected a 'Reusing' on_step_start on the second call"
        assert start_events[0][1][1] == "Reusing"

    def test_fresh_strategy_emits_no_host_events(
        self, tmp_path: Path, config_fresh: ExperimentConfig
    ):
        runner, progress = _make_runner_with_progress(tmp_path, config_fresh)
        runner._get_baseline(config_fresh)

        events = _baseline_step_events(progress)
        assert events == []  # host emits nothing for fresh strategy

    def test_measure_failure_emits_failure_substep(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        """When _measure_baseline returns None (e.g. baseline container
        dispatch crashed because the image is stale), the host must freeze
        the dangling live substep with a failure text before on_step_done —
        otherwise the UI silently shows ✓ and users mistake it for a
        successful measurement.
        """
        runner, progress = _make_runner_with_progress(tmp_path, config_cached)

        with (
            patch(_MEASURE, return_value=None),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE) as mock_save,
        ):
            result = runner._get_baseline(config_cached)

        assert result is None
        mock_save.assert_not_called()  # nothing to persist when measurement failed

        # Failure surfaces via on_substep_done (freezes any live substep with
        # the failure text) — never as on_substep which would leave the
        # dangling "launching ..." heartbeat alive beside it.
        done_calls = [
            call
            for call in progress.mock_calls
            if call[0] == "on_substep_done" and call[1] and call[1][0] == "baseline"
        ]
        assert done_calls, "baseline failure must surface as on_substep_done"
        # Final text is passed via kwarg (text=...) in runner.py
        fail_texts = [
            call.kwargs.get("text") or (call.args[1] if len(call.args) > 1 else "")
            for call in done_calls
        ]
        assert any("failed" in (t or "").lower() for t in fail_texts)

        events = _baseline_step_events(progress)
        names = [e[0] for e in events]
        # on_step_done still fires (host UI gets a terminal event) even though
        # the measurement itself failed — the substep carries the "why".
        assert "on_step_done" in names

    def test_docker_fresh_emits_live_stage_substeps(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        """Streamed stage markers from run_baseline_container drive
        ``on_substep_start`` / ``on_substep_done`` pairs under the active
        baseline step, producing heartbeating sub-bullets that animate while
        each stage is in flight and freeze with a dim tick on completion.

        This is the regression for "why did a 30s measurement take 37s?" —
        the user now sees a live breakdown of launching container → CUDA init
        → sampling → teardown instead of one opaque parent step.
        """
        spec = RunnerSpec(mode="docker", image="test/vllm:v0", source="yaml")
        runner, progress = _make_runner_with_progress(
            tmp_path, config_cached, runner_specs={"transformers": spec}
        )

        fake_cache = _make_baseline(
            power_w=42.68,
            sample_count=289,
            duration_sec=30.00,
            gpu_indices=[0],
        )

        captured_on_stage: list = []

        def _fake_run_container(*_args, on_stage=None, **_kwargs):
            captured_on_stage.append(on_stage)
            if on_stage is not None:
                on_stage("container_ready", 2.5, {})
                on_stage("cuda_primed", 4.1, {})
                on_stage("sampling_started", 4.2, {"duration": "30.0"})
                on_stage(
                    "sampling_done",
                    34.3,
                    {"power_w": "42.68", "samples": "289", "duration": "30.00"},
                )
                on_stage("result_written", 34.4, {})
            return fake_cache

        with (
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_RUN_BASELINE_CONTAINER, side_effect=_fake_run_container),
        ):
            result = runner._get_baseline(config_cached)

        assert result is fake_cache
        assert captured_on_stage and captured_on_stage[0] is not None

        # Helper to pull the text out of a callback record regardless of
        # whether it was passed positionally or as a kwarg.
        def _text(call) -> str:
            if len(call.args) > 1 and call.args[1] is not None:
                return call.args[1]
            return call.kwargs.get("text") or ""

        baseline_calls = [
            c
            for c in progress.mock_calls
            if c[0] in ("on_substep_start", "on_substep_done")
            and c.args
            and c.args[0] == "baseline"
        ]
        starts = [c for c in baseline_calls if c[0] == "on_substep_start"]
        dones = [c for c in baseline_calls if c[0] == "on_substep_done"]

        start_texts = [_text(c) for c in starts]
        done_texts = [_text(c) for c in dones]

        # 1. The pre-dispatch substep name uses the image tag (user asked
        #    for "the actual image/container name w/ slug") and explicitly
        #    says "baseline container" so users know this is a short-lived
        #    dedicated container, not the experiment one.
        assert any(
            "launching" in t and "test/vllm:v0" in t and "baseline container" in t
            for t in start_texts
        ), start_texts

        # 2. Stage lifecycle: launching → ready → CUDA init → primed →
        #    sampling → sampled → writing/teardown → cached/torn down.
        assert any("baseline container ready" in t and "test/vllm:v0" in t for t in done_texts), (
            done_texts
        )
        assert any("initialising CUDA runtime" in t for t in start_texts)
        assert any("CUDA runtime primed" in t for t in done_texts)
        assert any("sampling idle GPU draw" in t and "30s" in t for t in start_texts)
        assert any("sampled idle GPU draw" in t and "42.68" in t and "289" in t for t in done_texts)
        assert any(
            "writing baseline result" in t and "tearing down" in t and "test/vllm:v0" in t
            for t in start_texts
        )
        assert any(
            "baseline result cached" in t and "torn down" in t and "test/vllm:v0" in t
            for t in done_texts
        )

        # 3. No retroactive "container launch + teardown" substeps — those
        #    were the pre-streaming labels from before stage markers existed.
        all_texts = start_texts + done_texts
        assert not any("container launch + teardown" in t for t in all_texts)

        # 4. Start/done pairs are balanced (4 starts, 4 dones: launching,
        #    CUDA init, sampling, teardown). This guarantees the UI never
        #    has two live substeps animating at once.
        assert len(starts) == len(dones) == 4, (start_texts, done_texts)

    def test_validated_spot_check_no_drift_emits_validating(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        runner, progress = _make_runner_with_progress(tmp_path, config_validated)
        fake = _make_baseline(50.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT, return_value=51.0),  # small drift, within threshold
        ):
            runner._get_baseline(config_validated)  # initial
            for _ in range(3):
                runner._get_baseline(config_validated)

        events = _baseline_step_events(progress)
        validating_starts = [
            e for e in events if e[0] == "on_step_start" and e[1][1] == "Validating"
        ]
        assert len(validating_starts) == 1
        assert runner._baselines[_LOCAL_KEY].method == "validated"

    def test_validated_drift_emits_validating_then_update_then_done(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        runner, progress = _make_runner_with_progress(tmp_path, config_validated)
        original = _make_baseline(50.0)
        remeasured = _make_baseline(58.0)
        call_count = [0]

        def measure_side_effect(**_kwargs):
            call_count[0] += 1
            return original if call_count[0] == 1 else remeasured

        with (
            patch(_MEASURE, side_effect=measure_side_effect),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_SPOT, return_value=60.0),  # 20% drift, above 10% threshold
        ):
            runner._get_baseline(config_validated)
            for _ in range(3):
                runner._get_baseline(config_validated)

        # Look for exactly one Validating start, one update containing
        # "re-measuring", and a matching done.
        validating_starts = [
            call
            for call in progress.mock_calls
            if call[0] == "on_step_start"
            and call.args
            and call.args[0] == "baseline"
            and call.args[1] == "Validating"
        ]
        assert len(validating_starts) == 1

        step_updates = [
            call
            for call in progress.mock_calls
            if call[0] == "on_step_update"
            and call.args
            and call.args[0] == "baseline"
            and "re-measuring" in str(call.args[1])
        ]
        assert len(step_updates) == 1
        assert runner._baselines[_LOCAL_KEY].method == "validated"

    def test_progress_none_is_nonfatal(self, tmp_path: Path, config_cached: ExperimentConfig):
        """StudyRunner with progress=None must not raise AttributeError."""
        runner = _make_runner(tmp_path, config_cached)
        assert runner._progress is None
        fake = _make_baseline(60.0)

        with (
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            # Any strategy path — just verify no AttributeError is raised.
            runner._get_baseline(config_cached)
            runner._get_baseline(config_cached)


# =============================================================================
# Cache key sanitisation and per-target separation
# =============================================================================


class TestBaselineCacheKey:
    def test_cache_key_local_for_no_specs(self, tmp_path: Path, config_cached: ExperimentConfig):
        runner = _make_runner(tmp_path, config_cached)
        assert runner._baseline_cache_key(config_cached) == "local"

    def test_cache_key_local_for_local_spec(self, tmp_path: Path, config_cached: ExperimentConfig):
        spec = RunnerSpec(mode="local", image=None, source="yaml")
        runner, _ = _make_runner_with_progress(
            tmp_path, config_cached, runner_specs={"transformers": spec}
        )
        assert runner._baseline_cache_key(config_cached) == "local"

    def test_cache_key_image_sanitisation(self, tmp_path: Path, config_cached: ExperimentConfig):
        spec = RunnerSpec(
            mode="docker",
            image="ghcr.io/foo/bar:v1@sha256:abc",
            source="yaml",
        )
        runner, _ = _make_runner_with_progress(
            tmp_path, config_cached, runner_specs={"transformers": spec}
        )
        key = runner._baseline_cache_key(config_cached)
        # Key must contain no chars that would break either a filesystem path
        # or a Docker bind-mount source string (`:` is the mount-mode separator).
        assert key.startswith("image_")
        assert ":" not in key
        assert "/" not in key
        assert "@" not in key

    def test_cache_key_image_length_clipped(self, tmp_path: Path, config_cached: ExperimentConfig):
        long_image = "a" * 500
        spec = RunnerSpec(mode="docker", image=long_image, source="yaml")
        runner, _ = _make_runner_with_progress(
            tmp_path, config_cached, runner_specs={"transformers": spec}
        )
        key = runner._baseline_cache_key(config_cached)
        sanitized = key.removeprefix("image_")
        assert len(sanitized) <= 128

    def test_disk_path_per_cache_key(self, tmp_path: Path, config_cached: ExperimentConfig):
        runner = _make_runner(tmp_path, config_cached)
        p1 = runner._get_baseline_cache_path("local")
        p2 = runner._get_baseline_cache_path("image_test_img_v1")
        assert p1 != p2
        assert p1.name == "baseline_cache_local.json"
        assert p2.name == "baseline_cache_image_test_img_v1.json"
        # Regression: the filename must not contain ':' — Docker's bind-mount
        # parser treats it as the mode separator and rejects the mount.
        assert ":" not in p2.name


# =============================================================================
# Docker baseline container dispatch
# =============================================================================


def _docker_runner(tmp_path: Path, config: ExperimentConfig) -> tuple[StudyRunner, MagicMock]:
    spec = RunnerSpec(mode="docker", image="test/pytorch:v0", source="yaml")
    return _make_runner_with_progress(tmp_path, config, runner_specs={"transformers": spec})


class TestBaselineContainerDispatch:
    def test_docker_runner_dispatches_baseline_container_on_fresh_measure(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        runner, _ = _docker_runner(tmp_path, config_cached)
        fake = _make_baseline(42.6)

        with (
            patch(_RUN_BASELINE_CONTAINER, return_value=fake) as mock_dispatch,
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch(_MEASURE) as mock_host_measure,
        ):
            runner._get_baseline(config_cached)

        mock_dispatch.assert_called_once()
        kwargs = mock_dispatch.call_args.kwargs
        assert kwargs["mode"] == "measure"
        assert kwargs["image"] == "test/pytorch:v0"
        assert kwargs["duration_sec"] == config_cached.measurement.baseline.duration_seconds
        assert kwargs["gpu_indices"] == [0]
        mock_host_measure.assert_not_called()

    def test_docker_runner_dispatches_spot_check_container_on_validation(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        runner, _ = _docker_runner(tmp_path, config_validated)
        initial = _make_baseline(50.0)

        calls = []

        def dispatch_side_effect(**kwargs):
            calls.append(kwargs)
            if kwargs["mode"] == "measure":
                return initial
            # spot_check
            return _make_baseline(51.0)

        with (
            patch(_RUN_BASELINE_CONTAINER, side_effect=dispatch_side_effect),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_validated)
            for _ in range(3):
                runner._get_baseline(config_validated)

        modes = [c["mode"] for c in calls]
        assert modes.count("measure") == 1
        assert modes.count("spot_check") == 1

    def test_docker_runner_dispatches_measure_container_on_drift(
        self, tmp_path: Path, config_validated: ExperimentConfig
    ):
        runner, _ = _docker_runner(tmp_path, config_validated)
        initial = _make_baseline(50.0)
        remeasured = _make_baseline(70.0)

        calls = []

        def dispatch_side_effect(**kwargs):
            calls.append(kwargs)
            if kwargs["mode"] == "spot_check":
                return _make_baseline(70.0)  # drift > threshold
            # second measure is the re-measure
            return initial if len([c for c in calls if c["mode"] == "measure"]) == 1 else remeasured

        with (
            patch(_RUN_BASELINE_CONTAINER, side_effect=dispatch_side_effect),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_validated)
            for _ in range(3):
                runner._get_baseline(config_validated)

        modes = [c["mode"] for c in calls]
        # expect: measure (initial), spot_check, measure (re-measurement)
        assert modes.count("measure") == 2
        assert modes.count("spot_check") == 1

    def test_docker_runner_container_failure_returns_none_does_not_crash(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        runner, _ = _docker_runner(tmp_path, config_cached)

        with (
            patch(_RUN_BASELINE_CONTAINER, return_value=None),
            patch(_RESOLVE_GPU, return_value=[0]),
        ):
            result = runner._get_baseline(config_cached)

        assert result is None

    def test_local_runner_does_not_dispatch_container(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        runner, _ = _make_runner_with_progress(tmp_path, config_cached)
        fake = _make_baseline(50.0)

        with (
            patch(_RUN_BASELINE_CONTAINER) as mock_dispatch,
            patch(_MEASURE, return_value=fake),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
        ):
            runner._get_baseline(config_cached)

        mock_dispatch.assert_not_called()


# =============================================================================
# _run_one_docker ordering regression
# =============================================================================


class TestDockerPathOrdering:
    def test_begin_experiment_before_get_baseline(
        self, tmp_path: Path, config_cached: ExperimentConfig
    ):
        """begin_experiment must fire before any baseline on_step_start."""
        study = StudyConfig(
            experiments=[config_cached],
            study_name="test-ordering",
            study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
            study_design_hash=TEST_CONFIG_HASH,
        )
        spec = RunnerSpec(mode="docker", image="test/img:v1", source="yaml")
        manifest = MagicMock()
        progress = MagicMock()
        runner = StudyRunner(
            study,
            manifest,
            tmp_path,
            runner_specs={"transformers": spec},
            progress=progress,
        )
        runner._images_prepared = True

        class FakeDockerRunner:
            def __init__(self, **kwargs):
                pass

            def run(self, *args, **kwargs):
                raise DockerError("stop here")

        with (
            patch(_RUN_BASELINE_CONTAINER, return_value=_make_baseline(50.0)),
            patch(_RESOLVE_GPU, return_value=[0]),
            patch(_SAVE),
            patch("llenergymeasure.infra.docker_runner.DockerRunner", FakeDockerRunner),
            patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
            patch.object(runner, "_handle_result"),
            patch.object(runner, "_persist_failure_artefacts"),
        ):
            runner._run_one_docker(config_cached, spec, config_hash="abc123", cycle=1, index=1)

        # Find the first begin_experiment call and the first baseline step start.
        first_begin = None
        first_baseline_start = None
        for i, call in enumerate(progress.mock_calls):
            if call[0] == "begin_experiment" and first_begin is None:
                first_begin = i
            if (
                call[0] == "on_step_start"
                and call.args
                and call.args[0] == "baseline"
                and first_baseline_start is None
            ):
                first_baseline_start = i

        assert first_begin is not None, "begin_experiment was not called"
        assert first_baseline_start is not None, "baseline on_step_start was not emitted"
        assert first_begin < first_baseline_start, (
            "begin_experiment must run before baseline step events "
            "(see _run_one_docker ordering fix in runner.py §5.1.8)"
        )
