"""Tests that validate GPU-produced replay fixtures in GPU-free CI.

These tests use the `replay_results` conftest fixture which loads
ExperimentResult JSON from tests/fixtures/replay/. When no fixtures
exist (e.g. before first GPU CI run), tests are skipped gracefully.
"""

import pytest

from llenergymeasure.domain.experiment import ExperimentResult


class TestReplayFixtures:
    """Validate real GPU experiment results deserialise and contain sane values."""

    def _require_fixtures(self, replay_results):
        if not replay_results:
            pytest.skip("No replay fixtures available (run GPU CI first)")

    def test_replay_fixtures_deserialise(self, replay_results):
        """Replay fixtures load as valid ExperimentResult instances."""
        self._require_fixtures(replay_results)
        for result in replay_results:
            assert isinstance(result, ExperimentResult)

    def test_schema_version(self, replay_results):
        """All replay fixtures have schema_version 3.0."""
        self._require_fixtures(replay_results)
        for result in replay_results:
            from tests.conftest import EXPERIMENT_SCHEMA_VERSION

            assert result.schema_version == EXPERIMENT_SCHEMA_VERSION

    def test_energy_values_positive(self, replay_results):
        """Real GPU experiments produce positive energy measurements."""
        self._require_fixtures(replay_results)
        for result in replay_results:
            assert result.total_energy_j > 0
            assert result.avg_energy_per_token_j > 0

    def test_throughput_values_positive(self, replay_results):
        """Real GPU experiments produce positive throughput."""
        self._require_fixtures(replay_results)
        for result in replay_results:
            assert result.avg_tokens_per_second > 0
            assert result.total_tokens > 0
            assert result.total_inference_time_sec > 0

    def test_flops_populated(self, replay_results):
        """Real GPU experiments produce FLOPs estimates."""
        self._require_fixtures(replay_results)
        for result in replay_results:
            assert result.total_flops > 0

    def test_config_hash_format(self, replay_results):
        """Config hash is 16-char hex string."""
        self._require_fixtures(replay_results)
        for result in replay_results:
            assert len(result.measurement_config_hash) == 16
            assert all(c in "0123456789abcdef" for c in result.measurement_config_hash)

    def test_json_round_trip(self, replay_results):
        """Replay fixtures survive a JSON round-trip without data loss."""
        self._require_fixtures(replay_results)
        for result in replay_results:
            json_str = result.model_dump_json()
            restored = ExperimentResult.model_validate_json(json_str)
            assert restored.total_energy_j == result.total_energy_j
            assert restored.total_tokens == result.total_tokens
            assert restored.experiment_id == result.experiment_id
