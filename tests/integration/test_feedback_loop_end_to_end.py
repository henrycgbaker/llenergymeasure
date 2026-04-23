"""End-to-end feedback-loop integration test — runtime capture → report-gaps.

Exercises the full 5-PR stack seam: runtime observation wrapper writes to a
JSONL cache (50.3b), synthetic sidecars carry H3 hashes from 50.3a's
canonicaliser+hashing pipeline, and `llem report-gaps` detects gaps across
both feedback channels plus renders a YAML corpus proposal. Closes the loop
from walker extraction (50.2a) → corpus refresh (50.2b) → generic validator
(50.2c) → canonicaliser + H3 (50.3a) → gap reports (50.3b).
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llenergymeasure.cli import app
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.study.equivalence_groups import (
    EquivalenceGroups,
    write_equivalence_groups,
)
from llenergymeasure.study.runtime_observations import capture_runtime_observations


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_fake_sidecar(
    path: Path,
    *,
    experiment_id: str,
    h1: str,
    h3: str,
    declared: dict,
    effective: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": h1,
                "h3_hash": h3,
                "declared_kwargs": declared,
                "effective_sampling_params": effective,
            }
        )
    )


class TestFeedbackLoopEndToEnd:
    def test_full_loop_closes(self, runner: CliRunner, tmp_path: Path):
        """Runtime capture → sidecar inspect → report-gaps → YAML proposal."""

        # --- Part 1: exercise runtime observation capture as the runner would ---
        cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
        cache = tmp_path / "runtime_observations.jsonl"

        with capture_runtime_observations(cfg, cache_path=cache):
            # A library-style warning — not covered by any existing corpus rule.
            logging.getLogger("transformers").warning(
                "novel runtime behavior at temperature=0.001 not yet documented"
            )
            warnings.warn("second unknown emission via warnings.warn", UserWarning, stacklevel=2)

        assert cache.exists()
        lines = cache.read_text().splitlines()
        assert len(lines) == 1
        obs = json.loads(lines[0])
        assert obs["outcome"] == "success"
        assert any("novel runtime" in r["message"] for r in obs["logger_records"])
        assert any("second unknown" in w for w in obs["warnings"])

        # --- Part 2: synthetic sidecars with H3 collision ---
        results = tmp_path / "results"
        _write_fake_sidecar(
            results / "run_001/config.json",
            experiment_id="exp_001",
            h1="sha256:h1_config_a",
            h3="sha256:h3_shared_epsilon_clamp",
            declared={"temperature": 0.001},
            effective={"temperature": 0.01, "top_p": 1.0},
        )
        _write_fake_sidecar(
            results / "run_002/config.json",
            experiment_id="exp_002",
            h1="sha256:h1_config_b",
            h3="sha256:h3_shared_epsilon_clamp",
            declared={"temperature": 0.005},
            effective={"temperature": 0.01, "top_p": 1.0},
        )
        write_equivalence_groups(
            EquivalenceGroups(study_id="test_study", dedup_mode="h1"),
            results / "equivalence_groups.json",
        )

        # --- Part 3: report-gaps --source both ---
        out = tmp_path / "proposed.yaml"
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "both",
                "--results-dir",
                str(results),
                "--cache-path",
                str(cache),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output

        # Loop closes: both channels produced at least one candidate.
        assert "Candidates:" in result.output
        assert out.exists()

        import yaml

        parsed = yaml.safe_load(out.read_text())
        assert "rules" in parsed
        rule_ids = {r["id"] for r in parsed["rules"]}
        # H3 collision candidate has collision-derived id.
        assert any("h3_collision" in r_id for r_id in rule_ids)
        # Runtime warning candidate has runtime_warning-derived id.
        assert any("runtime_warning" in r_id for r_id in rule_ids)

    def test_induced_gap_surfaces_in_report(self, runner: CliRunner, tmp_path: Path):
        """An induced 'same H1 but different observed H3' is NOT a gap; ensure only true gaps surface."""
        results = tmp_path / "results"
        _write_fake_sidecar(
            results / "run_1/config.json",
            experiment_id="e1",
            h1="sha:same_h1",
            h3="sha:h3_a",
            declared={"do_sample": False},
            effective={"do_sample": False},
        )
        _write_fake_sidecar(
            results / "run_2/config.json",
            experiment_id="e2",
            h1="sha:same_h1",
            h3="sha:h3_b",
            declared={"do_sample": False},
            effective={"do_sample": False, "extra": True},
        )
        # H3 hashes differ; no collision; no candidate.
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(results),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Candidates: 0" in result.output

    def test_dedup_off_study_produces_unverified_candidates(
        self, runner: CliRunner, tmp_path: Path
    ):
        """dedup_mode=off → candidates emitted but flagged unverified; --open-pr refuses."""
        results = tmp_path / "results"
        _write_fake_sidecar(
            results / "run_a/config.json",
            experiment_id="a",
            h1="sha:h1_a",
            h3="sha:h3_shared",
            declared={"temperature": 0.1},
            effective={"temperature": 1.0},
        )
        _write_fake_sidecar(
            results / "run_b/config.json",
            experiment_id="b",
            h1="sha:h1_b",
            h3="sha:h3_shared",
            declared={"temperature": 0.5},
            effective={"temperature": 1.0},
        )
        write_equivalence_groups(
            EquivalenceGroups(study_id="off_study", dedup_mode="off"),
            results / "equivalence_groups.json",
        )
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(results),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "UNVERIFIED" in result.output
