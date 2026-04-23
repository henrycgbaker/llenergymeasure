"""Tests for the ``llem report-gaps`` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llenergymeasure.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_h3_collision(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    for exp_id, h1 in [("a", "h1_a"), ("b", "h1_b")]:
        d = results / f"run_{exp_id}"
        d.mkdir(parents=True)
        (d / "config.json").write_text(
            json.dumps(
                {
                    "experiment_id": exp_id,
                    "engine": "transformers",
                    "library_version": "4.56.0",
                    "h1_hash": h1,
                    "h3_hash": "h3_shared",
                    "declared_kwargs": {"temperature": 0.1 if exp_id == "a" else 0.5},
                    "effective_sampling_params": {"temperature": 1.0},
                }
            )
        )
    return results


class TestSourceFlag:
    def test_invalid_source_rejected(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(app, ["report-gaps", "--source", "nope"])
        assert result.exit_code != 0
        assert "must be one of" in result.output
        assert "'nope'" in result.output

    def test_invalid_engine_rejected(self, runner: CliRunner):
        result = runner.invoke(app, ["report-gaps", "--engine", "custom"])
        assert result.exit_code != 0
        assert "must be one of" in result.output
        assert "'custom'" in result.output

    def test_h3_source_on_empty_dir(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Candidates: 0" in result.output


class TestH3CollisionsSource:
    def test_emits_candidate(self, runner: CliRunner, tmp_path: Path):
        results = _seed_h3_collision(tmp_path)
        out = tmp_path / "proposed.yaml"
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(results),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Candidates: 1" in result.output
        assert out.exists()
        content = out.read_text()
        assert "dormant_silent" in content
        assert "transformers" in content


class TestDryRun:
    def test_dry_run_does_not_write_out(self, runner: CliRunner, tmp_path: Path):
        results = _seed_h3_collision(tmp_path)
        out = tmp_path / "proposed.yaml"
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(results),
                "--out",
                str(out),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "[dry-run]" in result.output
        assert not out.exists()

    def test_dry_run_skips_open_pr(self, runner: CliRunner, tmp_path: Path):
        results = _seed_h3_collision(tmp_path)
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(results),
                "--open-pr",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "[dry-run] Would open a draft PR" in result.output


class TestOpenPRWithMockedGh:
    def test_open_pr_invokes_gh_draft(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        results = _seed_h3_collision(tmp_path)
        invocations: list[list[str]] = []

        # Monkeypatch shutil.which so the guard passes, then monkeypatch the
        # inner runner that the API calls into.
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda _: "/usr/bin/gh")

        def _fake_run(argv):
            invocations.append(argv)
            if argv[:3] == ["gh", "pr", "create"]:
                return "https://github.com/fake/fake/pull/1"
            return ""

        from llenergymeasure.cli import report_gaps as _report_gaps_mod
        from llenergymeasure.study import _gh_automation

        monkeypatch.setattr(_gh_automation, "_run_gh", _fake_run)
        # Isolate the corpus edit — the open_draft_pr flow writes into
        # `{corpus_root}/{engine}.yaml` before it invokes gh. Without this
        # redirect the test would mutate the real configs/validation_rules/.
        sandbox_corpus = tmp_path / "validation_rules"
        sandbox_corpus.mkdir()
        (sandbox_corpus / "transformers.yaml").write_text("rules: []\n")
        monkeypatch.setattr(_report_gaps_mod, "_find_corpus_root", lambda: sandbox_corpus)

        # Run inside a git-like directory (we don't actually init a repo because
        # the runner is fully mocked).
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(results),
                "--open-pr",
            ],
        )
        assert result.exit_code == 0, result.output
        pr_invocation = next(a for a in invocations if a[:3] == ["gh", "pr", "create"])
        assert "--draft" in pr_invocation
        assert "--ready-for-review" not in pr_invocation
        assert "feat(rules)" in " ".join(pr_invocation)

    def test_missing_gh_falls_back(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        results = _seed_h3_collision(tmp_path)
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda _: None)
        result = runner.invoke(
            app,
            [
                "report-gaps",
                "--source",
                "h3-collisions",
                "--results-dir",
                str(results),
                "--open-pr",
            ],
        )
        assert result.exit_code != 0
        assert "gh unavailable" in result.output


class TestDedupOffRefusesAutoPR:
    def test_refuses_auto_pr_for_unverified_candidates(self, runner: CliRunner, tmp_path: Path):
        results = _seed_h3_collision(tmp_path)
        # Plant a dedup_mode=off manifest so candidates get flagged unverified.
        from llenergymeasure.study.equivalence_groups import (
            EquivalenceGroups,
            write_equivalence_groups,
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
                "--open-pr",
            ],
        )
        assert result.exit_code == 2
        assert "UNVERIFIED" in result.output


class TestSummaryRendering:
    def test_shows_confidence_breakdown(self, runner: CliRunner, tmp_path: Path):
        results = _seed_h3_collision(tmp_path)
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
        assert "high" in result.output and "medium" in result.output
