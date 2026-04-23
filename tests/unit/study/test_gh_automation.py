"""Tests for the gh-automation helper — PR body rendering and draft-only invariant."""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.study._gh_automation import (
    GhNotFoundError,
    open_draft_pr,
    plan_request,
)
from llenergymeasure.study.report_gaps import RuleCandidate


def _mk_candidate(
    *,
    engine: str = "transformers",
    source: str = "h3-collisions",
    verified: bool = True,
    confidence: str = "high",
) -> RuleCandidate:
    return RuleCandidate(
        candidate_id=f"{engine}_h3_collision_abc123",
        engine=engine,
        library_version="4.56.0",
        source=source,
        severity="dormant_silent",
        confidence=confidence,
        match_fields={"transformers.sampling.temperature": {"present": True}},
        kwargs_positive={"temperature": 0.1},
        kwargs_negative={"temperature": 0.5},
        expected_outcome={"outcome": "dormant", "emission_channel": "none"},
        message_template=None,
        observed_messages_regex=None,
        evidence={"needs_generalisation_review": True},
        verified=verified,
    )


class TestPlanRequest:
    def test_builds_branch_name(self, tmp_path: Path):
        req = plan_request("transformers", [_mk_candidate()], tmp_path)
        assert req.branch_name.startswith("feedback/report-gaps-transformers-")
        assert req.corpus_path == tmp_path / "transformers.yaml"

    def test_pr_title_includes_source_and_id(self, tmp_path: Path):
        req = plan_request("vllm", [_mk_candidate(engine="vllm")], tmp_path)
        assert "feat(rules)" in req.pr_title
        assert "h3-collisions" in req.pr_title

    def test_pr_body_shows_confidence_mix(self, tmp_path: Path):
        req = plan_request(
            "transformers",
            [
                _mk_candidate(confidence="high"),
                _mk_candidate(confidence="medium"),
            ],
            tmp_path,
        )
        assert "1 high" in req.pr_body
        assert "1 medium" in req.pr_body

    def test_pr_body_flags_unverified_candidates(self, tmp_path: Path):
        req = plan_request(
            "transformers",
            [_mk_candidate(verified=False)],
            tmp_path,
        )
        assert "UNVERIFIED" in req.pr_body


class TestOpenDraftPR:
    def test_creates_branch_commits_opens_draft(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        invocations: list[list[str]] = []

        def _fake(argv):
            invocations.append(argv)
            if argv[:3] == ["gh", "pr", "create"]:
                return "https://github.com/foo/bar/pull/42"
            return ""

        corpus = tmp_path / "validation_rules"
        corpus.mkdir()
        (corpus / "transformers.yaml").write_text("schema_version: 1.0.0\nrules: []\n")
        req = plan_request("transformers", [_mk_candidate()], corpus)
        url = open_draft_pr(req, runner=_fake)

        assert url == "https://github.com/foo/bar/pull/42"
        # Branch created.
        assert any(a[:2] == ["git", "checkout"] for a in invocations)
        # Commit made.
        assert any(a[:2] == ["git", "commit"] for a in invocations)
        # PR created with --draft, never --ready-for-review.
        pr_call = next(a for a in invocations if a[:3] == ["gh", "pr", "create"])
        assert "--draft" in pr_call
        assert "--ready-for-review" not in pr_call

    def test_unverified_candidates_omitted_from_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Unverified candidates show in PR body but don't land in the YAML diff."""
        invocations: list[list[str]] = []

        def _fake(argv):
            invocations.append(argv)
            return "https://github.com/foo/bar/pull/1"

        corpus = tmp_path / "validation_rules"
        corpus.mkdir()
        (corpus / "transformers.yaml").write_text("rules: []\n")
        candidate = _mk_candidate(verified=False)
        req = plan_request("transformers", [candidate], corpus)
        open_draft_pr(req, runner=_fake)

        # Verified-only YAML append skipped entirely when nothing verified.
        assert (corpus / "transformers.yaml").read_text().strip().endswith("rules: []")
        # No git add / commit for an empty YAML change.
        assert not any(a[:2] == ["git", "add"] for a in invocations)

    def test_missing_gh_binary_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _: None)
        corpus = tmp_path / "validation_rules"
        corpus.mkdir()
        (corpus / "transformers.yaml").write_text("rules: []\n")
        req = plan_request("transformers", [_mk_candidate()], corpus)
        with pytest.raises(GhNotFoundError):
            open_draft_pr(req)
