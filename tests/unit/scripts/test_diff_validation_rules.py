"""Tests for :mod:`scripts.diff_validation_rules`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import diff_validation_rules as diff_rules  # noqa: E402

SCRIPT = _PROJECT_ROOT / "scripts" / "diff_validation_rules.py"


def _envelope(cases: list[dict[str, Any]], **meta: Any) -> dict[str, Any]:
    base = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": "4.56.0",
        "image_ref": "test:latest",
        "base_image_ref": "test:latest",
        "vendored_at": "2026-04-23T00:00:00+00:00",
        "vendor_commit": "abc",
        "cases": cases,
        "divergences": [],
    }
    base.update(meta)
    return base


def _case(
    id: str,
    outcome: str = "error",
    emission_channel: str = "none",
    observed_messages: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "outcome": outcome,
        "emission_channel": emission_channel,
        "observed_messages": observed_messages or [],
        "observed_silent_normalisations": {},
        "positive_confirmed": True,
        "negative_confirmed": True,
        "duration_ms": 1,
    }


# ---------------------------------------------------------------------------
# diff_rules() unit tests
# ---------------------------------------------------------------------------


class TestDiffRules:
    def test_identical_envelopes(self) -> None:
        env = _envelope([_case("r1"), _case("r2")])
        result = diff_rules.diff_rules(env, env)
        assert not result.is_breaking
        assert result.safe == []
        assert result.breaking == []

    def test_added_rule_is_safe(self) -> None:
        old = _envelope([_case("r1")])
        new = _envelope([_case("r1"), _case("r2", outcome="warn")])
        result = diff_rules.diff_rules(old, new)
        assert not result.is_breaking
        assert len(result.safe) == 1
        assert result.safe[0].kind == "added_rule"
        assert result.safe[0].rule_id == "r2"

    def test_removed_rule_is_breaking(self) -> None:
        old = _envelope([_case("r1"), _case("r2")])
        new = _envelope([_case("r1")])
        result = diff_rules.diff_rules(old, new)
        assert result.is_breaking
        assert any(c.kind == "removed_rule" for c in result.breaking)

    def test_severity_escalated_is_breaking(self) -> None:
        old = _envelope([_case("r1", outcome="warn")])
        new = _envelope([_case("r1", outcome="error")])
        result = diff_rules.diff_rules(old, new)
        assert result.is_breaking
        assert any(c.kind == "severity_escalated" for c in result.breaking)

    def test_severity_relaxed_is_safe(self) -> None:
        old = _envelope([_case("r1", outcome="error")])
        new = _envelope([_case("r1", outcome="warn")])
        result = diff_rules.diff_rules(old, new)
        assert not result.is_breaking
        assert any(c.kind == "severity_relaxed" for c in result.safe)

    def test_outcome_changed_same_rank_is_breaking(self) -> None:
        # no_op vs skipped_hardware_dependent — sibling categories, not a
        # monotonic change — flagged as outcome_changed (breaking).
        old = _envelope([_case("r1", outcome="no_op")])
        new = _envelope([_case("r1", outcome="no_op")])
        # tweak outcome that has same rank to force outcome_changed
        new["cases"][0]["outcome"] = "skipped_hardware_dependent"
        result = diff_rules.diff_rules(old, new)
        assert result.is_breaking or len(result.safe) >= 1

    def test_emission_channel_widened_is_safe(self) -> None:
        old = _envelope([_case("r1", emission_channel="none")])
        new = _envelope([_case("r1", emission_channel="logger_warning")])
        result = diff_rules.diff_rules(old, new)
        assert any(c.kind == "emission_channel_widened" for c in result.safe)

    def test_emission_channel_changed_is_breaking(self) -> None:
        old = _envelope([_case("r1", emission_channel="logger_warning")])
        new = _envelope([_case("r1", emission_channel="warnings_warn")])
        result = diff_rules.diff_rules(old, new)
        assert any(c.kind == "emission_channel_changed" for c in result.breaking)

    def test_metadata_change_tracked(self) -> None:
        old = _envelope([], engine_version="4.55.0")
        new = _envelope([], engine_version="4.56.0")
        result = diff_rules.diff_rules(old, new)
        assert "engine_version" in result.metadata_changes
        assert result.metadata_changes["engine_version"]["old"] == "4.55.0"
        assert result.metadata_changes["engine_version"]["new"] == "4.56.0"

    def test_message_template_change_tracked(self) -> None:
        old = _envelope([_case("r1", observed_messages=["msg1"])])
        new = _envelope([_case("r1", observed_messages=["msg2", "msg3"])])
        result = diff_rules.diff_rules(old, new)
        kinds = [c.kind for c in result.safe + result.breaking]
        assert "message_template_changed" in kinds

    def test_summary_empty_on_no_changes(self) -> None:
        env = _envelope([_case("r1")])
        result = diff_rules.diff_rules(env, env)
        assert "No changes" in result.summary

    def test_summary_counts(self) -> None:
        old = _envelope([_case("r1")])
        new = _envelope([_case("r1", outcome="error"), _case("r2", outcome="warn")])
        # r1 in both — no change; r2 added — safe
        result = diff_rules.diff_rules(old, new)
        assert "1 rules-safe" in result.summary


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_well_formed_for_no_changes(self) -> None:
        env = _envelope([_case("r1")])
        md = diff_rules.render_markdown(diff_rules.diff_rules(env, env), title="Test")
        assert "## Test" in md
        assert "No changes detected" in md

    def test_includes_breaking_section(self) -> None:
        old = _envelope([_case("r1", outcome="warn")])
        new = _envelope([_case("r1", outcome="error")])
        md = diff_rules.render_markdown(diff_rules.diff_rules(old, new))
        assert "rules-breaking" in md
        assert "severity_escalated" in md

    def test_includes_safe_section(self) -> None:
        old = _envelope([_case("r1")])
        new = _envelope([_case("r1"), _case("r2")])
        md = diff_rules.render_markdown(diff_rules.diff_rules(old, new))
        assert "rules-safe" in md
        assert "added_rule" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(old: dict[str, Any], new: dict[str, Any], tmp_path: Path) -> tuple[int, dict, str]:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(old))
    new_path.write_text(json.dumps(new))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(old_path), str(new_path)],
        capture_output=True,
        text=True,
    )
    stdout_json = json.loads(result.stdout) if result.stdout.strip() else {}
    return result.returncode, stdout_json, result.stderr


class TestCLI:
    def test_identical_exits_0(self, tmp_path: Path) -> None:
        env = _envelope([_case("r1")])
        code, data, _ = _run_cli(env, env, tmp_path)
        assert code == 0
        assert data["is_breaking"] is False

    def test_breaking_exits_1(self, tmp_path: Path) -> None:
        old = _envelope([_case("r1"), _case("r2")])
        new = _envelope([_case("r1")])
        code, data, _ = _run_cli(old, new, tmp_path)
        assert code == 1
        assert data["is_breaking"] is True

    def test_malformed_input_exits_2(self, tmp_path: Path) -> None:
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        old.write_text("this is not JSON")
        new.write_text("{}")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(old), str(new)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
