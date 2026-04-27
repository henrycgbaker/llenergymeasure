"""Tests for scripts/diff_discovered_schemas.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "diff_discovered_schemas.py"


def _make_schema(
    engine_params: dict | None = None,
    sampling_params: dict | None = None,
    engine_version: str = "0.7.3",
) -> dict:
    """Build a minimal schema dict."""
    return {
        "schema_version": "1.0.0",
        "engine": "test",
        "engine_version": engine_version,
        "engine_commit_sha": None,
        "image_ref": "test:latest",
        "base_image_ref": "test:latest",
        "discovered_at": "2026-01-01T00:00:00Z",
        "discovery_method": "test",
        "discovery_limitations": [],
        "engine_params": engine_params or {},
        "sampling_params": sampling_params or {},
    }


def _run_diff(old: dict, new: dict, tmp_path: Path) -> tuple[int, dict, str]:
    """Run the differ and return (exit_code, stdout_json, stderr)."""
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    old_file.write_text(json.dumps(old))
    new_file.write_text(json.dumps(new))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(old_file), str(new_file)],
        capture_output=True,
        text=True,
    )
    stdout_data = json.loads(result.stdout) if result.stdout.strip() else {}
    return result.returncode, stdout_data, result.stderr


class TestIdenticalSchemas:
    def test_no_changes(self, tmp_path: Path):
        schema = _make_schema(engine_params={"model": {"type": "str", "default": "gpt2"}})
        code, output, _ = _run_diff(schema, schema, tmp_path)
        assert code == 0
        assert output["is_breaking"] is False
        assert output["safe"] == []
        assert output["breaking"] == []


class TestSafeChanges:
    def test_field_added(self, tmp_path: Path):
        old = _make_schema(engine_params={"model": {"type": "str", "default": "gpt2"}})
        new = _make_schema(
            engine_params={
                "model": {"type": "str", "default": "gpt2"},
                "new_param": {"type": "int", "default": 0},
            }
        )
        code, output, _ = _run_diff(old, new, tmp_path)
        assert code == 0
        assert output["is_breaking"] is False
        assert any(c["kind"] == "added" and c["field"] == "new_param" for c in output["safe"])

    def test_type_widened(self, tmp_path: Path):
        old = _make_schema(engine_params={"x": {"type": "int", "default": 0}})
        new = _make_schema(engine_params={"x": {"type": "int | None", "default": 0}})
        code, output, _ = _run_diff(old, new, tmp_path)
        assert code == 0
        assert any(c["kind"] == "type_widened" for c in output["safe"])

    def test_default_changed(self, tmp_path: Path):
        old = _make_schema(engine_params={"x": {"type": "float", "default": 0.9}})
        new = _make_schema(engine_params={"x": {"type": "float", "default": 0.95}})
        code, output, _ = _run_diff(old, new, tmp_path)
        assert code == 0
        assert any(c["kind"] == "default_changed" for c in output["safe"])

    def test_description_changed(self, tmp_path: Path):
        old = _make_schema(engine_params={"x": {"type": "int", "default": 0, "description": "old"}})
        new = _make_schema(engine_params={"x": {"type": "int", "default": 0, "description": "new"}})
        code, output, _ = _run_diff(old, new, tmp_path)
        assert code == 0
        assert any(c["kind"] == "description_changed" for c in output["safe"])


class TestBreakingChanges:
    def test_field_removed(self, tmp_path: Path):
        old = _make_schema(
            engine_params={
                "model": {"type": "str", "default": "gpt2"},
                "old_param": {"type": "int", "default": 0},
            }
        )
        new = _make_schema(engine_params={"model": {"type": "str", "default": "gpt2"}})
        code, output, _ = _run_diff(old, new, tmp_path)
        assert code == 1
        assert output["is_breaking"] is True
        assert any(c["kind"] == "removed" and c["field"] == "old_param" for c in output["breaking"])

    def test_type_narrowed(self, tmp_path: Path):
        old = _make_schema(engine_params={"x": {"type": "int | None", "default": None}})
        new = _make_schema(engine_params={"x": {"type": "int", "default": 0}})
        code, output, _ = _run_diff(old, new, tmp_path)
        assert code == 1
        assert any(c["kind"] == "type_narrowed" for c in output["breaking"])


class TestMetadata:
    def test_metadata_excluded_from_classification(self, tmp_path: Path):
        """discovered_at changes should not appear as safe/breaking."""
        old = _make_schema()
        new = _make_schema()
        new["discovered_at"] = "2026-12-31T00:00:00Z"
        code, output, _ = _run_diff(old, new, tmp_path)
        assert code == 0
        assert output["safe"] == []
        assert output["breaking"] == []

    def test_engine_version_change_reported(self, tmp_path: Path):
        old = _make_schema(engine_version="0.7.3")
        new = _make_schema(engine_version="0.8.0")
        _code, output, _ = _run_diff(old, new, tmp_path)
        assert "engine_version" in output["metadata_changes"]
        assert output["metadata_changes"]["engine_version"]["old"] == "0.7.3"
        assert output["metadata_changes"]["engine_version"]["new"] == "0.8.0"


class TestErrors:
    def test_malformed_json(self, tmp_path: Path):
        old_file = tmp_path / "old.json"
        new_file = tmp_path / "new.json"
        old_file.write_text("not json")
        new_file.write_text("{}")

        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(old_file), str(new_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_missing_file(self, tmp_path: Path):
        new_file = tmp_path / "new.json"
        new_file.write_text("{}")

        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(tmp_path / "nope.json"), str(new_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_missing_section_graceful(self, tmp_path: Path):
        """Schema with no engine_params section should not crash."""
        old = {"schema_version": "1.0.0", "engine": "test", "engine_version": "1.0"}
        new = {"schema_version": "1.0.0", "engine": "test", "engine_version": "1.0"}
        code, _output, _ = _run_diff(old, new, tmp_path)
        assert code == 0
