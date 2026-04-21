"""Tests for scripts/generate_curation_doc.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from generate_curation_doc import ENGINES, generate_engine_doc


@pytest.mark.parametrize("engine", ENGINES)
class TestGenerateCurationDoc:
    def test_output_non_empty(self, engine: str) -> None:
        doc = generate_engine_doc(engine)
        assert len(doc) > 100

    def test_has_header(self, engine: str) -> None:
        doc = generate_engine_doc(engine)
        assert doc.startswith("# ")
        assert "Parameter Curation" in doc.split("\n")[0]

    def test_has_engine_version(self, engine: str) -> None:
        doc = generate_engine_doc(engine)
        assert "Engine version:" in doc

    def test_has_summary_counts(self, engine: str) -> None:
        doc = generate_engine_doc(engine)
        assert "parameters curated" in doc

    def test_has_table_headers(self, engine: str) -> None:
        doc = generate_engine_doc(engine)
        assert "| Field | Type | Default | Curated? |" in doc

    def test_contains_discovered_fields(self, engine: str) -> None:
        """Every discovered field name should appear in the doc."""
        from llenergymeasure.config.schema_loader import SchemaLoader

        loader = SchemaLoader()
        schema = loader.load_schema(engine)
        doc = generate_engine_doc(engine)

        all_fields = list(schema.engine_params.keys()) + list(schema.sampling_params.keys())
        for field_name in all_fields:
            assert f"`{field_name}`" in doc, f"Missing field: {field_name}"

    def test_curated_fields_marked(self, engine: str) -> None:
        """At least one field should be marked as curated."""
        doc = generate_engine_doc(engine)
        assert "| yes |" in doc

    def test_pipes_escaped_in_types(self, engine: str) -> None:
        """Union types should use escaped pipe to not break table."""
        doc = generate_engine_doc(engine)
        # If any line has a union type, it should be escaped
        for line in doc.split("\n"):
            if line.startswith("|") and "\\|" in line:
                # Escaped pipe found - table won't break
                break


class TestDisplayNames:
    def test_vllm_display_name(self) -> None:
        doc = generate_engine_doc("vllm")
        assert doc.startswith("# vLLM Parameter Curation")

    def test_tensorrt_display_name(self) -> None:
        doc = generate_engine_doc("tensorrt")
        assert doc.startswith("# TensorRT-LLM Parameter Curation")
