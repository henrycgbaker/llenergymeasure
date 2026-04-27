"""Tests for SchemaLoader — loads vendored engine schemas from discovered_schemas/."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from llenergymeasure.config import (
    DiscoveredSchema,
    DiscoveryLimitation,
    SchemaLoader,
    UnsupportedSchemaVersionError,
)
from llenergymeasure.config.schema_loader import _parse_envelope
from llenergymeasure.config.ssot import Engine

KNOWN_ENGINES = tuple(Engine)


@pytest.mark.parametrize("engine", KNOWN_ENGINES)
def test_load_schema_returns_discovered_schema(engine: str) -> None:
    schema = SchemaLoader().load_schema(engine)

    assert isinstance(schema, DiscoveredSchema)
    assert schema.engine == engine
    assert schema.schema_version.startswith("1.")
    assert schema.engine_version
    assert schema.image_ref
    assert schema.base_image_ref
    assert isinstance(schema.discovered_at, datetime)
    assert schema.engine_params, f"{engine}.json has no engine_params"
    assert schema.sampling_params, f"{engine}.json has no sampling_params"


def test_load_schema_caches_per_instance() -> None:
    loader = SchemaLoader()
    first = loader.load_schema("vllm")
    second = loader.load_schema("vllm")
    assert first is second, "cached lookups should return the same object"


def test_invalidate_drops_cache() -> None:
    loader = SchemaLoader()
    first = loader.load_schema("vllm")
    loader.invalidate("vllm")
    second = loader.load_schema("vllm")
    assert first is not second, "after invalidate(engine), a reload produces a new object"
    assert first == second, "the schemas' content should be equal"


def test_invalidate_all() -> None:
    loader = SchemaLoader()
    loader.load_schema("vllm")
    loader.load_schema("tensorrt")
    loader.invalidate()
    # Private cache inspection is acceptable in tests
    assert not loader._cache


def test_load_schema_unknown_engine_raises() -> None:
    with pytest.raises(ValueError, match="Unknown engine 'made-up'"):
        SchemaLoader().load_schema("made-up")


def test_load_schema_missing_file_raises_with_hint(tmp_path: Path) -> None:
    # Point the loader at an empty resources package by patching the package constant
    fake_pkg = tmp_path / "empty_pkg"
    fake_pkg.mkdir()
    (fake_pkg / "__init__.py").write_text("")

    # Patch _KNOWN_ENGINES to include 'ghost' but no file exists for it
    with (
        patch("llenergymeasure.config.schema_loader._KNOWN_ENGINES", ("vllm", "ghost")),
        pytest.raises(FileNotFoundError, match=r"refresh_discovered_schemas\.sh ghost"),
    ):
        loader = SchemaLoader()
        loader.load_schema("ghost")


def test_major_version_mismatch_raises() -> None:
    envelope = _minimal_envelope(schema_version="2.0.0")
    with pytest.raises(UnsupportedSchemaVersionError, match="major=2"):
        _parse_envelope(engine="vllm", raw_text=json.dumps(envelope))


def test_unparseable_version_raises() -> None:
    envelope = _minimal_envelope(schema_version="not-semver")
    with pytest.raises(UnsupportedSchemaVersionError, match="Unparseable schema_version"):
        _parse_envelope(engine="vllm", raw_text=json.dumps(envelope))


def test_minor_version_accepted() -> None:
    envelope = _minimal_envelope(schema_version="1.7.3")
    parsed = _parse_envelope(engine="vllm", raw_text=json.dumps(envelope))
    assert parsed.schema_version == "1.7.3"


def test_iso_z_termination_accepted() -> None:
    envelope = _minimal_envelope(discovered_at="2026-04-13T22:00:00Z")
    parsed = _parse_envelope(engine="vllm", raw_text=json.dumps(envelope))
    assert parsed.discovered_at.isoformat().startswith("2026-04-13T22:00:00")


def test_base_image_ref_falls_back_to_image_ref() -> None:
    envelope = _minimal_envelope()
    envelope.pop("base_image_ref")
    parsed = _parse_envelope(engine="vllm", raw_text=json.dumps(envelope))
    assert parsed.base_image_ref == parsed.image_ref


def test_discovery_limitations_parsed_into_dataclass() -> None:
    envelope = _minimal_envelope()
    envelope["discovery_limitations"] = [
        {"section": "engine_params", "fields": ["foo", "bar"], "reason": "missing"}
    ]
    parsed = _parse_envelope(engine="vllm", raw_text=json.dumps(envelope))
    assert len(parsed.discovery_limitations) == 1
    lim = parsed.discovery_limitations[0]
    assert isinstance(lim, DiscoveryLimitation)
    assert lim.section == "engine_params"
    assert lim.fields == ["foo", "bar"]
    assert lim.reason == "missing"


def test_load_all_schemas_returns_all_known() -> None:
    all_schemas = SchemaLoader().load_all_schemas()
    assert set(all_schemas) == set(KNOWN_ENGINES)


@pytest.mark.parametrize("engine", KNOWN_ENGINES)
def test_vendored_schema_has_expected_shape(engine: str) -> None:
    schema = SchemaLoader().load_schema(engine)
    # Every param entry must be a dict with a 'type' key (common contract)
    for name, spec in schema.engine_params.items():
        assert isinstance(spec, dict), f"{engine}.engine_params[{name}] is not a dict"
        assert "type" in spec, f"{engine}.engine_params[{name}] has no 'type' key"
    for name, spec in schema.sampling_params.items():
        assert isinstance(spec, dict), f"{engine}.sampling_params[{name}] is not a dict"
        assert "type" in spec, f"{engine}.sampling_params[{name}] has no 'type' key"


def test_vllm_has_expected_field_floor() -> None:
    # Soft floors (upstream adds fields over time) to catch catastrophic loss
    schema = SchemaLoader().load_schema("vllm")
    assert len(schema.engine_params) >= 80, "vLLM EngineArgs should yield >=80 fields"
    assert len(schema.sampling_params) >= 20, "vLLM SamplingParams should yield >=20 fields"


def test_tensorrt_has_description_metadata() -> None:
    schema = SchemaLoader().load_schema("tensorrt")
    has_description = any(spec.get("description") for spec in schema.engine_params.values())
    assert has_description, "TRT-LLM TrtLlmArgs Pydantic schema should yield per-field descriptions"


def test_transformers_records_kwargs_as_limitations() -> None:
    schema = SchemaLoader().load_schema("transformers")
    kwargs_limitation = next(
        (lim for lim in schema.discovery_limitations if any("**kwargs" in f for f in lim.fields)),
        None,
    )
    assert kwargs_limitation is not None, (
        "Transformers from_pretrained kwargs should be recorded as a limitation"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_envelope(**overrides: object) -> dict[str, object]:
    """Produce a minimal but valid envelope for parser-targeted tests."""
    envelope: dict[str, object] = {
        "schema_version": "1.0.0",
        "engine": "vllm",
        "engine_version": "0.7.3",
        "engine_commit_sha": None,
        "image_ref": "vllm/vllm-openai:v0.7.3",
        "base_image_ref": "vllm/vllm-openai:v0.7.3",
        "discovered_at": "2026-04-13T22:00:00+00:00",
        "discovery_method": "unit test fixture",
        "discovery_limitations": [],
        "engine_params": {"dummy": {"type": "str", "default": None}},
        "sampling_params": {"dummy": {"type": "str", "default": None}},
    }
    envelope.update(overrides)
    return envelope
