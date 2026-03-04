"""Unit tests for config/provenance.py.

Covers:
- ParameterSource enum — string values and ordering
- ParameterProvenance model — construction, __str__, serialisation
- ResolvedConfig model — get_provenance(), get_parameters_by_source(),
  get_non_default_parameters(), get_cli_overrides(), to_summary_dict()
- flatten_dict() — nested dict flattening
- unflatten_dict() — dot-key to nested dict
- compare_dicts() — changed vs unchanged detection
"""

from __future__ import annotations

from llenergymeasure.config.provenance import (
    ParameterProvenance,
    ParameterSource,
    ResolvedConfig,
    compare_dicts,
    flatten_dict,
    unflatten_dict,
)

# ---------------------------------------------------------------------------
# ParameterSource
# ---------------------------------------------------------------------------


class TestParameterSource:
    def test_string_values(self):
        assert ParameterSource.PYDANTIC_DEFAULT.value == "pydantic_default"
        assert ParameterSource.PRESET.value == "preset"
        assert ParameterSource.CONFIG_FILE.value == "config_file"
        assert ParameterSource.CLI.value == "cli"

    def test_is_string_enum(self):
        """ParameterSource is a str enum — can be compared to strings."""
        assert ParameterSource.CLI == "cli"
        assert ParameterSource.PYDANTIC_DEFAULT == "pydantic_default"


# ---------------------------------------------------------------------------
# ParameterProvenance
# ---------------------------------------------------------------------------


class TestParameterProvenance:
    def test_construction_with_required_fields(self):
        p = ParameterProvenance(
            path="decoder.temperature",
            value=0.7,
            source=ParameterSource.CONFIG_FILE,
        )
        assert p.path == "decoder.temperature"
        assert p.value == 0.7
        assert p.source == ParameterSource.CONFIG_FILE
        assert p.source_detail is None

    def test_construction_with_all_fields(self):
        p = ParameterProvenance(
            path="model",
            value="gpt2",
            source=ParameterSource.CLI,
            source_detail="--model flag",
        )
        assert p.source_detail == "--model flag"

    def test_str_includes_path_and_source(self):
        p = ParameterProvenance(
            path="backend",
            value="pytorch",
            source=ParameterSource.PRESET,
        )
        s = str(p)
        assert "backend" in s
        assert "pytorch" in s
        assert "preset" in s

    def test_str_includes_source_detail_when_present(self):
        p = ParameterProvenance(
            path="model",
            value="llama",
            source=ParameterSource.CLI,
            source_detail="--model",
        )
        s = str(p)
        assert "--model" in s

    def test_str_excludes_parentheses_when_no_detail(self):
        p = ParameterProvenance(
            path="seed",
            value=42,
            source=ParameterSource.PYDANTIC_DEFAULT,
        )
        s = str(p)
        assert "(" not in s

    def test_serialises_to_dict(self):
        p = ParameterProvenance(
            path="decoder.temperature",
            value=0.9,
            source=ParameterSource.CLI,
        )
        d = p.model_dump()
        assert d["path"] == "decoder.temperature"
        assert d["value"] == 0.9
        assert d["source"] == "cli"


# ---------------------------------------------------------------------------
# ResolvedConfig
# ---------------------------------------------------------------------------


def _make_provenance(**kwargs: dict) -> dict[str, ParameterProvenance]:
    """Build a provenance dict from (path -> source) shorthand."""
    result = {}
    for path, (source, value) in kwargs.items():
        result[path] = ParameterProvenance(path=path, value=value, source=source)
    return result


class TestResolvedConfig:
    def test_get_provenance_returns_entry(self):
        prov = _make_provenance(**{"decoder.temperature": (ParameterSource.CLI, 0.8)})
        rc = ResolvedConfig(config=object(), provenance=prov)
        entry = rc.get_provenance("decoder.temperature")
        assert entry is not None
        assert entry.value == 0.8

    def test_get_provenance_returns_none_for_missing(self):
        rc = ResolvedConfig(config=object(), provenance={})
        assert rc.get_provenance("nonexistent") is None

    def test_get_parameters_by_source_filters_correctly(self):
        prov = _make_provenance(
            **{
                "model": (ParameterSource.CLI, "gpt2"),
                "backend": (ParameterSource.PRESET, "pytorch"),
                "seed": (ParameterSource.PYDANTIC_DEFAULT, 42),
            }
        )
        rc = ResolvedConfig(config=object(), provenance=prov)
        cli_params = rc.get_parameters_by_source(ParameterSource.CLI)
        assert len(cli_params) == 1
        assert cli_params[0].path == "model"

    def test_get_non_default_parameters_excludes_pydantic_defaults(self):
        prov = _make_provenance(
            **{
                "model": (ParameterSource.CLI, "gpt2"),
                "backend": (ParameterSource.CONFIG_FILE, "pytorch"),
                "seed": (ParameterSource.PYDANTIC_DEFAULT, 42),
                "max_tokens": (ParameterSource.PYDANTIC_DEFAULT, 512),
            }
        )
        rc = ResolvedConfig(config=object(), provenance=prov)
        non_defaults = rc.get_non_default_parameters()
        paths = {p.path for p in non_defaults}
        assert "model" in paths
        assert "backend" in paths
        assert "seed" not in paths
        assert "max_tokens" not in paths

    def test_get_cli_overrides_returns_only_cli_params(self):
        prov = _make_provenance(
            **{
                "model": (ParameterSource.CLI, "gpt2"),
                "backend": (ParameterSource.CLI, "vllm"),
                "seed": (ParameterSource.CONFIG_FILE, 42),
            }
        )
        rc = ResolvedConfig(config=object(), provenance=prov)
        cli_overrides = rc.get_cli_overrides()
        paths = {p.path for p in cli_overrides}
        assert paths == {"model", "backend"}

    def test_to_summary_dict_structure(self):
        prov = _make_provenance(
            **{
                "model": (ParameterSource.CLI, "gpt2"),
                "seed": (ParameterSource.PYDANTIC_DEFAULT, 42),
            }
        )
        rc = ResolvedConfig(
            config=object(),
            provenance=prov,
            preset_chain=["quick-test"],
            config_file_path="/configs/exp.yaml",
        )
        summary = rc.to_summary_dict()
        assert summary["preset_chain"] == ["quick-test"]
        assert summary["config_file"] == "/configs/exp.yaml"
        assert "model" in summary["non_default_params"]
        assert "seed" not in summary["non_default_params"]
        assert "model" in summary["cli_overrides"]

    def test_to_summary_dict_with_empty_provenance(self):
        rc = ResolvedConfig(config=object())
        summary = rc.to_summary_dict()
        assert summary["preset_chain"] == []
        assert summary["config_file"] is None
        assert summary["non_default_params"] == {}
        assert summary["cli_overrides"] == []


# ---------------------------------------------------------------------------
# flatten_dict
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flat_dict_unchanged(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_single_nesting_level(self):
        d = {"a": {"b": 1, "c": 2}}
        assert flatten_dict(d) == {"a.b": 1, "a.c": 2}

    def test_deep_nesting(self):
        d = {"a": {"b": {"c": 42}}}
        assert flatten_dict(d) == {"a.b.c": 42}

    def test_mixed_depths(self):
        d = {"x": 1, "y": {"z": 2}}
        result = flatten_dict(d)
        assert result == {"x": 1, "y.z": 2}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep="/")
        assert result == {"a/b": 1}

    def test_empty_dict(self):
        assert flatten_dict({}) == {}

    def test_preserves_non_dict_values(self):
        d = {"a": [1, 2, 3], "b": None}
        result = flatten_dict(d)
        assert result == {"a": [1, 2, 3], "b": None}


# ---------------------------------------------------------------------------
# unflatten_dict
# ---------------------------------------------------------------------------


class TestUnflattenDict:
    def test_flat_keys_unchanged(self):
        d = {"a": 1, "b": 2}
        assert unflatten_dict(d) == {"a": 1, "b": 2}

    def test_single_nesting(self):
        d = {"a.b": 1, "a.c": 2}
        assert unflatten_dict(d) == {"a": {"b": 1, "c": 2}}

    def test_deep_nesting(self):
        d = {"a.b.c": 42}
        assert unflatten_dict(d) == {"a": {"b": {"c": 42}}}

    def test_mixed_depths(self):
        d = {"x": 1, "y.z": 2}
        assert unflatten_dict(d) == {"x": 1, "y": {"z": 2}}

    def test_empty_dict(self):
        assert unflatten_dict({}) == {}

    def test_roundtrip_with_flatten(self):
        """flatten then unflatten should recover the original structure."""
        original = {"decoder": {"temperature": 0.7, "top_k": 50}, "backend": "pytorch"}
        assert unflatten_dict(flatten_dict(original)) == original


# ---------------------------------------------------------------------------
# compare_dicts
# ---------------------------------------------------------------------------


class TestCompareDicts:
    def test_identical_dicts_have_no_changes(self):
        d = {"a": 1, "b": 2}
        changed, unchanged = compare_dicts(d, d)
        assert changed == {}
        assert "a" in unchanged and "b" in unchanged

    def test_detects_changed_values(self):
        base = {"a": 1, "b": 2}
        overlay = {"a": 99, "b": 2}
        changed, unchanged = compare_dicts(base, overlay)
        assert changed == {"a": 99}
        assert "b" in unchanged

    def test_detects_new_keys_in_overlay(self):
        base = {"a": 1}
        overlay = {"a": 1, "new_key": "value"}
        changed, unchanged = compare_dicts(base, overlay)
        assert "new_key" in changed
        assert changed["new_key"] == "value"

    def test_unchanged_keys_preserved(self):
        base = {"a": 1, "b": 2, "c": 3}
        overlay = {"a": 99, "b": 2, "c": 3}
        changed, unchanged = compare_dicts(base, overlay)
        assert set(unchanged.keys()) == {"b", "c"}

    def test_empty_dicts(self):
        changed, unchanged = compare_dicts({}, {})
        assert changed == {}
        assert unchanged == {}

    def test_overlay_none_value_not_treated_as_changed(self):
        """None values in overlay are not treated as overrides (per implementation)."""
        base = {"a": 1}
        overlay = {"a": None}
        changed, unchanged = compare_dicts(base, overlay)
        # None is falsy — implementation skips None overlays
        assert "a" not in changed
