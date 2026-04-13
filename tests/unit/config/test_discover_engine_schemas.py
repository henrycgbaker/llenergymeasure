"""Tests for scripts/discover_engine_schemas.py — pure-Python helpers only.

Container-gated end-to-end discovery tests would use @pytest.mark.docker and
live in a separate test file if added later. This module tests the helpers
that power the discovery script without requiring any engine package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

import pytest

# Load scripts/discover_engine_schemas.py as a module.
REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "discover_engine_schemas.py"
_spec = importlib.util.spec_from_file_location("_discover_engine_schemas", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
discover = importlib.util.module_from_spec(_spec)
sys.modules["_discover_engine_schemas"] = discover
_spec.loader.exec_module(discover)


# ---------------------------------------------------------------------------
# _annotation_to_type_str
# ---------------------------------------------------------------------------


def test_type_str_simple_primitives() -> None:
    assert discover._annotation_to_type_str(int) == "int"
    assert discover._annotation_to_type_str(str) == "str"
    assert discover._annotation_to_type_str(bool) == "bool"


def test_type_str_none_type() -> None:
    assert discover._annotation_to_type_str(type(None)) == "None"


def test_type_str_pep604_union() -> None:
    assert discover._annotation_to_type_str(int | None) == "int | None"
    assert discover._annotation_to_type_str(int | str | None) == "int | str | None"


def test_type_str_typing_optional() -> None:
    # Deliberately exercising the legacy typing.Optional form — discovery sees
    # this syntax in third-party code even if we prefer X | None ourselves.
    from typing import Optional

    assert discover._annotation_to_type_str(Optional[int]) == "int | None"  # noqa: UP045


def test_type_str_typing_union() -> None:
    # Ditto for typing.Union — third-party engine packages still use it.
    from typing import Union

    assert discover._annotation_to_type_str(Union[int, str]) == "int | str"  # noqa: UP007


def test_type_str_generic_list_dict() -> None:
    assert discover._annotation_to_type_str(list[str]) == "list[str]"
    assert discover._annotation_to_type_str(dict[str, int]) == "dict[str, int]"
    assert discover._annotation_to_type_str(list[dict[str, int]]) == "list[dict[str, int]]"


def test_type_str_literal() -> None:
    assert discover._annotation_to_type_str(Literal["a", "b"]) == "Literal['a', 'b']"


def test_type_str_empty_means_unknown() -> None:
    import inspect

    assert discover._annotation_to_type_str(inspect.Parameter.empty) == "unknown"
    assert discover._annotation_to_type_str(inspect.Signature.empty) == "unknown"


# ---------------------------------------------------------------------------
# _read_dockerfile_from
# ---------------------------------------------------------------------------


def test_read_dockerfile_single_stage(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("ARG FOO_VERSION=1.2.3\nFROM foo/foo:${FOO_VERSION}\n")
    assert discover._read_dockerfile_from(df) == "foo/foo:1.2.3"


def test_read_dockerfile_prefers_runtime_stage(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text(
        "ARG DEVEL=a:1-devel\n"
        "ARG RUNTIME=a:1-runtime\n"
        "FROM foo:${DEVEL} AS builder\n"
        "FROM foo:${RUNTIME} AS runtime\n"
        "FROM runtime AS dev\n"
    )
    assert discover._read_dockerfile_from(df) == "foo:a:1-runtime"


def test_read_dockerfile_no_runtime_stage_falls_back(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM foo:1 AS builder\n"
        "FROM bar:2 AS packager\n"
        "FROM builder\n"  # references prior stage — should be skipped
    )
    # No `AS runtime` → first external FROM wins (foo:1)
    assert discover._read_dockerfile_from(df) == "foo:1"


def test_read_dockerfile_expands_only_default_args(tmp_path: Path, monkeypatch) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("ARG VER=default\nFROM foo:${VER} AS runtime\n")
    monkeypatch.setenv("VER", "from-env")  # must be ignored
    assert discover._read_dockerfile_from(df) == "foo:default"


def test_read_dockerfile_no_from_raises(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("ARG X=1\n# no FROM\n")
    with pytest.raises(ValueError, match="No FROM directive"):
        discover._read_dockerfile_from(df)


def test_read_dockerfile_against_real_dockerfiles() -> None:
    vllm_from = discover._read_dockerfile_from(REPO_ROOT / "docker/Dockerfile.vllm")
    assert vllm_from.startswith("vllm/vllm-openai:")

    trt_from = discover._read_dockerfile_from(REPO_ROOT / "docker/Dockerfile.tensorrt")
    assert trt_from.startswith("nvcr.io/nvidia/tensorrt-llm/release:")

    tx_from = discover._read_dockerfile_from(REPO_ROOT / "docker/Dockerfile.transformers")
    # runtime stage uses non-devel tag
    assert "pytorch/pytorch:" in tx_from and "devel" not in tx_from


# ---------------------------------------------------------------------------
# _jsonable
# ---------------------------------------------------------------------------


def test_jsonable_primitives_passthrough() -> None:
    for v in (None, True, 1, 1.5, "x"):
        assert discover._jsonable(v) == v


def test_jsonable_sets_sorted_list() -> None:
    assert discover._jsonable({3, 1, 2}) == [1, 2, 3]


def test_jsonable_tuple_to_list() -> None:
    assert discover._jsonable((1, "a", None)) == [1, "a", None]


def test_jsonable_nested_dict() -> None:
    got = discover._jsonable({"k": (1, {2, 3})})
    assert got == {"k": [1, [2, 3]]}


def test_jsonable_type_to_name() -> None:
    assert discover._jsonable(int) == "int"


def test_jsonable_fallback_to_str() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert discover._jsonable(Opaque()) == "<opaque>"


# ---------------------------------------------------------------------------
# Envelope shape
# ---------------------------------------------------------------------------


def test_make_envelope_fills_required_keys() -> None:
    env = discover._make_envelope(
        engine="vllm",
        engine_version="0.7.3",
        engine_commit_sha=None,
        image_ref="foo:1",
        base_image_ref="foo:1",
        discovery_method="unit test",
        discovery_limitations=[],
        engine_params={"a": {"type": "int", "default": 0}},
        sampling_params={"b": {"type": "str", "default": ""}},
    )
    assert env["schema_version"] == discover.SCHEMA_VERSION
    assert env["engine"] == "vllm"
    assert env["discovered_at"]  # ISO string
    assert "engine_params" in env and "sampling_params" in env


def test_schema_version_is_semver_with_major_one() -> None:
    major = int(discover.SCHEMA_VERSION.split(".")[0])
    assert major == 1, "PR 48.1 ships schema_version 1.x; bumping major requires loader update"
