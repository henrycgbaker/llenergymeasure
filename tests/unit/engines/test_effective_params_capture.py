"""Tests for :func:`extract_effective_params` and the per-engine capture path.

PoC-C finding (sweep-dedup.md §3.2): the extractor must strip private
(``_``-prefixed) fields by default — transformers and vLLM both leak
state that would poison H3 otherwise. These tests verify the default
behaviour and the allowlist escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from llenergymeasure.engines._helpers import extract_effective_params

# ---------------------------------------------------------------------------
# Dispatch behaviour — one per native-type shape
# ---------------------------------------------------------------------------


class _PydanticShape(BaseModel):
    a: int = 1
    b: str = "x"


@dataclass
class _DataclassShape:
    a: int = 1
    b: str = "x"


class _SlotsShape:
    __slots__ = ("a", "b")

    def __init__(self) -> None:
        self.a = 1
        self.b = "x"


class _DictShape:
    def __init__(self) -> None:
        self.a = 1
        self.b = "x"


class TestDispatch:
    def test_pydantic(self):
        out = extract_effective_params(_PydanticShape())
        assert out == {"a": 1, "b": "x"}

    def test_dataclass(self):
        out = extract_effective_params(_DataclassShape())
        assert out == {"a": 1, "b": "x"}

    def test_slots(self):
        out = extract_effective_params(_SlotsShape())
        assert out == {"a": 1, "b": "x"}

    def test_dict_fallback(self):
        out = extract_effective_params(_DictShape())
        assert out == {"a": 1, "b": "x"}


# ---------------------------------------------------------------------------
# Private-field handling (PoC-C finding)
# ---------------------------------------------------------------------------


class _LeakyPydantic(BaseModel):
    """Models transformers GenerationConfig leaking private state."""

    model_config = {"extra": "allow"}

    temperature: float = 1.0


class _LeakyDataclass:
    """Models a dataclass-ish leak via __dict__ fallback."""

    def __init__(self) -> None:
        self.temperature = 1.0
        self._commit_hash = "deadbeef"
        self._from_model_config = True


class TestPrivateFieldStripping:
    def test_pydantic_private_fields_stripped_by_default(self):
        obj = _LeakyPydantic(temperature=0.7, _commit_hash="abc", _from_model_config=True)
        out = extract_effective_params(obj)
        assert out == {"temperature": 0.7}

    def test_dict_private_fields_stripped_by_default(self):
        obj = _LeakyDataclass()
        out = extract_effective_params(obj)
        assert out == {"temperature": 1.0}

    def test_allowlist_preserves_private_field(self):
        obj = _LeakyPydantic(temperature=0.7, _commit_hash="abc")
        out = extract_effective_params(obj, private_field_allowlist={"_commit_hash"})
        assert out == {"temperature": 0.7, "_commit_hash": "abc"}


# ---------------------------------------------------------------------------
# Byte stability — same kwargs → same dump
# ---------------------------------------------------------------------------


class TestByteStability:
    def test_repeat_constructions_same_output(self):
        a = extract_effective_params(_PydanticShape(a=5, b="hi"))
        b = extract_effective_params(_PydanticShape(a=5, b="hi"))
        assert a == b

    def test_different_kwargs_different_output(self):
        a = extract_effective_params(_PydanticShape(a=5, b="hi"))
        b = extract_effective_params(_PydanticShape(a=6, b="hi"))
        assert a != b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_object_returns_empty_dict(self):
        class _Empty:
            pass

        out = extract_effective_params(_Empty())
        assert out == {}

    def test_model_dump_type_error_falls_through(self):
        # A model_dump that only accepts no args should still work via the
        # fallback path (TypeError on model_dump(mode=...) → retry with no kwargs).
        class _ModelDumpNoArgs:
            def model_dump(self) -> dict[str, Any]:
                return {"x": 1}

        out = extract_effective_params(_ModelDumpNoArgs())
        assert out == {"x": 1}
