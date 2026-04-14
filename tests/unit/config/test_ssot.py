"""Unit tests for SSOT engine constants."""

from enum import StrEnum

from llenergymeasure.config.ssot import ALL_ENGINES, Engine


def test_engine_is_str_enum() -> None:
    assert issubclass(Engine, StrEnum)


def test_engine_members_are_strings() -> None:
    assert Engine.TRANSFORMERS == "transformers"
    assert Engine.VLLM == "vllm"
    assert Engine.TENSORRT == "tensorrt"


def test_engine_members_iterate_in_definition_order() -> None:
    assert tuple(Engine) == (Engine.TRANSFORMERS, Engine.VLLM, Engine.TENSORRT)


def test_all_engines_has_three_entries() -> None:
    assert len(ALL_ENGINES) == 3


def test_all_engines_derives_from_engine_enum() -> None:
    assert set(ALL_ENGINES) == set(Engine)


def test_all_engines_is_frozenset() -> None:
    assert isinstance(ALL_ENGINES, frozenset)


def test_all_engines_membership_accepts_strings() -> None:
    # StrEnum: string literals hash equal to members, so membership works for both
    assert "transformers" in ALL_ENGINES
    assert Engine.VLLM in ALL_ENGINES
    assert "unknown_engine" not in ALL_ENGINES
