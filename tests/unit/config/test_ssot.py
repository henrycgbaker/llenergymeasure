"""Unit tests for SSOT engine constants."""

from llenergymeasure.config.ssot import (
    ALL_ENGINES,
    ENGINE_TENSORRT,
    ENGINE_TRANSFORMERS,
    ENGINE_VLLM,
)


def test_all_engines_has_three_entries() -> None:
    assert len(ALL_ENGINES) == 3


def test_all_engines_contents() -> None:
    assert set(ALL_ENGINES) == {ENGINE_TRANSFORMERS, ENGINE_VLLM, ENGINE_TENSORRT}


def test_all_engines_is_frozenset() -> None:
    assert isinstance(ALL_ENGINES, frozenset)


def test_all_engines_membership() -> None:
    assert ENGINE_TRANSFORMERS in ALL_ENGINES
    assert ENGINE_VLLM in ALL_ENGINES
    assert ENGINE_TENSORRT in ALL_ENGINES
    assert "unknown_engine" not in ALL_ENGINES
