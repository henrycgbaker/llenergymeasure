"""Unit tests for domain/progress.py — ProgressCallback protocol and step vocabulary."""

from __future__ import annotations

from llenergymeasure.domain.progress import (
    STEP_BASELINE,
    STEP_CONTAINER_PREFLIGHT,
    STEP_CONTAINER_START,
    STEP_IMAGE_CHECK,
    STEP_LABELS,
    STEP_MEASURE,
    STEP_MODEL,
    STEP_PHASES,
    STEP_PREFLIGHT,
    STEP_PULL,
    STEP_SAVE,
    STEP_WARMUP,
    ProgressCallback,
)


def test_protocol_is_runtime_checkable():
    """ProgressCallback can be used with isinstance() at runtime."""

    class Impl:
        def on_step_start(self, step: str, description: str, detail: str = "") -> None:
            pass

        def on_step_update(self, step: str, detail: str) -> None:
            pass

        def on_step_done(self, step: str, elapsed_sec: float) -> None:
            pass

        def on_step_skip(self, step: str, reason: str = "") -> None:
            pass

    assert isinstance(Impl(), ProgressCallback)


def test_non_conforming_class_fails_protocol():
    """A class missing methods does not satisfy the protocol."""

    class Incomplete:
        def on_step_start(self, step: str, description: str, detail: str = "") -> None:
            pass

        # Missing on_step_update and on_step_done

    assert not isinstance(Incomplete(), ProgressCallback)


def test_step_labels_cover_all_constants():
    """Every step constant has a corresponding label."""
    for step in [
        STEP_PREFLIGHT,
        STEP_IMAGE_CHECK,
        STEP_PULL,
        STEP_CONTAINER_START,
        STEP_CONTAINER_PREFLIGHT,
        STEP_BASELINE,
        STEP_MODEL,
        STEP_WARMUP,
        STEP_MEASURE,
        STEP_SAVE,
    ]:
        assert step in STEP_LABELS, f"Missing label for step: {step}"


def test_step_labels_are_verbs():
    """Labels should be action-oriented (basic smoke test)."""
    assert STEP_LABELS[STEP_PREFLIGHT] == "Checking"
    assert STEP_LABELS[STEP_MODEL] == "Loading"
    assert STEP_LABELS[STEP_MEASURE] == "Measuring"


def test_step_phases_cover_all_constants():
    """Every step constant has a phase assignment."""
    for step in [
        STEP_PREFLIGHT,
        STEP_IMAGE_CHECK,
        STEP_PULL,
        STEP_CONTAINER_START,
        STEP_CONTAINER_PREFLIGHT,
        STEP_BASELINE,
        STEP_MODEL,
        STEP_WARMUP,
        STEP_MEASURE,
        STEP_SAVE,
    ]:
        assert step in STEP_PHASES, f"Missing phase for step: {step}"
