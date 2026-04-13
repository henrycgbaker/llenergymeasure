"""Unit tests for config.engine_detection.py.

Covers:
- is_engine_available() — tries importing the engine package
- get_available_engines() — filters KNOWN_ENGINES by availability
- get_engine_install_hint() — returns correct install hint strings

All import-level side effects are mocked via unittest.mock.patch; no real
engine packages are imported or required.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from llenergymeasure.config.engine_detection import (
    KNOWN_ENGINES,
    get_available_engines,
    get_engine_install_hint,
    is_engine_available,
)


@contextmanager
def _hide_module(name: str):
    """Temporarily poison *name* in sys.modules so imports see it as missing.

    Snapshots and restores **all** submodules (``name.*``) to prevent
    order-dependent failures when pytest-randomly reorders tests.
    """
    saved = {k: sys.modules[k] for k in list(sys.modules) if k == name or k.startswith(f"{name}.")}
    sys.modules[name] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        # Remove any entries added during the poisoned import attempt
        for k in list(sys.modules):
            if k == name or k.startswith(f"{name}."):
                sys.modules.pop(k, None)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# is_engine_available
# ---------------------------------------------------------------------------


class TestIsEngineAvailable:
    def test_returns_true_when_torch_importable(self):
        pytest.importorskip("torch")
        result = is_engine_available("transformers")
        assert result is True

    def test_returns_false_when_torch_not_importable(self):
        with _hide_module("torch"):
            result = is_engine_available("transformers")
        assert result is False

    def test_returns_false_when_vllm_not_importable(self):
        with _hide_module("vllm"):
            result = is_engine_available("vllm")
        assert result is False

    def test_returns_false_when_tensorrt_not_importable(self):
        with _hide_module("tensorrt_llm"):
            result = is_engine_available("tensorrt")
        assert result is False

    def test_returns_false_for_unknown_engine(self):
        result = is_engine_available("unknown_backend_xyz")
        assert result is False

    def test_returns_false_when_oserror_on_import(self):
        """OSError (e.g. missing .so) should be caught and return False."""
        import llenergymeasure.config.engine_detection as _bd_mod

        real_import = __import__

        def _raise_oserror(name, *args, **kwargs):
            if name == "tensorrt_llm":
                raise OSError("libcudart.so not found")
            return real_import(name, *args, **kwargs)

        original_builtins_import = _bd_mod.__builtins__.get("__import__")  # type: ignore[union-attr]
        with _hide_module("tensorrt_llm"):
            _bd_mod.__builtins__["__import__"] = _raise_oserror  # type: ignore[index]
            try:
                result = is_engine_available("tensorrt")
            finally:
                if original_builtins_import is not None:
                    _bd_mod.__builtins__["__import__"] = original_builtins_import  # type: ignore[index]
                else:
                    del _bd_mod.__builtins__["__import__"]  # type: ignore[attr-defined]

        assert result is False

    def test_handles_generic_exception_during_import(self):
        """Unexpected exception during import should be caught and return False."""
        import llenergymeasure.config.engine_detection as _bd_mod

        real_import = __import__

        def _raise_runtime(name, *args, **kwargs):
            if name == "vllm":
                raise RuntimeError("CUDA init failed")
            return real_import(name, *args, **kwargs)

        original_builtins_import = _bd_mod.__builtins__.get("__import__")  # type: ignore[union-attr]
        with _hide_module("vllm"):
            _bd_mod.__builtins__["__import__"] = _raise_runtime  # type: ignore[index]
            try:
                result = is_engine_available("vllm")
            finally:
                if original_builtins_import is not None:
                    _bd_mod.__builtins__["__import__"] = original_builtins_import  # type: ignore[index]
                else:
                    del _bd_mod.__builtins__["__import__"]  # type: ignore[attr-defined]

        assert result is False


# ---------------------------------------------------------------------------
# get_available_engines
# ---------------------------------------------------------------------------


class TestGetAvailableEngines:
    def test_returns_list_of_available_engines(self):
        with patch(
            "llenergymeasure.config.engine_detection.is_engine_available",
            side_effect=lambda b: b == "transformers",
        ):
            result = get_available_engines()

        assert result == ["transformers"]

    def test_returns_empty_list_when_none_available(self):
        with patch(
            "llenergymeasure.config.engine_detection.is_engine_available",
            return_value=False,
        ):
            result = get_available_engines()

        assert result == []

    def test_returns_all_when_all_available(self):
        with patch(
            "llenergymeasure.config.engine_detection.is_engine_available",
            return_value=True,
        ):
            result = get_available_engines()

        assert result == KNOWN_ENGINES

    def test_order_matches_known_engines_order(self):
        """get_available_engines preserves KNOWN_ENGINES order."""
        with patch(
            "llenergymeasure.config.engine_detection.is_engine_available",
            side_effect=lambda b: b in ("transformers", "vllm"),
        ):
            result = get_available_engines()

        assert result == ["transformers", "vllm"]

    def test_pytorch_comes_before_vllm_in_known_engines(self):
        """Locked decision: pytorch takes priority over vllm."""
        pytorch_idx = KNOWN_ENGINES.index("transformers")
        vllm_idx = KNOWN_ENGINES.index("vllm")
        assert pytorch_idx < vllm_idx


# ---------------------------------------------------------------------------
# get_engine_install_hint
# ---------------------------------------------------------------------------


class TestGetEngineInstallHint:
    def test_pytorch_hint_is_base_install(self):
        hint = get_engine_install_hint("transformers")
        assert "llenergymeasure" in hint
        # pytorch ships with the base package — no extra bracket needed
        assert "[" not in hint

    def test_vllm_hint_recommends_docker(self):
        hint = get_engine_install_hint("vllm")
        assert "Docker" in hint or "docker" in hint.lower()

    def test_tensorrt_hint_recommends_docker(self):
        hint = get_engine_install_hint("tensorrt")
        assert "Docker" in hint or "docker" in hint.lower()

    def test_unknown_engine_returns_pip_install_with_extra(self):
        hint = get_engine_install_hint("somebackend")
        assert "somebackend" in hint
        assert "pip install" in hint

    def test_all_known_engines_have_hints(self):
        for engine in KNOWN_ENGINES:
            hint = get_engine_install_hint(engine)
            assert isinstance(hint, str)
            assert len(hint) > 0
