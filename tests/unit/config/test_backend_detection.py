"""Unit tests for config/backend_detection.py.

Covers:
- is_backend_available() — tries importing the backend package
- get_available_backends() — filters KNOWN_BACKENDS by availability
- get_backend_install_hint() — returns correct install hint strings

All import-level side effects are mocked via unittest.mock.patch; no real
backend packages are imported or required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llenergymeasure.config.backend_detection import (
    KNOWN_BACKENDS,
    get_available_backends,
    get_backend_install_hint,
    is_backend_available,
)

# ---------------------------------------------------------------------------
# is_backend_available
# ---------------------------------------------------------------------------


class TestIsBackendAvailable:
    def test_returns_true_when_torch_importable(self):
        pytest.importorskip("torch")
        result = is_backend_available("pytorch")
        assert result is True

    def test_returns_false_when_torch_not_importable(self):
        import sys

        # Temporarily hide torch from sys.modules so the deferred import inside
        # is_backend_available() sees it as missing — no global builtins patch needed.
        torch_mod = sys.modules.pop("torch", None)
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            result = is_backend_available("pytorch")
        finally:
            sys.modules.pop("torch", None)
            if torch_mod is not None:
                sys.modules["torch"] = torch_mod

        assert result is False

    def test_returns_false_when_vllm_not_importable(self):
        import sys

        vllm_mod = sys.modules.pop("vllm", None)
        sys.modules["vllm"] = None  # type: ignore[assignment]
        try:
            result = is_backend_available("vllm")
        finally:
            sys.modules.pop("vllm", None)
            if vllm_mod is not None:
                sys.modules["vllm"] = vllm_mod

        assert result is False

    def test_returns_false_when_tensorrt_not_importable(self):
        import sys

        trt_mod = sys.modules.pop("tensorrt_llm", None)
        sys.modules["tensorrt_llm"] = None  # type: ignore[assignment]
        try:
            result = is_backend_available("tensorrt")
        finally:
            sys.modules.pop("tensorrt_llm", None)
            if trt_mod is not None:
                sys.modules["tensorrt_llm"] = trt_mod

        assert result is False

    def test_returns_false_for_unknown_backend(self):
        result = is_backend_available("unknown_backend_xyz")
        assert result is False

    def test_returns_false_when_oserror_on_import(self):
        """OSError (e.g. missing .so) should be caught and return False."""
        import sys

        import llenergymeasure.config.backend_detection as _bd_mod

        # Temporarily remove tensorrt_llm from sys.modules so the bare `import tensorrt_llm`
        # inside is_backend_available() is re-evaluated. Then patch __import__ narrowly on
        # the backend_detection module's builtins to raise OSError for that one module.
        trt_mod = sys.modules.pop("tensorrt_llm", None)

        real_import = __import__

        def _raise_oserror(name, *args, **kwargs):
            if name == "tensorrt_llm":
                raise OSError("libcudart.so not found")
            return real_import(name, *args, **kwargs)

        # Patch __import__ on the backend_detection module's builtins dict (narrow scope)
        original_builtins_import = _bd_mod.__builtins__.get("__import__")  # type: ignore[union-attr]
        _bd_mod.__builtins__["__import__"] = _raise_oserror  # type: ignore[index]
        try:
            result = is_backend_available("tensorrt")
        finally:
            if original_builtins_import is not None:
                _bd_mod.__builtins__["__import__"] = original_builtins_import  # type: ignore[index]
            else:
                del _bd_mod.__builtins__["__import__"]  # type: ignore[attr-defined]
            if trt_mod is not None:
                sys.modules["tensorrt_llm"] = trt_mod

        assert result is False

    def test_handles_generic_exception_during_import(self):
        """Unexpected exception during import should be caught and return False."""
        import sys

        import llenergymeasure.config.backend_detection as _bd_mod

        vllm_mod = sys.modules.pop("vllm", None)

        real_import = __import__

        def _raise_runtime(name, *args, **kwargs):
            if name == "vllm":
                raise RuntimeError("CUDA init failed")
            return real_import(name, *args, **kwargs)

        original_builtins_import = _bd_mod.__builtins__.get("__import__")  # type: ignore[union-attr]
        _bd_mod.__builtins__["__import__"] = _raise_runtime  # type: ignore[index]
        try:
            result = is_backend_available("vllm")
        finally:
            if original_builtins_import is not None:
                _bd_mod.__builtins__["__import__"] = original_builtins_import  # type: ignore[index]
            else:
                del _bd_mod.__builtins__["__import__"]  # type: ignore[attr-defined]
            if vllm_mod is not None:
                sys.modules["vllm"] = vllm_mod

        assert result is False


# ---------------------------------------------------------------------------
# get_available_backends
# ---------------------------------------------------------------------------


class TestGetAvailableBackends:
    def test_returns_list_of_available_backends(self):
        with patch(
            "llenergymeasure.config.backend_detection.is_backend_available",
            side_effect=lambda b: b == "pytorch",
        ):
            result = get_available_backends()

        assert result == ["pytorch"]

    def test_returns_empty_list_when_none_available(self):
        with patch(
            "llenergymeasure.config.backend_detection.is_backend_available",
            return_value=False,
        ):
            result = get_available_backends()

        assert result == []

    def test_returns_all_when_all_available(self):
        with patch(
            "llenergymeasure.config.backend_detection.is_backend_available",
            return_value=True,
        ):
            result = get_available_backends()

        assert result == KNOWN_BACKENDS

    def test_order_matches_known_backends_order(self):
        """get_available_backends preserves KNOWN_BACKENDS order."""
        with patch(
            "llenergymeasure.config.backend_detection.is_backend_available",
            side_effect=lambda b: b in ("pytorch", "vllm"),
        ):
            result = get_available_backends()

        assert result == ["pytorch", "vllm"]

    def test_pytorch_comes_before_vllm_in_known_backends(self):
        """Locked decision: pytorch takes priority over vllm."""
        pytorch_idx = KNOWN_BACKENDS.index("pytorch")
        vllm_idx = KNOWN_BACKENDS.index("vllm")
        assert pytorch_idx < vllm_idx


# ---------------------------------------------------------------------------
# get_backend_install_hint
# ---------------------------------------------------------------------------


class TestGetBackendInstallHint:
    def test_pytorch_hint_is_base_install(self):
        hint = get_backend_install_hint("pytorch")
        assert "llenergymeasure" in hint
        # pytorch ships with the base package — no extra bracket needed
        assert "[" not in hint

    def test_vllm_hint_recommends_docker(self):
        hint = get_backend_install_hint("vllm")
        assert "Docker" in hint or "docker" in hint.lower()

    def test_tensorrt_hint_recommends_docker(self):
        hint = get_backend_install_hint("tensorrt")
        assert "Docker" in hint or "docker" in hint.lower()

    def test_unknown_backend_returns_pip_install_with_extra(self):
        hint = get_backend_install_hint("somebackend")
        assert "somebackend" in hint
        assert "pip install" in hint

    def test_all_known_backends_have_hints(self):
        for backend in KNOWN_BACKENDS:
            hint = get_backend_install_hint(backend)
            assert isinstance(hint, str)
            assert len(hint) > 0
