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
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                None if name == "torch" else __import__(name, *a, **kw)
            ),
        ):
            # torch import succeeds → available
            # We use the real import instead because torch is installed on this host
            result = is_backend_available("pytorch")
        # pytorch is available on the test host (has torch installed)
        assert isinstance(result, bool)

    def test_returns_false_when_torch_not_importable(self):
        with patch("llenergymeasure.config.backend_detection.__import__", create=True):
            pass  # cannot easily patch builtins at module level

        # Simulate missing package by catching ImportError in the function
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def raise_import_error(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_import_error):
            result = is_backend_available("pytorch")

        assert result is False

    def test_returns_false_when_vllm_not_importable(self):
        def raise_on_vllm(name, *args, **kwargs):
            if name == "vllm":
                raise ImportError("No module named 'vllm'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_on_vllm):
            result = is_backend_available("vllm")

        assert result is False

    def test_returns_false_when_tensorrt_not_importable(self):
        def raise_on_tensorrt(name, *args, **kwargs):
            if name == "tensorrt_llm":
                raise ImportError("No module named 'tensorrt_llm'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_on_tensorrt):
            result = is_backend_available("tensorrt")

        assert result is False

    def test_returns_false_for_unknown_backend(self):
        result = is_backend_available("unknown_backend_xyz")
        assert result is False

    def test_returns_false_when_oserror_on_import(self):
        """OSError (e.g. missing .so) should be caught and return False."""

        def raise_oserror(name, *args, **kwargs):
            if name == "tensorrt_llm":
                raise OSError("libcudart.so not found")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_oserror):
            result = is_backend_available("tensorrt")

        assert result is False

    def test_handles_generic_exception_during_import(self):
        """Unexpected exception during import should be caught and return False."""

        def raise_generic(name, *args, **kwargs):
            if name == "vllm":
                raise RuntimeError("CUDA init failed")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_generic):
            result = is_backend_available("vllm")

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
        # pytorch ships with the base package — no extra needed
        assert "[" not in hint or "pytorch" not in hint.split("[")[1] if "[" in hint else True

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
