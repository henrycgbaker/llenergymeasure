"""Inference backends for llenergymeasure."""

import importlib.util

from llenergymeasure.core.backends.protocol import BackendPlugin, InferenceBackend
from llenergymeasure.exceptions import BackendError

__all__ = ["BackendPlugin", "InferenceBackend", "detect_default_backend", "get_backend"]


def detect_default_backend() -> str:
    """Detect the default available backend.

    Returns 'pytorch' if transformers is installed, 'vllm' if vllm is installed.
    PyTorch takes priority over vLLM if both are installed.

    Raises:
        BackendError: If no supported backend is installed.
    """
    if importlib.util.find_spec("transformers") is not None:
        return "pytorch"
    if importlib.util.find_spec("vllm") is not None:
        return "vllm"
    raise BackendError(
        "No inference backend installed. Install one with: "
        "pip install llenergymeasure[pytorch] or pip install llenergymeasure[vllm]"
    )


def get_backend(name: str) -> BackendPlugin:
    """Get an inference backend instance by name.

    Args:
        name: Backend name ('pytorch', 'vllm', 'tensorrt').

    Returns:
        A BackendPlugin instance.

    Raises:
        BackendError: If the backend name is unknown.
    """
    if name == "pytorch":
        from llenergymeasure.core.backends.pytorch import PyTorchBackend

        return PyTorchBackend()
    if name == "vllm":
        from llenergymeasure.core.backends.vllm import VLLMBackend

        return VLLMBackend()
    raise BackendError(f"Unknown backend: {name!r}. Available: pytorch, vllm")
