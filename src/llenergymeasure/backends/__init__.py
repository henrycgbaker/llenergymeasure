"""Inference backends for llenergymeasure."""

import importlib.util

from llenergymeasure.backends.protocol import BackendPlugin
from llenergymeasure.utils.exceptions import BackendError

__all__ = ["BackendPlugin", "detect_default_backend", "get_backend"]


def detect_default_backend() -> str:
    """Detect the default available backend.

    Returns 'pytorch' if transformers is installed, 'tensorrt' if tensorrt_llm is
    installed (and not transformers), 'vllm' if vllm is installed.
    Priority: pytorch > tensorrt > vllm.

    Raises:
        BackendError: If no supported backend is installed.
    """
    if importlib.util.find_spec("transformers") is not None:
        return "pytorch"
    if importlib.util.find_spec("tensorrt_llm") is not None:
        return "tensorrt"
    if importlib.util.find_spec("vllm") is not None:
        return "vllm"
    raise BackendError(
        "No inference backend installed. Install one with: "
        "pip install llenergymeasure[pytorch], "
        "pip install llenergymeasure[vllm], or "
        "pip install llenergymeasure[tensorrt]"
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
        from llenergymeasure.backends.pytorch import PyTorchBackend

        return PyTorchBackend()
    if name == "vllm":
        from llenergymeasure.backends.vllm import VLLMBackend

        return VLLMBackend()
    if name == "tensorrt":
        from llenergymeasure.backends.tensorrt import TensorRTBackend

        return TensorRTBackend()
    raise BackendError(f"Unknown backend: {name!r}. Available: pytorch, vllm, tensorrt")
