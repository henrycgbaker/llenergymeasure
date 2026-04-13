"""Inference backends for llenergymeasure."""

from llenergymeasure.backends.protocol import BackendPlugin
from llenergymeasure.config.backend_detection import is_backend_available
from llenergymeasure.config.ssot import BACKEND_PYTORCH, BACKEND_TENSORRT, BACKEND_VLLM
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
    if is_backend_available(BACKEND_PYTORCH):
        return BACKEND_PYTORCH
    if is_backend_available(BACKEND_TENSORRT):
        return BACKEND_TENSORRT
    if is_backend_available(BACKEND_VLLM):
        return BACKEND_VLLM
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
    if name == BACKEND_PYTORCH:
        from llenergymeasure.backends.pytorch import PyTorchBackend

        return PyTorchBackend()
    if name == BACKEND_VLLM:
        from llenergymeasure.backends.vllm import VLLMBackend

        return VLLMBackend()
    if name == BACKEND_TENSORRT:
        from llenergymeasure.backends.tensorrt import TensorRTBackend

        return TensorRTBackend()
    raise BackendError(f"Unknown backend: {name!r}. Available: pytorch, vllm, tensorrt")
