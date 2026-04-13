"""Inference engines for llenergymeasure."""

from llenergymeasure.config.engine_detection import is_engine_available
from llenergymeasure.config.ssot import ENGINE_TENSORRT, ENGINE_TRANSFORMERS, ENGINE_VLLM
from llenergymeasure.engines.protocol import EnginePlugin
from llenergymeasure.utils.exceptions import EngineError

__all__ = ["EnginePlugin", "detect_default_engine", "get_engine"]


def detect_default_engine() -> str:
    """Detect the default available inference engine.

    Returns 'transformers' if transformers is installed, 'tensorrt' if tensorrt_llm is
    installed (and not transformers), 'vllm' if vllm is installed.
    Priority: transformers > tensorrt > vllm.

    Raises:
        EngineError: If no supported engine is installed.
    """
    if is_engine_available(ENGINE_TRANSFORMERS):
        return ENGINE_TRANSFORMERS
    if is_engine_available(ENGINE_TENSORRT):
        return ENGINE_TENSORRT
    if is_engine_available(ENGINE_VLLM):
        return ENGINE_VLLM
    raise EngineError(
        "No inference engine installed. Install one with: "
        "pip install llenergymeasure[transformers], "
        "pip install llenergymeasure[vllm], or "
        "pip install llenergymeasure[tensorrt]"
    )


def get_engine(name: str) -> EnginePlugin:
    """Get an inference engine instance by name.

    Args:
        name: Engine name ('transformers', 'vllm', 'tensorrt').

    Returns:
        An EnginePlugin instance.

    Raises:
        EngineError: If the engine name is unknown.
    """
    if name == ENGINE_TRANSFORMERS:
        from llenergymeasure.engines.transformers import TransformersEngine

        return TransformersEngine()
    if name == ENGINE_VLLM:
        from llenergymeasure.engines.vllm import VLLMEngine

        return VLLMEngine()
    if name == ENGINE_TENSORRT:
        from llenergymeasure.engines.tensorrt import TensorRTEngine

        return TensorRTEngine()
    raise EngineError(f"Unknown engine: {name!r}. Available: transformers, vllm, tensorrt")
