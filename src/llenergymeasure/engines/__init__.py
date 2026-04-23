"""Inference engines for llenergymeasure."""

from llenergymeasure.config.engine_detection import is_engine_available
from llenergymeasure.config.ssot import Engine
from llenergymeasure.engines.protocol import EnginePlugin
from llenergymeasure.utils.exceptions import EngineError

__all__ = ["EnginePlugin", "build_config_probe", "detect_default_engine", "get_engine"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy re-export of ``build_config_probe`` to avoid package-init ordering."""
    if name == "build_config_probe":
        from llenergymeasure.engines.probe_adapter import build_config_probe

        return build_config_probe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def detect_default_engine() -> Engine:
    """Detect the default available inference engine.

    Returns 'transformers' if transformers is installed, 'tensorrt' if tensorrt_llm is
    installed (and not transformers), 'vllm' if vllm is installed.
    Priority: transformers > tensorrt > vllm.

    Raises:
        EngineError: If no supported engine is installed.
    """
    if is_engine_available(Engine.TRANSFORMERS):
        return Engine.TRANSFORMERS
    if is_engine_available(Engine.TENSORRT):
        return Engine.TENSORRT
    if is_engine_available(Engine.VLLM):
        return Engine.VLLM
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
    if name == Engine.TRANSFORMERS:
        from llenergymeasure.engines.transformers import TransformersEngine

        return TransformersEngine()
    if name == Engine.VLLM:
        from llenergymeasure.engines.vllm import VLLMEngine

        return VLLMEngine()
    if name == Engine.TENSORRT:
        from llenergymeasure.engines.tensorrt import TensorRTEngine

        return TensorRTEngine()
    raise EngineError(f"Unknown engine: {name!r}. Available: {', '.join(Engine)}")
