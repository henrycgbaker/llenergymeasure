"""Runtime engine availability detection."""

from __future__ import annotations

from llenergymeasure.config.ssot import ENGINE_TENSORRT, ENGINE_TRANSFORMERS, ENGINE_VLLM

KNOWN_ENGINES: list[str] = [ENGINE_TRANSFORMERS, ENGINE_VLLM, ENGINE_TENSORRT]


def is_engine_available(engine: str) -> bool:
    """Check if an engine is available (installed and importable).

    Args:
        engine: Engine name ("transformers", "vllm", or "tensorrt").

    Returns:
        True if engine is importable, False otherwise.
    """
    try:
        if engine == ENGINE_TRANSFORMERS:
            import torch  # noqa: F401
        elif engine == ENGINE_VLLM:
            import vllm  # noqa: F401
        elif engine == ENGINE_TENSORRT:
            import tensorrt_llm  # noqa: F401
        else:
            # Unknown engine
            return False
        return True
    except (ImportError, OSError, Exception):
        # ImportError: package not installed
        # OSError: library dependency missing (tensorrt_llm on some systems)
        # Exception: catch-all for any other import-time errors
        return False


def get_available_engines() -> list[str]:
    """Get list of all available engines on the system.

    Returns:
        List of engine names that are installed and importable.
    """
    return [e for e in KNOWN_ENGINES if is_engine_available(e)]


def get_engine_install_hint(engine: str) -> str:
    """Get installation command for an engine.

    Args:
        engine: Engine name.

    Returns:
        pip install command string for the engine.
    """
    hints = {
        "transformers": "pip install llenergymeasure",
        "vllm": "Docker recommended — see docs/deployment.md",
        "tensorrt": "Docker recommended — see docs/deployment.md",
    }
    return hints.get(engine, f"pip install llenergymeasure[{engine}]")
