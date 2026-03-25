"""llem config command — environment and configuration display.

Prints GPU info, installed backends, energy backend status, and user config
path. Designed to be the first thing a researcher runs to verify their
environment is correctly set up.

Always exits 0 — this command is purely informational.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Annotated, Any

import typer

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _probe_gpu() -> list[dict[str, Any]] | None:
    """Return list of GPU info dicts or None if unavailable."""
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        gpus: list[dict[str, Any]] = []
        with nvml_context():
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({"name": name, "vram_gb": mem.total / 1e9})
        return gpus if gpus else None
    except Exception:
        return None


from llenergymeasure.config.ssot import BACKEND_PACKAGES


def _probe_backend_version(backend: str) -> str | None:
    """Try to retrieve version string for an installed inference backend."""
    try:
        if backend == "pytorch":
            import torch

            return str(torch.__version__)
        elif backend == "vllm":
            import vllm

            return str(vllm.__version__)
        elif backend == "tensorrt":
            import tensorrt_llm

            return str(tensorrt_llm.__version__)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def config_command(
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v=INFO, -vv=DEBUG)"),
    ] = 0,
) -> None:
    """Show environment and configuration status."""
    from llenergymeasure.cli import _setup_logging

    _setup_logging(verbose)
    # --- GPU ---
    print("GPU")
    gpus = _probe_gpu()
    if gpus:
        for gpu in gpus:
            print(f"  {gpu['name']}  {gpu['vram_gb']:.1f} GB")
        if verbose > 0:
            try:
                import pynvml

                from llenergymeasure.device.gpu_info import nvml_context

                driver: str | None = None
                with nvml_context():
                    raw = pynvml.nvmlSystemGetDriverVersion()
                    driver = raw.decode() if isinstance(raw, bytes) else str(raw)
                if driver:
                    print(f"  Driver: {driver}")
            except Exception:
                pass
    else:
        print("  No GPU detected")

    # --- Inference backends ---
    print("Backends")
    for backend, package in BACKEND_PACKAGES.items():
        installed = importlib.util.find_spec(package) is not None
        if installed:
            print(f"  {backend}: installed", end="")
            if verbose > 0:
                version = _probe_backend_version(backend)
                if version:
                    print(f"  ({version})", end="")
            print()
        else:
            print(f"  {backend}: not installed  (pip install llenergymeasure[{backend}])")

    # --- Energy backends ---
    print("Energy")
    has_nvml = importlib.util.find_spec("pynvml") is not None
    has_zeus = importlib.util.find_spec("zeus") is not None
    has_codecarbon = importlib.util.find_spec("codecarbon") is not None

    if has_zeus:
        selected = "zeus"
    elif has_nvml:
        selected = "nvml"
    elif has_codecarbon:
        selected = "codecarbon"
    else:
        selected = None

    if selected:
        print(f"  Energy: {selected}")
    else:
        print("  Energy: none (install nvidia-ml-py for NVML)")

    # --- User config ---
    print("Config")
    from llenergymeasure.config.user_config import get_user_config_path

    config_path = get_user_config_path()
    print(f"  Path: {config_path}")
    if config_path.exists():
        print("  Status: loaded")
        if verbose > 0:
            from llenergymeasure.config.user_config import (
                UserConfig,
                load_user_config,
            )

            try:
                user_cfg = load_user_config()
                defaults = UserConfig()
                cfg_dump = user_cfg.model_dump()
                default_dump = defaults.model_dump()
                non_defaults: dict[str, Any] = {}
                for section, values in cfg_dump.items():
                    section_defaults = default_dump.get(section, {})
                    if isinstance(values, dict) and isinstance(section_defaults, dict):
                        diff = {k: v for k, v in values.items() if v != section_defaults.get(k)}
                        if diff:
                            non_defaults[section] = diff
                    elif values != section_defaults:
                        non_defaults[section] = values
                if non_defaults:
                    print("  Non-default values:")
                    for section, values in non_defaults.items():
                        if isinstance(values, dict):
                            for k, v in values.items():
                                print(f"    {section}.{k}: {v}")
                        else:
                            print(f"    {section}: {values}")
                else:
                    print("  (all values are defaults)")
            except Exception as e:
                print(f"  (could not load config: {e})")
    else:
        print("  Status: using defaults (no config file)")

    # --- Python ---
    print("Python")
    print(f"  {sys.version.split()[0]}")
