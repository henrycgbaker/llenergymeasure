"""GPU-free unit tests for EnvironmentSnapshot and related detection functions.

All tests run without a GPU. torch, subprocess, and huggingface_hub are never
directly imported — access is always via monkeypatch.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import llenergymeasure.domain.environment as env_module
from llenergymeasure.domain.environment import (
    EnvironmentSnapshot,
    _capture_conda_list,
    _capture_pip_freeze,
    detect_cuda_version_with_source,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hardware() -> env_module.EnvironmentMetadata:
    """Build a minimal EnvironmentMetadata for testing."""
    return env_module.EnvironmentMetadata(
        gpu=env_module.GPUEnvironment(name="NVIDIA A100-SXM4-80GB", vram_total_mb=81920),
        cuda=env_module.CUDAEnvironment(version="12.4", driver_version="535.104"),
        thermal=env_module.ThermalEnvironment(),
        cpu=env_module.CPUEnvironment(platform="Linux"),
        container=env_module.ContainerEnvironment(),
        collected_at=datetime(2026, 1, 1, 12, 0, 0),
    )


# ---------------------------------------------------------------------------
# Test: EnvironmentSnapshot model construction
# ---------------------------------------------------------------------------


def test_environment_snapshot_model() -> None:
    hardware = _make_hardware()
    from llenergymeasure import __version__

    snap = EnvironmentSnapshot(
        hardware=hardware,
        python_version="3.11.5",
        pip_freeze="numpy==1.26.0\ntorch==2.2.0\n",
        conda_list=None,
        tool_version=__version__,
        cuda_version="12.4",
        cuda_version_source="torch",
    )

    assert snap.python_version == "3.11.5"
    assert snap.tool_version == __version__
    assert snap.cuda_version == "12.4"
    assert snap.cuda_version_source == "torch"
    assert snap.conda_list is None
    assert "numpy" in snap.pip_freeze


def test_environment_snapshot_optional_fields() -> None:
    hardware = _make_hardware()
    from llenergymeasure import __version__

    snap = EnvironmentSnapshot(
        hardware=hardware,
        python_version="3.10.0",
        pip_freeze="",
        tool_version=__version__,
    )
    assert snap.cuda_version is None
    assert snap.cuda_version_source is None
    assert snap.conda_list is None


# ---------------------------------------------------------------------------
# Test: CUDA version detection — torch source
# ---------------------------------------------------------------------------


def test_cuda_version_from_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_torch = MagicMock()
    mock_torch.version.cuda = "12.1"
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str):
        if name == "torch":
            return MagicMock()  # truthy — torch is "installed"
        return original_find_spec(name)

    monkeypatch.setattr(env_module.importlib.util, "find_spec", patched_find_spec)

    version, source = detect_cuda_version_with_source()
    assert version == "12.1"
    assert source == "torch"


def test_cuda_version_from_torch_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls through to next source when torch.version.cuda is falsy."""
    mock_torch = MagicMock()
    mock_torch.version.cuda = None
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str):
        if name == "torch":
            return MagicMock()
        return original_find_spec(name)

    monkeypatch.setattr(env_module.importlib.util, "find_spec", patched_find_spec)

    # No version.txt, no nvcc → should fall through to (None, None)
    monkeypatch.setattr(
        env_module.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
    )

    # Also mock open to fail
    monkeypatch.setattr(
        "builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
    )

    version, source = detect_cuda_version_with_source()
    assert version is None
    assert source is None


# ---------------------------------------------------------------------------
# Test: CUDA version detection — version.txt source
# ---------------------------------------------------------------------------


def test_cuda_version_from_version_txt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No torch
    monkeypatch.setattr(env_module.importlib.util, "find_spec", lambda name: None)

    # Create a fake version.txt
    version_file = tmp_path / "version.txt"
    version_file.write_text("CUDA Version 12.4.0\n")

    # Patch open to serve our fake file for the expected paths
    real_open = open

    def fake_open(path, *args, **kwargs):
        if "version.txt" in str(path) or "version.json" in str(path):
            return real_open(str(version_file), *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    version, source = detect_cuda_version_with_source()
    assert version == "12.4"
    assert source == "version_txt"


# ---------------------------------------------------------------------------
# Test: CUDA version detection — nvcc source
# ---------------------------------------------------------------------------


def test_cuda_version_from_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    # No torch
    monkeypatch.setattr(env_module.importlib.util, "find_spec", lambda name: None)

    # version.txt open fails
    monkeypatch.setattr(
        "builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
    )

    # Mock subprocess.run to return nvcc output
    mock_result = MagicMock()
    mock_result.stdout = "nvcc: NVIDIA (R) Cuda compiler driver\nrelease 12.2, V12.2.140\n"

    monkeypatch.setattr(env_module.subprocess, "run", lambda *a, **kw: mock_result)

    version, source = detect_cuda_version_with_source()
    assert version == "12.2"
    assert source == "nvcc"


# ---------------------------------------------------------------------------
# Test: CUDA version detection — all sources fail → (None, None)
# ---------------------------------------------------------------------------


def test_cuda_version_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # No torch
    monkeypatch.setattr(env_module.importlib.util, "find_spec", lambda name: None)

    # No version.txt
    monkeypatch.setattr(
        "builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
    )

    # nvcc not available
    monkeypatch.setattr(
        env_module.subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
    )

    version, source = detect_cuda_version_with_source()
    assert version is None
    assert source is None


# ---------------------------------------------------------------------------
# Test: pip freeze capture
# ---------------------------------------------------------------------------


def test_pip_freeze_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_result = MagicMock()
    mock_result.stdout = "numpy==1.26.0\ntorch==2.2.0\n"
    monkeypatch.setattr(env_module.subprocess, "run", lambda *a, **kw: mock_result)

    result = _capture_pip_freeze()
    assert "numpy==1.26.0" in result
    assert "torch==2.2.0" in result


def test_pip_freeze_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns empty string when subprocess fails."""
    monkeypatch.setattr(
        env_module.subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("pip not found")),
    )
    result = _capture_pip_freeze()
    assert result == ""


# ---------------------------------------------------------------------------
# Test: conda list returns None when conda not installed
# ---------------------------------------------------------------------------


def test_conda_list_no_conda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module.shutil, "which", lambda name: None)
    result = _capture_conda_list()
    assert result is None


def test_conda_list_with_conda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/conda")

    mock_result = MagicMock()
    mock_result.stdout = "# packages in environment at /opt/conda:\nnumpy 1.26.0 py311\n"
    monkeypatch.setattr(env_module.subprocess, "run", lambda *a, **kw: mock_result)

    result = _capture_conda_list()
    assert result is not None
    assert "numpy" in result


def test_conda_list_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None if conda list subprocess raises."""
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/conda")
    monkeypatch.setattr(
        env_module.subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("conda failed")),
    )
    result = _capture_conda_list()
    assert result is None


# ---------------------------------------------------------------------------
# Test: collect_environment_snapshot integration (mocked hardware)
# ---------------------------------------------------------------------------


def test_collect_environment_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware = _make_hardware()

    # Mock core/environment collect function
    monkeypatch.setattr(
        "llenergymeasure.core.environment.collect_environment_metadata",
        lambda: hardware,
    )

    # Mock pip freeze
    mock_pip_result = MagicMock()
    mock_pip_result.stdout = "numpy==1.26.0\n"

    # Mock conda: not installed
    monkeypatch.setattr(env_module.shutil, "which", lambda name: None)

    # Mock subprocess.run (only pip freeze will be called)
    monkeypatch.setattr(env_module.subprocess, "run", lambda *a, **kw: mock_pip_result)

    # Mock CUDA detection to return a known result
    monkeypatch.setattr(
        env_module,
        "detect_cuda_version_with_source",
        lambda: ("12.4", "torch"),
    )

    # Import and call the function
    from llenergymeasure.domain.environment import collect_environment_snapshot

    snap = collect_environment_snapshot()

    assert snap.hardware is hardware
    from llenergymeasure import __version__

    assert snap.tool_version == __version__
    assert snap.cuda_version == "12.4"
    assert snap.cuda_version_source == "torch"
    assert snap.conda_list is None
    assert snap.python_version != ""  # Real platform.python_version()
    assert "numpy" in snap.pip_freeze
