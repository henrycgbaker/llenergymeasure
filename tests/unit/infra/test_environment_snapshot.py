"""GPU-free unit tests for EnvironmentSnapshot and related detection functions.

All tests run without a GPU. torch, subprocess, and huggingface_hub are never
directly imported — access is always via monkeypatch.
"""

from __future__ import annotations

import importlib
import sys
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import llenergymeasure.domain.environment as env_module
from llenergymeasure.domain.environment import (
    EnvironmentSnapshot,
    _collect_installed_packages,
    collect_environment_snapshot_async,
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
        installed_packages=["numpy==1.26.0", "torch==2.2.0"],
        tool_version=__version__,
        cuda_version="12.4",
        cuda_version_source="torch",
    )

    assert snap.python_version == "3.11.5"
    assert snap.tool_version == __version__
    assert snap.cuda_version == "12.4"
    assert snap.cuda_version_source == "torch"
    assert "numpy==1.26.0" in snap.installed_packages


def test_environment_snapshot_optional_fields() -> None:
    hardware = _make_hardware()
    from llenergymeasure import __version__

    snap = EnvironmentSnapshot(
        hardware=hardware,
        python_version="3.10.0",
        installed_packages=[],
        tool_version=__version__,
    )
    assert snap.cuda_version is None
    assert snap.cuda_version_source is None


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

    # Narrow patch: only affects open() calls within the environment module
    monkeypatch.setattr(
        "llenergymeasure.domain.environment.open",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
        raising=False,
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

    # Narrow patch: only affects open() calls within the environment module
    real_open = open

    def fake_open(path, *args, **kwargs):
        if "version.txt" in str(path) or "version.json" in str(path):
            return real_open(str(version_file), *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("llenergymeasure.domain.environment.open", fake_open, raising=False)

    version, source = detect_cuda_version_with_source()
    assert version == "12.4"
    assert source == "version_txt"


# ---------------------------------------------------------------------------
# Test: CUDA version detection — nvcc source
# ---------------------------------------------------------------------------


def test_cuda_version_from_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    # No torch
    monkeypatch.setattr(env_module.importlib.util, "find_spec", lambda name: None)

    # Narrow patch: version.txt open fails within environment module only
    monkeypatch.setattr(
        "llenergymeasure.domain.environment.open",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
        raising=False,
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

    # Narrow patch: no version.txt within environment module only
    monkeypatch.setattr(
        "llenergymeasure.domain.environment.open",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
        raising=False,
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
# Test: installed packages enumeration
# ---------------------------------------------------------------------------


def test_collect_installed_packages_returns_sorted_list() -> None:
    """_collect_installed_packages returns a sorted list of name==version strings."""
    result = _collect_installed_packages()
    assert isinstance(result, list)
    assert len(result) > 0
    # All entries should be name==version format
    for entry in result:
        assert "==" in entry, f"Expected name==version format, got {entry!r}"
    # Should be sorted
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# Test: async snapshot collection returns a Future
# ---------------------------------------------------------------------------


def test_collect_environment_snapshot_async_returns_future(monkeypatch: pytest.MonkeyPatch) -> None:
    """collect_environment_snapshot_async returns a Future that resolves to a snapshot."""
    hardware = _make_hardware()

    monkeypatch.setattr(
        "llenergymeasure.infra.environment.collect_environment_metadata",
        lambda: hardware,
    )
    monkeypatch.setattr(
        env_module,
        "detect_cuda_version_with_source",
        lambda: ("12.4", "torch"),
    )

    future = collect_environment_snapshot_async()
    assert isinstance(future, Future)

    snap = future.result(timeout=10)
    assert isinstance(snap, EnvironmentSnapshot)
    assert snap.hardware is hardware


# ---------------------------------------------------------------------------
# Test: collect_environment_snapshot integration (mocked hardware)
# ---------------------------------------------------------------------------


def test_collect_environment_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware = _make_hardware()

    # Mock core/environment collect function
    monkeypatch.setattr(
        "llenergymeasure.infra.environment.collect_environment_metadata",
        lambda: hardware,
    )

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
    assert snap.python_version != ""  # Real platform.python_version()
    assert isinstance(snap.installed_packages, list)
    assert len(snap.installed_packages) > 0  # At least llenergymeasure itself
