"""Environment metadata models for experiment reproducibility.

Captures the hardware and software environment at experiment time,
enabling post-hoc analysis of environmental factors affecting measurements.
"""

import importlib.util
import logging
import subprocess
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GPUEnvironment(BaseModel):
    """GPU hardware information."""

    name: str = Field(..., description="GPU model name (e.g., 'NVIDIA A100-SXM4-80GB')")
    vram_total_mb: float = Field(..., description="Total VRAM in MB")
    compute_capability: str | None = Field(
        default=None,
        description="CUDA compute capability (e.g., '8.0')",
    )
    pcie_gen: int | None = Field(
        default=None,
        description="PCIe generation",
    )
    mig_enabled: bool = Field(
        default=False,
        description="Whether MIG (Multi-Instance GPU) is enabled",
    )


class CUDAEnvironment(BaseModel):
    """CUDA runtime information."""

    version: str = Field(..., description="CUDA version (e.g., '12.4')")
    driver_version: str = Field(..., description="NVIDIA driver version string")
    cudnn_version: str | None = Field(
        default=None,
        description="cuDNN version string",
    )


class ThermalEnvironment(BaseModel):
    """GPU thermal state at experiment start."""

    temperature_c: float | None = Field(
        default=None,
        description="GPU temperature at experiment start in Celsius",
    )
    power_limit_w: float | None = Field(
        default=None,
        description="Configured GPU power limit in Watts",
    )
    default_power_limit_w: float | None = Field(
        default=None,
        description="Factory default GPU power limit in Watts",
    )
    fan_speed_pct: float | None = Field(
        default=None,
        description="Fan speed as percentage (0-100)",
    )


class CPUEnvironment(BaseModel):
    """CPU and OS information."""

    governor: str = Field(
        default="unknown",
        description="CPU frequency governor (e.g., 'performance', 'powersave')",
    )
    model: str | None = Field(
        default=None,
        description="CPU model string",
    )
    platform: str = Field(..., description="OS platform (e.g., 'Linux')")


class ContainerEnvironment(BaseModel):
    """Container runtime detection."""

    detected: bool = Field(
        default=False,
        description="Whether running inside a container",
    )
    runtime: str | None = Field(
        default=None,
        description="Container runtime (e.g., 'docker', 'podman')",
    )


class EnvironmentMetadata(BaseModel):
    """Complete environment metadata for an experiment.

    Captures GPU, CUDA, thermal, CPU, and container information
    at experiment time for reproducibility and post-hoc analysis.
    """

    gpu: GPUEnvironment = Field(..., description="GPU hardware information")
    cuda: CUDAEnvironment = Field(..., description="CUDA runtime information")
    thermal: ThermalEnvironment = Field(
        default_factory=ThermalEnvironment,
        description="GPU thermal state at experiment start",
    )
    cpu: CPUEnvironment = Field(..., description="CPU and OS information")
    container: ContainerEnvironment = Field(
        default_factory=ContainerEnvironment,
        description="Container runtime detection",
    )
    collected_at: datetime = Field(..., description="When metadata was collected")

    @property
    def summary_line(self) -> str:
        """One-line environment summary for logging.

        Example: "A100 80GB | CUDA 12.4 | Driver 535.104 | 42C | container"
        """
        # Extract short GPU name (last part before memory)
        gpu_short = self.gpu.name.replace("NVIDIA ", "")
        vram_gb = int(self.gpu.vram_total_mb / 1024)

        parts = [
            f"{gpu_short} {vram_gb}GB",
            f"CUDA {self.cuda.version}",
            f"Driver {self.cuda.driver_version}",
        ]

        if self.thermal.temperature_c is not None:
            parts.append(f"{self.thermal.temperature_c:.0f}C")

        if self.container.detected:
            parts.append("container")

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# EnvironmentSnapshot — full software + hardware context (CM-32)
# ---------------------------------------------------------------------------


class EnvironmentSnapshot(BaseModel):
    """Full software+hardware environment snapshot for experiment reproducibility."""

    hardware: EnvironmentMetadata
    python_version: str
    installed_packages: list[str]
    tool_version: str
    cuda_version: str | None = None
    cuda_version_source: str | None = None  # "torch" | "version_txt" | "nvcc" | None


# ---------------------------------------------------------------------------
# CUDA version detection (CM-33) — multi-source fallback chain
# ---------------------------------------------------------------------------


def detect_cuda_version_with_source() -> tuple[str | None, str | None]:
    """Detect the CUDA version using a fallback chain.

    Returns:
        Tuple of (version_string, source_name) where source_name is one of:
        "torch", "version_txt", "nvcc", or None if detection failed.
    """
    # Source 1: torch.version.cuda
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            cuda_ver = torch.version.cuda
            if cuda_ver:
                return cuda_ver, "torch"
        except Exception:
            logger.debug("CUDA version: torch source failed", exc_info=True)

    # Source 2: /usr/local/cuda/version.txt or version.json
    import re

    for version_file in (
        "/usr/local/cuda/version.txt",
        "/usr/local/cuda/version.json",
    ):
        try:
            with open(version_file) as f:
                content = f.read()
            match = re.search(r"(\d+\.\d+)", content)
            if match:
                return match.group(1), "version_txt"
        except Exception:
            pass

    # Source 3: nvcc --version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if match:
            return match.group(1), "nvcc"
    except Exception:
        logger.debug("CUDA version: nvcc source failed", exc_info=True)

    # Source 4: Give up
    return None, None
