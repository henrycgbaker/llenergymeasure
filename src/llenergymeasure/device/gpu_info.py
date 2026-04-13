"""GPU topology detection with MIG support.

This module provides utilities for detecting GPU configuration,
including NVIDIA Multi-Instance GPU (MIG) partitions.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig

from llenergymeasure.config.ssot import (
    ENGINE_PYTORCH,
    ENGINE_TENSORRT,
    ENGINE_VLLM,
    TIMEOUT_NVIDIA_SMI,
)

logger = logging.getLogger(__name__)


@contextmanager
def nvml_context() -> Generator[None, None, None]:
    """Context manager for NVML init/shutdown lifecycle.

    Best-effort: silently ignores ImportError (pynvml not installed) and
    NVMLError (no NVIDIA GPU). Callers receive None on failure - handle gracefully.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            yield
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        yield  # pynvml absent or nvmlInit failed — caller proceeds without NVML


@dataclass
class GPUInfo:
    """Information about a GPU or MIG instance.

    Attributes:
        index: CUDA device index (0, 1, 2, ...)
        name: GPU model name (e.g., "NVIDIA A100-SXM4-80GB")
        uuid: Unique GPU identifier
        memory_mb: Total memory in megabytes
        is_mig_capable: Whether the GPU supports MIG partitioning
        is_mig_enabled: Whether MIG mode is currently enabled
        is_mig_instance: Whether this is a MIG slice (not parent GPU)
        parent_gpu_index: If MIG instance, the parent GPU's index
        mig_profile: MIG profile name (e.g., "1g.10gb", "3g.40gb")
        compute_capability: GPU compute capability (e.g., "8.0")
    """

    index: int
    name: str
    uuid: str
    memory_mb: int
    is_mig_capable: bool = False
    is_mig_enabled: bool = False
    is_mig_instance: bool = False
    parent_gpu_index: int | None = None
    mig_profile: str | None = None
    compute_capability: str | None = None
    mig_instance_count: int = 0

    @property
    def memory_gb(self) -> float:
        """Memory in gigabytes."""
        return self.memory_mb / 1024

    @property
    def display_name(self) -> str:
        """Short display name for the GPU."""
        if self.is_mig_instance and self.mig_profile:
            return f"MIG {self.mig_profile}"
        return self.name

    @property
    def status_label(self) -> str:
        """Status label for display."""
        if self.is_mig_instance:
            return "MIG Instance"
        elif self.is_mig_enabled:
            count_str = f" ({self.mig_instance_count} instances)" if self.mig_instance_count else ""
            return f"MIG Enabled{count_str}"
        elif self.is_mig_capable:
            return "MIG Capable (not enabled)"
        else:
            return "Full GPU"


@dataclass
class GPUTopology:
    """Complete GPU topology of the system.

    Attributes:
        devices: List of all GPU devices (full GPUs and MIG instances)
        has_mig: Whether any MIG instances are present
        parent_gpus: Indices of parent GPUs (non-MIG or MIG-enabled parents)
        mig_instances: Indices of MIG instance devices
    """

    devices: list[GPUInfo] = field(default_factory=list)

    @property
    def has_mig(self) -> bool:
        """Whether any MIG instances are present."""
        return any(d.is_mig_instance for d in self.devices)

    @property
    def parent_gpus(self) -> list[int]:
        """Indices of parent GPUs."""
        return [d.index for d in self.devices if not d.is_mig_instance]

    @property
    def mig_instances(self) -> list[int]:
        """Indices of MIG instance devices."""
        return [d.index for d in self.devices if d.is_mig_instance]

    def get_device(self, index: int) -> GPUInfo | None:
        """Get device by index."""
        for d in self.devices:
            if d.index == index:
                return d
        return None


def detect_gpu_topology() -> GPUTopology:
    """Detect all visible CUDA devices and their MIG status.

    Uses pynvml (NVIDIA Management Library) to query GPU information.
    Falls back to PyTorch if pynvml is unavailable.

    Returns:
        GPUTopology with all detected devices.
    """
    devices: list[GPUInfo] = []

    # Try pynvml first (more detailed info, especially for MIG)
    try:
        devices = _detect_via_pynvml()
        if devices:
            return GPUTopology(devices=devices)
    except Exception as e:
        logger.debug("pynvml detection failed: %s", e)

    # Fall back to PyTorch
    try:
        devices = _detect_via_torch()
        if devices:
            return GPUTopology(devices=devices)
    except Exception as e:
        logger.debug("PyTorch detection failed: %s", e)

    # No GPUs detected
    logger.warning("No CUDA devices detected")
    return GPUTopology(devices=[])


def _detect_via_pynvml() -> list[GPUInfo]:
    """Detect GPUs using pynvml."""
    import pynvml

    devices: list[GPUInfo] = []

    # Get MIG instance counts from nvidia-smi (pynvml doesn't expose this easily)
    mig_counts = _get_mig_instance_counts()

    with nvml_context():
        device_count = pynvml.nvmlDeviceGetCount()
        logger.debug("pynvml detected %d device(s)", device_count)

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = _get_device_info_pynvml(handle, i, mig_counts.get(i, 0))
            devices.append(info)

    return devices


def _get_mig_instance_counts() -> dict[int, int]:
    """Get MIG instance counts per GPU by parsing nvidia-smi -L output."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_NVIDIA_SMI,
        )
        if result.returncode != 0:
            return {}

        counts: dict[int, int] = {}
        current_gpu = -1

        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("GPU "):
                # Parse "GPU 0: NVIDIA A100-PCIE-40GB ..."
                try:
                    gpu_idx = int(line.split(":")[0].replace("GPU ", ""))
                    current_gpu = gpu_idx
                    counts[current_gpu] = 0
                except (ValueError, IndexError):
                    pass
            elif "MIG" in line and current_gpu >= 0:
                # This is a MIG instance line
                counts[current_gpu] = counts.get(current_gpu, 0) + 1

        return counts
    except Exception as e:
        logger.debug("Failed to get MIG instance counts: %s", e)
        return {}


def _get_device_info_pynvml(handle: Any, index: int, mig_instance_count: int = 0) -> GPUInfo:
    """Extract GPU info from pynvml handle."""
    import pynvml

    # Basic info
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid, bytes):
        uuid = uuid.decode("utf-8")

    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_mb = memory_info.total // (1024 * 1024)

    # Check MIG capability and status
    is_mig_capable = False
    is_mig_enabled = False
    is_mig_instance = False
    parent_gpu_index = None
    mig_profile = None

    # Check if this is a MIG device by looking at the UUID format
    # MIG device UUIDs contain "MIG-" prefix
    if uuid.startswith("MIG-"):
        is_mig_instance = True
        # Extract MIG profile from name if available
        # MIG device names often include profile info
        mig_profile = _extract_mig_profile(name)

    # For non-MIG devices, check if MIG is supported/enabled
    if not is_mig_instance:
        try:
            current_mode, _pending_mode = pynvml.nvmlDeviceGetMigMode(handle)
            is_mig_capable = True
            is_mig_enabled = current_mode == pynvml.NVML_DEVICE_MIG_ENABLE
        except pynvml.NVMLError:
            # MIG not supported on this GPU
            pass

    # Get compute capability
    compute_capability = None
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = f"{major}.{minor}"
    except (pynvml.NVMLError, AttributeError):
        pass

    return GPUInfo(
        index=index,
        name=name,
        uuid=uuid,
        memory_mb=memory_mb,
        is_mig_capable=is_mig_capable,
        is_mig_enabled=is_mig_enabled,
        is_mig_instance=is_mig_instance,
        parent_gpu_index=parent_gpu_index,
        mig_profile=mig_profile,
        compute_capability=compute_capability,
        mig_instance_count=mig_instance_count,
    )


def _extract_mig_profile(name: str) -> str | None:
    """Extract MIG profile from device name.

    MIG profiles follow patterns like "1g.10gb", "2g.20gb", "3g.40gb", etc.
    """
    import re

    # Look for patterns like "1g.10gb", "3g.40gb", etc.
    match = re.search(r"(\d+g\.\d+gb)", name.lower())
    if match:
        return match.group(1)

    # Alternative: look for memory-based profile indication
    match = re.search(r"MIG\s+(\d+)GB", name, re.IGNORECASE)
    if match:
        return f"?g.{match.group(1)}gb"

    return None


def _detect_via_torch() -> list[GPUInfo]:
    """Detect GPUs using PyTorch (fallback)."""
    import torch

    if not torch.cuda.is_available():
        return []

    devices: list[GPUInfo] = []
    device_count = torch.cuda.device_count()
    logger.debug("PyTorch detected %d device(s)", device_count)

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)

        # Check for MIG by looking at device name
        name = props.name
        is_mig_instance = "MIG" in name.upper()
        mig_profile = _extract_mig_profile(name) if is_mig_instance else None

        devices.append(
            GPUInfo(
                index=i,
                name=name,
                uuid=f"torch-{i}",  # PyTorch doesn't expose UUID
                memory_mb=props.total_memory // (1024 * 1024),
                is_mig_capable=False,  # Can't detect via PyTorch
                is_mig_enabled=False,
                is_mig_instance=is_mig_instance,
                parent_gpu_index=None,
                mig_profile=mig_profile,
                compute_capability=f"{props.major}.{props.minor}",
            )
        )

    return devices


def format_gpu_topology(topology: GPUTopology) -> str:
    """Format GPU topology as a tree for display.

    Returns a human-readable string showing the GPU hierarchy.
    """
    if not topology.devices:
        return "No CUDA devices detected"

    lines = [f"GPU Topology ({len(topology.devices)} device(s) detected)"]

    # Group MIG instances by parent (if we can determine parent)
    # For now, just list all devices with appropriate formatting
    for i, device in enumerate(topology.devices):
        is_last = i == len(topology.devices) - 1
        prefix = "└──" if is_last else "├──"

        mem_str = f"{device.memory_gb:.0f}GB" if device.memory_gb >= 1 else f"{device.memory_mb}MB"

        if device.is_mig_instance:
            profile = device.mig_profile or "unknown profile"
            lines.append(f"{prefix} [{device.index}] MIG {profile} ({mem_str})")
        else:
            status = device.status_label
            lines.append(f"{prefix} [{device.index}] {device.name} ({mem_str}) - {status}")

    # Add notes about MIG usage if any GPUs have MIG enabled
    has_mig_enabled = any(d.is_mig_enabled and d.mig_instance_count > 0 for d in topology.devices)
    if has_mig_enabled:
        lines.append("")
        lines.append("MIG Usage Notes:")
        lines.append("  • MIG instances must be addressed by UUID, not parent GPU index")
        lines.append("  • Example: CUDA_VISIBLE_DEVICES=MIG-<uuid> lem ...")
        lines.append("  • Run: nvidia-smi -L  to see MIG instance UUIDs")
        lines.append("  • MIG instances are isolated - no distributed inference across them")
        lines.append("  • Energy readings reflect parent GPU total, not per-instance")

    return "\n".join(lines)


def validate_gpu_selection(
    gpus: list[int],
    topology: GPUTopology,
) -> list[str]:
    """Validate GPU selection and return any warnings.

    Args:
        gpus: List of GPU indices the user wants to use.
        topology: Detected GPU topology.

    Returns:
        List of warning messages (empty if no issues).
    """
    warnings: list[str] = []

    if not topology.devices:
        warnings.append("No CUDA devices detected on this system")
        return warnings

    max_index = max(d.index for d in topology.devices)

    for idx in gpus:
        device = topology.get_device(idx)
        if device is None:
            warnings.append(f"GPU index {idx} not found (available: 0-{max_index})")
        elif device.is_mig_instance and not any("MIG" in w for w in warnings):
            # Warn about MIG energy measurement (only once)
            warnings.append(
                "Selected GPU(s) include MIG instance(s). "
                "Energy measurements will reflect parent GPU total."
            )

    # Check for mixing MIG and non-MIG
    selected_devices: list[GPUInfo] = [
        d for d in (topology.get_device(i) for i in gpus) if d is not None
    ]

    has_mig = any(d.is_mig_instance for d in selected_devices)
    has_full = any(not d.is_mig_instance for d in selected_devices)

    if has_mig and has_full:
        warnings.append("Mixing MIG instances and full GPUs may produce inconsistent results")

    return warnings


def get_device_mig_info(cuda_device_index: int) -> dict[str, Any]:
    """Get MIG-specific info for a CUDA device (for result metadata).

    This uses PyTorch to query the device, which correctly respects
    CUDA_VISIBLE_DEVICES. Using pynvml directly would give wrong results
    when CUDA_VISIBLE_DEVICES remaps device indices.

    Args:
        cuda_device_index: CUDA device index (as seen by PyTorch).

    Returns:
        Dictionary with MIG metadata fields.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_is_mig": False,
                "gpu_mig_profile": None,
                "gpu_name": "unknown",
            }

        if cuda_device_index >= torch.cuda.device_count():
            return {
                "gpu_is_mig": False,
                "gpu_mig_profile": None,
                "gpu_name": "unknown",
            }

        # Get properties from PyTorch (respects CUDA_VISIBLE_DEVICES)
        props = torch.cuda.get_device_properties(cuda_device_index)
        name = props.name

        # Detect MIG from device name (MIG instances have "MIG" in name)
        is_mig = "MIG" in name.upper()
        mig_profile = _extract_mig_profile(name) if is_mig else None

        return {
            "gpu_is_mig": is_mig,
            "gpu_mig_profile": mig_profile,
            "gpu_name": name,
        }

    except Exception as e:
        logger.debug("Failed to get MIG info for device %d: %s", cuda_device_index, e)
        return {
            "gpu_is_mig": False,
            "gpu_mig_profile": None,
            "gpu_name": "unknown",
        }


def get_compute_capability(gpu_index: int = 0) -> tuple[int, int] | None:
    """Return (major, minor) SM version via pynvml, or None on failure.

    Args:
        gpu_index: NVML device index (default: 0).

    Returns:
        Tuple of (major, minor) SM version, e.g. (8, 0) for A100.
        Returns None if pynvml is unavailable or query fails.
    """
    try:
        import pynvml

        with nvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return (major, minor)
    except Exception:
        return None


def get_gpu_architecture(device_index: int = 0) -> str:
    """Get GPU compute capability string (e.g., "sm_80" for A100).

    This is the SSOT for GPU architecture detection. Use this instead of
    duplicating torch.cuda.get_device_properties() calls.

    Args:
        device_index: CUDA device index (default: 0).

    Returns:
        Architecture string like "sm_80" (A100) or "sm_89" (L40).
        Returns "unknown" if detection fails.
    """
    try:
        import torch

        if torch.cuda.is_available() and device_index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_index)
            return f"sm_{props.major}{props.minor}"
    except Exception as e:
        logger.debug("Failed to get GPU architecture for device %d: %s", device_index, e)

    return "unknown"


def _resolve_gpu_indices(config: ExperimentConfig) -> list[int]:
    """Determine GPU indices to monitor for an experiment.

    Rules per engine:
    - **vLLM**: tensor_parallel_size * pipeline_parallel_size GPUs. Both are known
      from config before the harness runs, so gpu_indices = list(range(total)).
    - **PyTorch**: device_map="auto" (or any non-None device_map) monitors all
      NVML-visible GPUs. Model sharding is determined at load time inside
      harness.run(), but gpu_indices must be passed *before* load. Using all
      visible GPUs is correct and safe.
    - **TensorRT-LLM**: tp_size GPUs. Known from config before harness runs.
    - **Otherwise**: [0] (single-GPU default, backward compatible).

    Note: num_processes > 1 (data parallelism via Accelerate) is not handled here.
    For local runs this path is not yet implemented; for Docker each subprocess calls
    the harness independently.
    """
    if config.engine == ENGINE_VLLM and config.vllm is not None:
        tp = 1
        pp = 1
        if config.vllm.engine is not None:
            tp = config.vllm.engine.tensor_parallel_size or 1
            pp = config.vllm.engine.pipeline_parallel_size or 1
        total = tp * pp
        if total > 1:
            return list(range(total))
    elif config.engine == ENGINE_TENSORRT and config.tensorrt is not None:
        tp = config.tensorrt.tp_size or 1
        if tp > 1:
            return list(range(tp))
    elif (
        config.engine == ENGINE_PYTORCH
        and config.pytorch is not None
        and config.pytorch.device_map is not None
    ):
        # Model will shard across all visible GPUs — measure all of them.
        # Best-effort: if pynvml is absent or no NVIDIA GPU, fall through to [0].
        try:
            import pynvml

            with nvml_context():
                count = pynvml.nvmlDeviceGetCount()
            if count > 1:
                return list(range(count))
        except Exception:
            pass  # pynvml absent or no NVIDIA GPU — fall through to [0]
    return [0]
