"""Environment metadata collection for experiment reproducibility.

Collects GPU, CUDA, driver, thermal, CPU, and container information via NVML.
Gracefully degrades when NVML is unavailable — returns EnvironmentMetadata
with reasonable defaults instead of crashing.
"""

import importlib.util
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from llenergymeasure.core.gpu_info import nvml_context
from llenergymeasure.domain.environment import (
    ContainerEnvironment,
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    GPUEnvironment,
    ThermalEnvironment,
)


def collect_environment_metadata(device_index: int = 0) -> EnvironmentMetadata:
    """Collect full environment metadata for an experiment.

    Queries NVML for GPU, CUDA, driver, and thermal information. Falls back
    to reasonable defaults when NVML or specific queries are unavailable.

    Args:
        device_index: CUDA device index to query.

    Returns:
        EnvironmentMetadata with all available hardware/software info.
    """
    if importlib.util.find_spec("pynvml") is None:
        logger.debug("Environment: pynvml not available, returning defaults")
        return _unavailable_metadata()

    import pynvml

    result: EnvironmentMetadata | None = None

    with nvml_context():
        logger.debug("Environment: NVML initialised")
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            logger.debug(f"Environment: failed to get device handle: {e}")
            return _unavailable_metadata()

        try:
            gpu = _collect_gpu(pynvml, handle)
            cuda = _collect_cuda(pynvml)
            thermal = _collect_thermal(pynvml, handle)
            cpu = _collect_cpu()
            container = _collect_container()

            result = EnvironmentMetadata(
                gpu=gpu,
                cuda=cuda,
                thermal=thermal,
                cpu=cpu,
                container=container,
                collected_at=datetime.now(),
            )
        except Exception as e:
            logger.debug(f"Environment: collection failed: {e}")

    if result is not None:
        return result
    return _unavailable_metadata()


def _collect_gpu(pynvml: Any, handle: Any) -> GPUEnvironment:
    """Collect GPU hardware information."""
    name = "unknown"
    vram_total_mb = 0.0
    compute_capability = None

    try:
        raw_name = pynvml.nvmlDeviceGetName(handle)
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        logger.debug(f"Environment: GPU name = {name}")
    except pynvml.NVMLError as e:
        logger.debug(f"Environment: failed to get GPU name: {e}")

    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total_mb = mem_info.total / (1024 * 1024)
        logger.debug(f"Environment: VRAM = {vram_total_mb:.0f} MB")
    except pynvml.NVMLError as e:
        logger.debug(f"Environment: failed to get memory info: {e}")

    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = f"{major}.{minor}"
        logger.debug(f"Environment: compute capability = {compute_capability}")
    except (pynvml.NVMLError, AttributeError) as e:
        logger.debug(f"Environment: failed to get compute capability: {e}")

    return GPUEnvironment(
        name=name,
        vram_total_mb=vram_total_mb,
        compute_capability=compute_capability,
    )


def _collect_cuda(pynvml: Any) -> CUDAEnvironment:
    """Collect CUDA and driver version information."""
    cuda_version = "unknown"
    driver_version = "unknown"

    try:
        raw_driver = pynvml.nvmlSystemGetDriverVersion()
        driver_version = (
            raw_driver.decode("utf-8") if isinstance(raw_driver, bytes) else str(raw_driver)
        )
        logger.debug(f"Environment: driver version = {driver_version}")
    except pynvml.NVMLError as e:
        logger.debug(f"Environment: failed to get driver version: {e}")

    try:
        cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion()
        major = cuda_driver_version // 1000
        minor = (cuda_driver_version % 1000) // 10
        cuda_version = f"{major}.{minor}"
        logger.debug(f"Environment: CUDA version = {cuda_version}")
    except (pynvml.NVMLError, AttributeError) as e:
        logger.debug(f"Environment: failed to get CUDA version: {e}")

    return CUDAEnvironment(
        version=cuda_version,
        driver_version=driver_version,
    )


def _collect_thermal(pynvml: Any, handle: Any) -> ThermalEnvironment:
    """Collect GPU thermal state."""
    temperature_c: float | None = None
    power_limit_w: float | None = None
    default_power_limit_w: float | None = None

    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        temperature_c = float(temp)
        logger.debug(f"Environment: temperature = {temperature_c:.0f}C")
    except pynvml.NVMLError as e:
        logger.debug(f"Environment: failed to get temperature: {e}")

    try:
        limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        power_limit_w = limit_mw / 1000.0
        logger.debug(f"Environment: power limit = {power_limit_w:.0f}W")
    except pynvml.NVMLError as e:
        logger.debug(f"Environment: failed to get power limit: {e}")

    try:
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        default_power_limit_w = default_mw / 1000.0
        logger.debug(f"Environment: default power limit = {default_power_limit_w:.0f}W")
    except pynvml.NVMLError as e:
        logger.debug(f"Environment: failed to get default power limit: {e}")

    return ThermalEnvironment(
        temperature_c=temperature_c,
        power_limit_w=power_limit_w,
        default_power_limit_w=default_power_limit_w,
    )


def _collect_cpu() -> CPUEnvironment:
    """Collect CPU and OS information."""
    governor = "unknown"

    # Read CPU governor (Linux only)
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        governor = governor_path.read_text().strip()
        logger.debug(f"Environment: CPU governor = {governor}")
    except (FileNotFoundError, PermissionError, OSError):
        logger.debug("Environment: CPU governor not available")

    cpu_model = platform.processor() or None
    logger.debug(f"Environment: CPU model = {cpu_model}")

    return CPUEnvironment(
        governor=governor,
        model=cpu_model,
        platform=platform.system(),
    )


def _collect_container() -> ContainerEnvironment:
    """Detect container runtime."""
    detected = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    runtime: str | None = None

    if detected:
        runtime = _detect_container_runtime()
        logger.debug(f"Environment: container detected, runtime = {runtime}")
    else:
        # Also check cgroup for container detection
        try:
            cgroup_content = Path("/proc/1/cgroup").read_text()
            if "docker" in cgroup_content or "containerd" in cgroup_content:
                detected = True
                runtime = "docker"
            elif "lxc" in cgroup_content:
                detected = True
                runtime = "lxc"
            logger.debug(f"Environment: cgroup container check = {detected}")
        except (FileNotFoundError, PermissionError, OSError):
            pass

    return ContainerEnvironment(
        detected=detected,
        runtime=runtime,
    )


def _detect_container_runtime() -> str | None:
    """Detect which container runtime is in use."""
    if os.path.exists("/.dockerenv"):
        return "docker"
    if os.path.exists("/run/.containerenv"):
        return "podman"
    return None


def _unavailable_metadata() -> EnvironmentMetadata:
    """Create metadata with reasonable defaults when NVML unavailable."""
    return EnvironmentMetadata(
        gpu=GPUEnvironment(name="unavailable", vram_total_mb=0.0),
        cuda=CUDAEnvironment(version="unknown", driver_version="unknown"),
        thermal=ThermalEnvironment(),
        cpu=_collect_cpu(),
        container=_collect_container(),
        collected_at=datetime.now(),
    )
