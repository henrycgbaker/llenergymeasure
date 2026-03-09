"""Pre-dispatch GPU memory residual check using NVML.

Called by StudyRunner before spawning each experiment subprocess to detect
driver-level GPU memory leaks or residual allocations from prior processes.
Warnings are informational only — the check never blocks execution.
"""

from __future__ import annotations

import logging

from llenergymeasure.core.gpu_info import nvml_context

logger = logging.getLogger(__name__)


def check_gpu_memory_residual(
    device_index: int = 0,
    threshold_mb: float = 1024.0,
) -> None:
    """Query NVML for residual GPU memory before an experiment dispatch.

    If the memory used on ``device_index`` exceeds ``threshold_mb``, a warning
    is logged. The function never raises — any failure (missing pynvml, NVML
    initialisation error, device query error) is caught and logged at DEBUG
    level only.

    Args:
        device_index: GPU device index to query (default 0, single-GPU assumption).
        threshold_mb: Warning threshold in megabytes (default 1024 MB).
            NVML driver overhead varies by GPU: ~600 MB on A100, lower on
            consumer cards. 1 GB accommodates driver baselines while catching
            real residual allocations (loaded models, KV caches).
    """
    try:
        import pynvml
    except ImportError:
        logger.debug("pynvml not available, skipping GPU memory check")
        return

    used_mb: float | None = None
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem_info.used / (1024 * 1024)
        except Exception as e:
            logger.debug("GPU memory check failed: %s", e)
            return

    if used_mb is None:
        return

    if used_mb > threshold_mb:
        logger.warning(
            "Residual GPU memory detected on device %d: %.0f MB used "
            "(threshold: %.0f MB). Prior process may not have released GPU "
            "resources. Measurement accuracy could be affected.",
            device_index,
            used_mb,
            threshold_mb,
        )
