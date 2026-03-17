"""GPU index resolution utility shared by API and infra layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


def _resolve_gpu_indices(config: ExperimentConfig) -> list[int]:
    """Determine GPU indices to monitor for an experiment.

    Rules per backend:
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
    if config.backend == "vllm" and config.vllm is not None:
        tp = 1
        pp = 1
        if config.vllm.engine is not None:
            tp = config.vllm.engine.tensor_parallel_size or 1
            pp = config.vllm.engine.pipeline_parallel_size or 1
        total = tp * pp
        if total > 1:
            return list(range(total))
    elif config.backend == "tensorrt" and config.tensorrt is not None:
        tp = config.tensorrt.tp_size or 1
        if tp > 1:
            return list(range(tp))
    elif (
        config.backend == "pytorch"
        and config.pytorch is not None
        and config.pytorch.device_map is not None
    ):
        # Model will shard across all visible GPUs — measure all of them.
        # Best-effort: if pynvml is absent or no NVIDIA GPU, fall through to [0].
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
            finally:
                pynvml.nvmlShutdown()
            if count > 1:
                return list(range(count))
        except Exception:
            pass  # pynvml absent or no NVIDIA GPU — fall through to [0]
    return [0]
