"""Shared formatting utilities (Layer 0 — importable from any layer)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


def format_elapsed(seconds: float) -> str:
    """Format seconds as human-readable elapsed time.

    Examples:
        4.2  -> "4.2s"
        272  -> "4m 32s"
        3900 -> "1h 05m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes:02d}m"


_DETAIL_MAX_LEN = 34
"""Maximum detail string length before truncation (80-col terminal safe)."""


def truncate_detail(detail: str) -> str:
    """Truncate detail text to fit 80-column terminals."""
    if len(detail) > _DETAIL_MAX_LEN:
        return detail[: _DETAIL_MAX_LEN - 3] + "..."
    return detail


def sig3(value: float) -> str:
    """Format a float to 3 significant figures.

    Examples:
        312.4  -> "312"
        3.12   -> "3.12"
        0.00312 -> "0.00312"
        847.0  -> "847"
        0      -> "0"
        1234   -> "1230"
    """
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    # Number of decimal places needed for 3 sig figs
    decimal_places = max(0, 2 - magnitude)
    rounded = round(value, decimal_places - int(magnitude >= 3) * 0)
    # Recompute for clarity: round to 3 sig figs
    factor = 10 ** (magnitude - 2)
    rounded = round(value / factor) * factor
    # Format without trailing zeros
    if decimal_places <= 0:
        return str(int(rounded))
    formatted = f"{rounded:.{decimal_places}f}"
    # Strip trailing zeros after decimal point
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def model_short_name(full_model: str) -> str:
    """Strip provider prefix from model name (e.g. ``Qwen/Qwen2.5-0.5B`` -> ``Qwen2.5-0.5B``)."""
    return str(full_model).rsplit("/", 1)[-1]


def compute_mj_per_tok(
    energy_j: float, throughput_tok_s: float, duration_sec: float
) -> float | None:
    """Compute millijoules per token from energy, throughput, and duration.

    Returns None if any input is missing or zero.
    """
    if not energy_j or not throughput_tok_s or not duration_sec:
        return None
    total_tokens = throughput_tok_s * duration_sec
    if total_tokens <= 0:
        return None
    return (energy_j / total_tokens) * 1000


_HEADER_MAX_LEN = 70
"""Maximum experiment header length before truncation."""

# ExperimentConfig defaults for non-default param detection.
# Keep in sync with config/models.py ExperimentConfig field defaults.
_EXPERIMENT_DEFAULTS: dict[str, object] = {
    "precision": "bf16",
    "n": 100,
    "max_input_tokens": 512,
    "max_output_tokens": 256,
}

# Short display names for backend-specific params.
_BACKEND_PARAM_LABELS: dict[str, str] = {
    "batch_size": "batch",
    "attn_implementation": "attn",
    "torch_compile": "compile",
    "torch_compile_mode": "compile_mode",
    "gpu_memory_utilization": "gpu_mem",
    "max_num_seqs": "seqs",
    "enforce_eager": "eager",
    "enable_chunked_prefill": "chunked",
    "tensor_parallel_size": "tp",
    "max_batch_size": "max_batch",
    "max_seq_len": "max_seq",
    "quant_algo": "quant",
}


def format_experiment_header(config: ExperimentConfig) -> str:
    """Build a compact experiment header string for CLI display.

    Format: ``model_short / backend / key=val key=val ...``

    Shows the model name without provider prefix, the backend, and all
    parameters that differ from ExperimentConfig class defaults. Truncated
    to ~70 chars with ``...`` if too long.
    """
    model_short = model_short_name(config.model)

    # Collect non-default top-level params
    params: list[str] = []
    for field_name, default_val in _EXPERIMENT_DEFAULTS.items():
        actual = getattr(config, field_name, None)
        if actual is not None and actual != default_val:
            params.append(f"{field_name}={actual}")

    # Collect non-default backend-specific params
    backend_config = getattr(config, config.backend, None)
    if backend_config is not None:
        _collect_backend_params(backend_config, params, prefix="")

    header = f"{model_short} / {config.backend}"
    if params:
        header += f" / {' '.join(params)}"
    if len(header) > _HEADER_MAX_LEN:
        header = header[: _HEADER_MAX_LEN - 3] + "..."
    return header


def _collect_backend_params(
    obj: object, params: list[str], prefix: str, *, max_depth: int = 2
) -> None:
    """Recursively extract non-None, non-default backend params with short labels."""
    if max_depth <= 0:
        return
    # Walk Pydantic model fields
    model_fields: dict[str, object] = {}
    if hasattr(obj, "model_fields"):
        model_fields = obj.model_fields  # type: ignore[assignment]
    for field_name in model_fields:
        value = getattr(obj, field_name, None)
        if value is None:
            continue
        # Skip sub-config objects that are defaults (recurse into non-None ones)
        if hasattr(value, "model_fields"):
            _collect_backend_params(value, params, prefix=field_name + ".", max_depth=max_depth - 1)
            continue
        # Skip True/False for boolean fields where True is the "interesting" case
        # and skip values that match field defaults
        field_info = model_fields[field_name]
        field_default = getattr(field_info, "default", None)
        if value == field_default:
            continue
        short = _BACKEND_PARAM_LABELS.get(field_name, field_name)
        params.append(f"{prefix}{short}={value}")
