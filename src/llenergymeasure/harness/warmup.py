"""Thermal stabilisation utilities for MeasurementHarness.

Provides thermal_floor_wait() for use by MeasurementHarness after warmup.

Note: warmup_until_converged() and create_warmup_inference_fn() live in
backends/_helpers.py (the canonical location imported by all backends).
"""

from __future__ import annotations

import logging
import time

from llenergymeasure.config.models import ExperimentConfig

logger = logging.getLogger(__name__)


def thermal_floor_wait(config: ExperimentConfig) -> float:
    """Sleep for thermal_floor_seconds after warmup. Returns actual wait time in seconds.

    Returns 0.0 immediately if warmup is disabled or thermal_floor_seconds <= 0.
    Called by MeasurementHarness after backend.warmup(), before energy tracking starts.

    Args:
        config: Experiment configuration (reads warmup.enabled and warmup.thermal_floor_seconds).

    Returns:
        Actual elapsed wait time in seconds (>= 0.0).
    """
    if not config.warmup.enabled or config.warmup.thermal_floor_seconds <= 0:
        return 0.0
    logger.debug(
        "Thermal stabilisation: waiting %.0fs...",
        config.warmup.thermal_floor_seconds,
    )
    t0 = time.monotonic()
    time.sleep(config.warmup.thermal_floor_seconds)
    return time.monotonic() - t0
