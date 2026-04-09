"""Shared fixtures and factories for harness/ tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.conftest import TEST_POWER_MW


def make_pynvml_mock(
    *,
    power_mw: int = TEST_POWER_MW,
    power_mw_values: list[int] | None = None,
) -> MagicMock:
    """Build a minimal pynvml mock for baseline/power tests.

    The thermal-throttle variant in test_power_thermal.py and the
    memory variant in test_gpu_memory.py remain local (different shape).

    Args:
        power_mw: Constant return value for nvmlDeviceGetPowerUsage.
        power_mw_values: Side-effect list (overrides power_mw).
    """
    mock = MagicMock()
    mock.NVMLError = Exception

    if power_mw_values is not None:
        mock.nvmlDeviceGetPowerUsage.side_effect = power_mw_values
    else:
        mock.nvmlDeviceGetPowerUsage.return_value = power_mw

    return mock
