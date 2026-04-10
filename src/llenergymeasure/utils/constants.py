"""Constants for LLenergyMeasure framework."""

from typing import Final

# Completion marker prefix for aggregated result files
COMPLETION_MARKER_PREFIX = ".completed_"

# Subprocess timeouts needed by layer-0 modules (config, domain).
# Kept here rather than in config/ssot.py because domain/ and config/ are
# sibling layers — domain cannot import config, so shared constants live in
# utils/ (the layer below both). config/ssot.py re-exports these for ergonomic
# access alongside the rest of the infrastructure constants.
TIMEOUT_NVCC: Final = 5
"""``nvcc --version`` subprocess timeout (seconds)."""
