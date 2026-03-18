"""Constants for LLenergyMeasure framework."""

import os
from pathlib import Path

# Results directories
# Precedence: CLI --results-dir > LLM_ENERGY_RESULTS_DIR env var > "results"
DEFAULT_RESULTS_DIR = Path(os.environ.get("LLM_ENERGY_RESULTS_DIR", "results"))
RAW_RESULTS_SUBDIR = "raw"
AGGREGATED_RESULTS_SUBDIR = "aggregated"

# Experiment defaults
DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLING_INTERVAL_SEC = 1.0
DEFAULT_ACCELERATE_PORT = 29500

# Inference defaults
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0

# Streaming latency measurement
DEFAULT_STREAMING_WARMUP_REQUESTS = 5

# Schema version for result files
SCHEMA_VERSION = "2.0.0"

# State management
# Precedence: LLM_ENERGY_STATE_DIR env var > ".state"
DEFAULT_STATE_DIR = Path(os.environ.get("LLM_ENERGY_STATE_DIR", ".state"))
COMPLETION_MARKER_PREFIX = ".completed_"

# Timeouts
GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 2
DEFAULT_BARRIER_TIMEOUT_SEC = 600  # 10 minutes for distributed sync
DEFAULT_FLOPS_TIMEOUT_SEC = 30
DEFAULT_GPU_INFO_TIMEOUT_SEC = 10
DEFAULT_SIGKILL_WAIT_SEC = 2
