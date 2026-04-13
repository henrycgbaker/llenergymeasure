# utils/ - Cross-cutting Utilities

Shared exceptions, constants, and security utilities. Layer 0 in the six-layer architecture.

## Purpose

Provides the foundation all other layers import: exception hierarchy, framework-wide constants, and filesystem security helpers.

## Modules

| Module | Description |
|--------|-------------|
| `exceptions.py` | Exception hierarchy rooted at `LLEMError` |
| `constants.py` | Framework-wide constants (timeouts, defaults, schema version) |
| `security.py` | Path safety and experiment ID sanitisation |
| `__init__.py` | Package marker |

## exceptions.py

```python
from llenergymeasure.utils.exceptions import (
    LLEMError,           # base
    ConfigError,         # invalid or missing config
    EngineError,        # inference engine failures
    PreFlightError,      # pre-flight check failures
    ExperimentError,     # experiment execution errors
    StudyError,          # study orchestration errors
    DockerError,         # Docker container dispatch
    DockerPreFlightError,  # Docker pre-flight check (inherits PreFlightError)
    InvalidStateTransitionError,  # invalid state machine transition
)
```

`DockerError` carries structured fields: `fix_suggestion` and `stderr_snippet` for actionable error messages.

## constants.py

```python
from llenergymeasure.utils.constants import (
    DEFAULT_RESULTS_DIR,           # Path("results") or LLM_ENERGY_RESULTS_DIR env var
    RAW_RESULTS_SUBDIR,            # "raw"
    AGGREGATED_RESULTS_SUBDIR,     # "aggregated"
    DEFAULT_WARMUP_RUNS,           # 3
    DEFAULT_SAMPLING_INTERVAL_SEC, # 1.0
    DEFAULT_MAX_NEW_TOKENS,        # 256
    DEFAULT_TEMPERATURE,           # 1.0
    SCHEMA_VERSION,                # "2.0.0"
    DEFAULT_STATE_DIR,             # Path(".state") or LLM_ENERGY_STATE_DIR env var
    GRACEFUL_SHUTDOWN_TIMEOUT_SEC, # 2
    DEFAULT_FLOPS_TIMEOUT_SEC,     # 30
)
```

`DEFAULT_RESULTS_DIR` and `DEFAULT_STATE_DIR` respect environment variable overrides.

## security.py

```python
from llenergymeasure.utils.security import sanitize_experiment_id, is_safe_path, validate_path

# Sanitise experiment IDs for filesystem use (allows alphanumeric, _, -, .)
safe_id = sanitize_experiment_id("my experiment/2024")  # "my_experiment_2024"

# Prevent path traversal (uses Path.is_relative_to(), handles edge cases)
if is_safe_path(base_dir=Path("results"), target_path=user_path):
    ...

# Validate and resolve a path
resolved = validate_path(path, must_exist=True)
```

## Layer constraints

- Layer 0 — base layer; no imports from other llenergymeasure layers
- Can be imported by all layers above
- Do not add logic here that belongs in a higher layer (domain, config, etc.)
