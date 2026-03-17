# tests/ - Test Suite

Unit and integration tests for v2.0.

## Structure

- **Unit tests**: `unit/` — fast, GPU-free tests using protocol injection fakes
- **Integration tests**: `integration/` — tests marked `@pytest.mark.gpu`, require GPU hardware
- **Fixtures**: `fixtures/` — shared test data; `fixtures/replay/` holds Tier 2 ExperimentResult JSON

## Running Tests

```bash
# GPU-free unit tests (default, runs in CI)
pytest tests/unit/

# All unit tests with xdist parallel execution
pytest tests/unit/ -m "not gpu and not docker" -n auto

# Domain-specific subset
pytest tests/unit/core/
pytest tests/unit/backends/

# Integration tests (GPU required)
pytest tests/integration/ -m gpu
```

## Markers

| Marker   | Meaning                                            |
| -------- | -------------------------------------------------- |
| `gpu`    | Requires GPU hardware (excluded from Tier 1)       |
| `docker` | Requires Docker with NVIDIA runtime                |
| `slow`   | Slow test (>30s); excluded from Tier 1 fast runs   |

## Shared Test Infrastructure

```python
# tests/conftest.py — factories available to all tests
from tests.conftest import make_config, make_result

# tests/fakes.py — protocol injection fakes (no MagicMock)
from tests.fakes import FakeInferenceBackend, FakeEnergyBackend, FakeResultsRepository
```

## Unit Test Organisation

Tests are organised in domain subdirectories that mirror `src/llenergymeasure/`:

### Domain subdirectories

| Directory              | Covers                                                    |
| ---------------------- | --------------------------------------------------------- |
| `unit/core/`           | Energy backends, warmup, FLOPs, measurement, GPU memory, env snapshot |
| `unit/backends/`       | Backend protocol, factory, detection; vLLM backend        |
| `unit/cli/`            | `llem run`, `llem config`, CLI display utilities          |
| `unit/study/`          | Study runner, manifest, grid, gaps, result, preflight     |
| `unit/docker/`         | Docker runner, errors, preflight, entrypoint, image registry |
| `unit/config/`         | Config schema, loader, introspection, user config         |

### Cross-cutting files (stay at `unit/`)

| File                        | Covers                                |
| --------------------------- | ------------------------------------- |
| `test_api.py`               | Public API surface (`run_experiment`, `run_study`) |
| `test_protocols.py`         | Protocol definitions                  |
| `test_exceptions.py`        | Exception hierarchy                   |
| `test_state_machine.py`     | State machine                         |
| `test_runner_resolution.py` | Runner resolution                     |
| `test_aggregation_v2.py`    | Results aggregation                   |
| `test_experiment_result_v2.py` | ExperimentResult domain model      |
| `test_persistence_v2.py`    | Results persistence                   |
| `test_preflight.py`         | General preflight checks              |

## Key Patterns

- **Protocol injection**: Inject `FakeInferenceBackend` via constructor args — no `unittest.mock.patch` on internals
- **No GPU in unit tests**: All unit tests run without CUDA hardware
- **GPU mark**: Integration tests use `@pytest.mark.gpu` to declare GPU requirement
- **Docker mark**: Tests requiring Docker runtime use `@pytest.mark.docker`
- **Factories**: Use `make_config()` and `make_result()` from `conftest.py` — override only what you care about

## See Also

- `tests/fakes.py` — protocol fakes
- `tests/conftest.py` — shared fixtures and factories
- `tests/fixtures/replay/` — Tier 2 ExperimentResult JSON fixtures (scaffold)
- `pyproject.toml` — pytest configuration and marker registration
