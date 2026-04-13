# tests/ - Test Suite

Unit, integration, e2e, and runtime tests for the LLenergyMeasure framework.

## Structure

```
tests/
├── conftest.py          # Shared pytest fixtures
├── fixtures/            # Test data and fixtures
├── unit/                # Unit tests (fast, isolated)
├── integration/         # Integration tests (component interaction)
├── e2e/                 # End-to-end tests (full workflows, simulated)
└── runtime/             # Runtime tests (GPU-required, actual inference)
```

## Running Tests

```bash
# Run all tests (excluding runtime)
make test-all
# or
poetry run pytest tests/ -v --ignore=tests/runtime/

# Unit tests only (fast)
make test
# or
poetry run pytest tests/unit/ -v

# Integration tests
make test-integration
# or
poetry run pytest tests/integration/ -v

# E2E tests (simulated, no GPU required)
poetry run pytest tests/e2e/ -v

# Runtime tests (requires GPU)
make test-runtime
# or
poetry run pytest tests/runtime/ -v

# Runtime tests with engine filter
pytest tests/runtime/ -v --engine transformers
pytest tests/runtime/ -v --engine vllm
pytest tests/runtime/ -v --quick    # Quick subset

# Specific test file
poetry run pytest tests/unit/test_config_models.py -v

# With coverage
poetry run pytest tests/ --cov=llenergymeasure --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual components.

| File | Tests |
|------|-------|
| `test_config_models.py` | Pydantic config validation |
| `test_config_loader.py` | Config loading, inheritance |
| `test_core_inference.py` | Inference metrics calculation |
| `test_core_prompts.py` | Batch creation, tokenization |
| `test_core_distributed.py` | Distributed utilities |
| `test_core_model_loader.py` | Model loading logic |
| `test_core_energy_backends.py` | Energy sampler interface |
| `test_core_compute_metrics.py` | Memory/utilization stats |
| `test_domain_metrics.py` | Metric model validation |
| `test_domain_experiment.py` | Result model validation |
| `test_results_aggregation.py` | Aggregation logic |
| `test_repository.py` | FileSystemRepository |
| `test_cli.py` | CLI command parsing |

### Integration Tests (`tests/integration/`)

Tests for component interaction.

| File | Tests |
|------|-------|
| `test_config_aggregation_pipeline.py` | Config -> Aggregation flow |
| `test_cli_workflows.py` | CLI multi-step workflows |
| `test_repository_operations.py` | Repository CRUD operations |
| `test_error_handling.py` | Error propagation |
| `test_config_params_wired.py` | Config parameter wiring |

### E2E Tests (`tests/e2e/`)

Full workflow tests with simulated results (no GPU required).

| File | Tests |
|------|-------|
| `test_cli_e2e.py` | Full CLI workflows |

### Runtime Tests (`tests/runtime/`)

GPU-required tests that run actual inference to validate parameters.

| File | Purpose |
|------|---------|
| `conftest.py` | GPU fixtures, engine detection, skip markers |
| `test_all_params.py` | CANONICAL param testing (standalone + importable) |
| `test_runtime_params.py` | Pytest parametrised wrapper |
| `discover_params.py` | Param discovery utility |

**Features:**
- **Single source of truth**: Params auto-discovered from Pydantic models
- **Strict validation**: Tests fail if inference doesn't actually run
- **Engine filtering**: `--engine transformers` to test one engine
- **Quick mode**: `--quick` for fewer parameter variations
- **Discovery mode**: `--discover` to use auto-discovered params
- **Result verification**: Checks output tokens, throughput, energy

**Skip Markers:**
- `@pytest.mark.requires_gpu` - CUDA GPU available
- `@pytest.mark.requires_vllm` - vLLM installed
- `@pytest.mark.requires_tensorrt` - TensorRT-LLM installed
- `@pytest.mark.slow` - Test takes >1 minute

## Writing Tests

### Fixtures

Common fixtures in `conftest.py`:
```python
@pytest.fixture
def sample_config():
    return ExperimentConfig(
        config_name="test",
        model_name="test/model",
    )

@pytest.fixture
def temp_results_dir(tmp_path):
    return FileSystemRepository(tmp_path)
```

### Mocking GPU Operations

For tests that would require GPU:
```python
@pytest.fixture
def mock_cuda():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=4):
            yield
```

### Test Naming

- `test_<function>_<scenario>` for functions
- `test_<class>_<method>_<scenario>` for methods
- Use descriptive names: `test_load_config_with_inheritance_resolves_extends`

## Generating Documentation

After running runtime tests, generate the parameter support matrix:

```bash
# Run tests and generate results (from project root)
python -m tests.runtime.test_all_params --engine transformers --output results/test_results_pytorch.json

# Or with auto-discovery from Pydantic models
python -m tests.runtime.test_all_params --discover --engine transformers --output results/test_results_pytorch.json

# List discovered params without running tests
python -m tests.runtime.test_all_params --list-params --engine transformers

# Generate documentation from results
python scripts/generate_param_matrix.py
```

This generates `docs/generated/parameter-support-matrix.md`.

## CI Integration

Tests run via GitHub Actions:
```yaml
# .github/workflows/ci.yml
- run: make test
```

Coverage reports uploaded to Codecov.

## Related

- See `src/llenergymeasure/README.md` for package structure
- See `Makefile` for test commands
- See `tests/runtime/test_all_params.py` for parameter testing (canonical location)
- See `scripts/generate_param_matrix.py` for docs generation
