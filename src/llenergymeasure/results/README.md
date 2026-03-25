# results/ - Results Persistence

Results storage, loading, and aggregation for experiment outputs.

## Purpose

Implements the late aggregation pattern where raw per-process results are persisted separately and aggregated on demand. This enables partial results recovery and flexible aggregation strategies.

## Key Files

### repository.py
File system repository for results.

**FileSystemRepository** - Persist and load results:
```python
from llenergymeasure.results import FileSystemRepository

repo = FileSystemRepository(base_path=Path("results"))

# Save raw per-process result
path = repo.save_raw(experiment_id, raw_result)

# Load raw results
results = repo.load_all_raw(experiment_id)

# Save aggregated result
path = repo.save_aggregated(aggregated_result)

# Load aggregated
result = repo.load_aggregated(experiment_id)

# Query operations
experiments = repo.list_experiments()  # With raw results
aggregated = repo.list_aggregated()    # With aggregated
has_raw = repo.has_raw(experiment_id)
has_agg = repo.has_aggregated(experiment_id)

# Cleanup
repo.delete_experiment(experiment_id)
```

**Directory structure:**
```
results/
├── raw/
│   └── exp_20240115_123456/
│       ├── process_0.json
│       ├── process_1.json
│       └── process_2.json
└── aggregated/
    └── exp_20240115_123456.json
```

### aggregation.py
Aggregate raw results from multiple processes.

```python
from llenergymeasure.results.aggregation import AggregationContext, aggregate_results

# Build context with all aggregation parameters
ctx = AggregationContext(
    experiment_id="exp_20240115_123456",
    measurement_config_hash="abc123def456abcd",
)

# Aggregate raw results into single result
aggregated = aggregate_results(raw_results, ctx)
```

**Aggregation logic:**
- Energy: **Sum** across all processes (each GPU contributes)
- Throughput: **Average** across processes
- Tokens: **Sum** across processes
- FLOPs: **Sum** across processes
- Timestamps: Min start, max end

**Verification flags:**
```python
AggregationMetadata(
    method="sum_energy_avg_throughput",
    num_processes=4,
    temporal_overlap_verified=True,   # Processes ran concurrently
    gpu_attribution_verified=True,    # Unique GPU IDs (no double count)
    warnings=["GPU 0 appears in multiple processes"],
)
```

### exporters.py
Export results to various formats.

```python
from llenergymeasure.results import export_to_csv, export_to_json

# Export aggregated results to CSV
export_to_csv(results, "output.csv")

# Export to JSON
export_to_json(results, "output.json")
```

## Result Schema

Raw results and aggregated results both include:
```json
{
  "schema_version": "2.0.0",
  "experiment_id": "exp_20240115_123456",
  ...
}
```

Schema version allows backward-compatible loading of older results.

## Security

- Experiment IDs are sanitized before filesystem operations
- Path traversal prevented via `is_safe_path()` checks
- See `../utils/security.py` for implementation

## CLI Commands

```bash
# Aggregate single experiment
lem aggregate exp_20240115_123456

# Aggregate all pending
lem aggregate --all

# Re-aggregate (overwrite existing)
lem aggregate exp_id --force

# List experiments
lem results list      # Aggregated only
lem results list --all  # Include pending

# Show results
lem results show exp_id
lem results show exp_id --raw
lem results show exp_id --json
```

## Related

- See `../domain/README.md` for result models
- See `../cli/` for CLI commands
