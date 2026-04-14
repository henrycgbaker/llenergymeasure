# configs/

Example YAML configs for `llem run`.

## Files

| File | Purpose |
|------|---------|
| `example-study-full.yaml` | **Maximalist showcase** — every curated typed field, once. Not runnable as-is. |
| `test.yaml` | Ad-hoc scratch config (not committed). |

## Showcase vs practical configs

`example-study-full.yaml` is a **full-surface reference**, not a starting point for real
runs. It populates all three engine backends simultaneously (Transformers, vLLM,
TensorRT-LLM), which is invalid for an actual study — a real run uses one engine per
experiment. Its purpose is discoverability and CI coverage.

For a practical starting point, use the minimal YAML patterns in `docs/study-config.md`.

## Coverage verifier

```bash
uv run python scripts/verify_example_coverage.py
```

Asserts that every curated typed field (94 total across all engines) appears in
`example-study-full.yaml`. Exits 1 if any field is missing or if no curated fields
are found (indicating a broken `CurationMetadata` setup).
