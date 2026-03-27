"""Dataset loading utilities for LLM efficiency measurement.

Provides JSONL loading for bundled datasets, file-path datasets, and synthetic
prompt generation. Used by inference backends to load the prompt workload for
each experiment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


# ---------------------------------------------------------------------------
# Built-in dataset registry
# ---------------------------------------------------------------------------

_BUILTIN_DIR: Path = Path(__file__).parent / "builtin"

BUILTIN_DATASETS: dict[str, Path] = {
    "aienergyscore": _BUILTIN_DIR / "aienergyscore.jsonl",
}

# Column names tried in order when detecting the prompt field in JSONL records.
AUTO_DETECT_COLUMNS: list[str] = ["prompt", "text", "instruction", "input", "question"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_prompts(config: ExperimentConfig) -> list[str]:
    """Load prompts according to the experiment configuration.

    Dispatches based on config.dataset type:
    - SyntheticDatasetConfig -> generate deterministic synthetic prompts
    - str matching a built-in alias -> load from bundled JSONL file
    - str path to an existing .jsonl file -> load from that file
    - anything else -> raise ValueError with valid options

    Args:
        config: Experiment configuration. Uses config.dataset, config.n,
            config.dataset_order, and config.random_seed.

    Returns:
        List of exactly config.n prompt strings.

    Raises:
        ValueError: If dataset name is unknown or file not found.
        ValueError: If the dataset has fewer prompts than config.n requests.
    """
    # Lazy import to avoid circular dependency (config.models imports nothing
    # from datasets, but keeping the import lazy is belt-and-suspenders).
    from llenergymeasure.config.models import SyntheticDatasetConfig

    if isinstance(config.dataset, SyntheticDatasetConfig):
        return _load_synthetic(config.dataset, config.n, fallback_seed=config.random_seed)

    if isinstance(config.dataset, str):
        # Built-in alias
        if config.dataset in BUILTIN_DATASETS:
            path = BUILTIN_DATASETS[config.dataset]
            return _load_jsonl(
                path,
                n=config.n,
                name=config.dataset,
                order=config.dataset_order,
                seed=config.random_seed,
            )

        # User-supplied file path
        path = Path(config.dataset)
        if path.suffix == ".jsonl" and path.exists():
            return _load_jsonl(
                path,
                n=config.n,
                name=str(path),
                order=config.dataset_order,
                seed=config.random_seed,
            )

        valid = ", ".join(f'"{k}"' for k in sorted(BUILTIN_DATASETS))
        raise ValueError(
            f"Unknown dataset {config.dataset!r}. "
            f"Valid built-in aliases: {valid}. "
            "To use a custom dataset, provide a path to an existing .jsonl file."
        )

    raise ValueError(
        f"config.dataset must be a str or SyntheticDatasetConfig, got {type(config.dataset)!r}"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_jsonl(
    path: Path,
    n: int,
    name: str,
    order: str = "interleaved",
    *,
    seed: int,
) -> list[str]:
    """Load prompts from a JSONL file.

    Skips lines that are provenance headers (keys starting with '_').
    Tries 'prompt' then 'text' as the prompt field for each record.

    Args:
        path: Path to the JSONL file.
        n: Number of prompts to return.
        name: Dataset name (for error messages).
        order: One of 'interleaved', 'grouped', 'shuffled'.
        seed: Random seed used when order='shuffled'.

    Returns:
        List of exactly n prompt strings.

    Raises:
        ValueError: If fewer than n prompts are available.
    """
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip provenance header lines (keys start with '_')
            if any(k.startswith("_") for k in record):
                continue

            records.append(record)

            if order == "interleaved" and len(records) >= n:
                break

    if len(records) < n:
        raise ValueError(
            f"Dataset {name!r} has only {len(records)} prompts but {n} were requested."
        )

    # Apply ordering
    if order == "interleaved":
        # File order is already interleaved (written round-robin by source)
        ordered = records
    elif order == "grouped":
        # Sort by 'source' field (stable sort preserves intra-group order)
        ordered = sorted(records, key=lambda r: r.get("source", ""))
    elif order == "shuffled":
        ordered = list(records)
        random.Random(seed).shuffle(ordered)
    else:
        raise ValueError(
            f"Unknown dataset_order {order!r}. Expected: interleaved, grouped, shuffled."
        )

    selected = ordered[:n]

    # Extract prompt text from each record
    prompts: list[str] = []
    for record in selected:
        for col in AUTO_DETECT_COLUMNS:
            if col in record and isinstance(record[col], str) and record[col].strip():
                prompts.append(record[col].strip())
                break

    if len(prompts) < n:
        raise ValueError(
            f"Could not extract {n} prompts from {name!r}. "
            f"Only {len(prompts)} records had a recognisable prompt field."
        )

    return prompts


def _load_synthetic(config: object, n: int, *, fallback_seed: int) -> list[str]:
    """Generate deterministic synthetic prompts.

    Uses the same word-repetition approach as the M1 placeholder but with
    seeded randomness for reproducibility.

    Args:
        config: SyntheticDatasetConfig instance (duck-typed to avoid circular import).
        n: Number of prompts to generate.
        fallback_seed: Seed to use when config.seed is None (typically
            ExperimentConfig.random_seed).

    Returns:
        List of n synthetic prompt strings.
    """
    input_len: int = getattr(config, "input_len", 512)
    explicit_seed: int | None = getattr(config, "seed", None)
    seed: int = explicit_seed if explicit_seed is not None else fallback_seed

    # Approximate: ~4 chars per token
    words_per_prompt = max(1, input_len // 4)
    rng = random.Random(seed)
    words = ["Hello", "world", "the", "a", "is", "it", "this", "that", "with", "for"]

    prompts: list[str] = []
    for _i in range(n):
        selected = [rng.choice(words) for _ in range(words_per_prompt)]
        prompts.append(" ".join(selected))

    return prompts
