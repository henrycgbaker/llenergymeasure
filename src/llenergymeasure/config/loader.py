"""YAML/JSON configuration loader for experiment configs (v2.0).

Implements the v2.0 loading contract:
- Collect ALL errors before raising (not one-at-a-time)
- ConfigError with file path + did-you-mean for unknown fields
- CLI override merging at highest priority
- Native YAML anchor support via yaml.safe_load

Priority (highest wins): cli_overrides > path YAML > user_config_defaults
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llenergymeasure.config._dict_utils import _unflatten, deep_merge
from llenergymeasure.config.grid import (
    ExperimentOrder,
    apply_cycles,
    compute_study_design_hash,
    expand_grid,
)
from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.utils.exceptions import ConfigError

__all__ = ["deep_merge", "load_experiment_config", "load_study_config"]


# =============================================================================
# Public API
# =============================================================================


def load_experiment_config(
    path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    user_config_defaults: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load and validate experiment configuration.

    Priority (highest wins): cli_overrides > path YAML > user_config_defaults

    Args:
        path: Path to YAML or JSON config file. None = only CLI/defaults.
        cli_overrides: Dict of CLI flag overrides (e.g. {"model": "gpt2", "backend": "pytorch"}).
            Keys match ExperimentConfig field names. None values are ignored (unset flags).
        user_config_defaults: Dict of user config defaults to apply as lowest priority.
            Only fields valid on ExperimentConfig (e.g. output_dir, backend defaults).

    Returns:
        Validated ExperimentConfig.

    Raises:
        ConfigError: File not found, parse error, unknown fields, or structural validation failure.
            Includes all errors collected at once (not one-at-a-time).
        ValidationError: Pydantic field-level validation errors pass through unchanged.
            (Bad values like n=-1 are Pydantic's domain; unknown keys become ConfigError.)
    """
    # Start with user config defaults (lowest priority)
    merged: dict[str, Any] = {}
    if user_config_defaults:
        merged = deep_merge(
            merged, {k: v for k, v in user_config_defaults.items() if v is not None}
        )

    # Load and apply YAML/JSON file
    if path is not None:
        file_dict = _load_file(path)  # raises ConfigError on missing/parse error
        merged = deep_merge(merged, file_dict)

    # Apply CLI overrides (highest priority, skip None values)
    if cli_overrides:
        overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        merged = deep_merge(
            merged, _unflatten(overrides)
        )  # handle "pytorch.batch_size" dotted keys

    # Strip optional version field — not an ExperimentConfig field
    merged.pop("version", None)

    # Collect unknown field errors before handing to Pydantic
    known_fields = set(ExperimentConfig.model_fields.keys())
    unknown = set(merged.keys()) - known_fields
    if unknown:
        errors = []
        for key in sorted(unknown):
            suggestion = _did_you_mean(key, known_fields)
            msg = f"Unknown field '{key}'"
            if suggestion:
                msg += f" — did you mean '{suggestion}'?"
            if path:
                msg += f" (in {path})"
            errors.append(msg)
        raise ConfigError("\n".join(errors))

    # Construct ExperimentConfig — let ValidationError pass through unchanged
    try:
        return ExperimentConfig(**merged)
    except ValidationError:
        raise  # Pydantic field-level errors are not our domain to wrap
    except Exception as e:
        context = f" (in {path})" if path else ""
        raise ConfigError(f"Config construction failed{context}: {e}") from e


def load_study_config(
    path: Path | str,
    cli_overrides: dict[str, Any] | None = None,
) -> StudyConfig:
    """Load, expand, and validate a study YAML file.

    Resolution order:
      1. Load YAML file
      2. Apply CLI overrides on execution block
      3. Parse execution block (Pydantic validates it)
      4. expand_grid() — Cartesian product + Pydantic validation of each ExperimentConfig
      5. Guard: empty or all-invalid → ConfigError
      6. compute_study_design_hash() over valid experiments
      7. apply_cycles() for cycle ordering
      8. Construct and return StudyConfig

    This is the CFG-12 contract: sweep resolution at YAML parse time, before
    Pydantic sees the individual ExperimentConfig objects.

    Args:
        path: Path to study YAML file.
        cli_overrides: Optional dict of CLI flag overrides for execution block
            (e.g. {"study_execution": {"n_cycles": 5}}). The CLI translates
            --cycles/--order/--no-gaps flags into this dict.

    Returns:
        Resolved StudyConfig with ordered experiments and study_design_hash.

    Raises:
        ConfigError: File not found, parse error, base file missing, ALL configs invalid,
            empty study (no sweep and no experiments).
        ValidationError: Pydantic structural errors on ExecutionConfig pass through.
    """
    path = Path(path)
    raw = _load_file(path)  # reuse existing _load_file — raises ConfigError on missing/parse error

    # Apply CLI overrides (--cycles etc. translated into this dict)
    if cli_overrides:
        raw = deep_merge(raw, cli_overrides)

    # Strip version key (same as experiment loader)
    raw.pop("version", None)

    # Extract study-level metadata
    name = raw.get("study_name")
    # runners: per-backend runner config (e.g. {"pytorch": "local", "vllm": "docker"})
    # None if not specified in YAML — caller uses user config / auto-detection.
    runners: dict[str, str] | None = raw.get("runners") or None

    # Parse execution block — Pydantic validates it
    execution = ExecutionConfig(**(raw.get("study_execution") or {}))

    # Expand sweep → list[ExperimentConfig], collect skipped
    # This is CFG-12: sweep resolution at YAML parse time, before Pydantic
    valid_experiments, skipped = expand_grid(raw, study_yaml_path=path)

    # Guard: empty study — expand_grid already raises if all_raw_configs is empty,
    # but we also need to handle the degenerate case where expand_grid itself
    # returns no valid experiments and raises. If we reach here, valid_experiments
    # has at least one entry (expand_grid raises ConfigError if all invalid).
    # The "empty study" case (no model, no sweep, no experiments) is already
    # caught inside expand_grid(). We add an extra guard here for clarity:
    total = len(valid_experiments) + len(skipped)
    if total == 0:
        raise ConfigError("Study produced no experiments (empty sweep and no experiments: list).")
    if not valid_experiments:
        skip_details = "\n".join(f"  {s.short_label}: {s.reason}" for s in skipped[:5])
        raise ConfigError(
            f"All {total} generated configs are invalid. "
            "Nothing to run. Check sweep dimensions against backend constraints.\n" + skip_details
        )

    # Compute study_design_hash BEFORE applying cycles (hash is over unique configs)
    study_hash = compute_study_design_hash(valid_experiments)

    # Apply cycle ordering to produce execution sequence
    ordered = apply_cycles(
        valid_experiments,
        n_cycles=execution.n_cycles,
        experiment_order=ExperimentOrder(execution.experiment_order),
        study_design_hash=study_hash,
        shuffle_seed=execution.shuffle_seed,
    )

    return StudyConfig(
        experiments=ordered,
        study_name=name,
        study_execution=execution,
        runners=runners,
        study_design_hash=study_hash,
        skipped_configs=[s.to_dict() for s in skipped],
    )


# =============================================================================
# Private helpers
# =============================================================================


def _load_file(path: Path | str) -> dict[str, Any]:
    """Load YAML or JSON config file into a dict.

    Args:
        path: Path to config file.

    Returns:
        Parsed config dictionary.

    Raises:
        ConfigError: If file not found, unsupported format, parse error, or not a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            result = yaml.safe_load(content)  # native YAML anchors (&/*) handled automatically
        elif path.suffix == ".json":
            result = json.loads(content)
        else:
            raise ConfigError(f"Unsupported config format '{path.suffix}': use .yaml or .json")
        if not isinstance(result, dict):
            raise ConfigError(f"Config must be a mapping (got {type(result).__name__}): {path}")
        return result
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigError(f"Parse error in {path}: {e}") from e


def _did_you_mean(key: str, candidates: set[str], max_distance: int = 3) -> str | None:
    """Return the closest candidate if within max_distance edits, else None.

    Args:
        key: Unknown key to find a suggestion for.
        candidates: Set of valid field names.
        max_distance: Maximum Levenshtein distance to suggest (default 3).

    Returns:
        Closest candidate string, or None if nothing is close enough.
    """
    best: str | None = None
    best_dist = max_distance + 1
    for candidate in candidates:
        dist = _levenshtein(key, candidate)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best if best_dist <= max_distance else None


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]
