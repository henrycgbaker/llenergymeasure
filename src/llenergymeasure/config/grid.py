"""Sweep grid expansion and cycle ordering for study configurations."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError
from rich.panel import Panel
from rich.text import Text

from llenergymeasure.config._dict_utils import _unflatten, deep_merge
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import SOURCE_MULTI_BACKEND_ELEVATION
from llenergymeasure.utils.exceptions import ConfigError

if TYPE_CHECKING:
    from llenergymeasure.config.models import StudyConfig
    from llenergymeasure.infra.runner_resolution import RunnerSpec

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""


logger = logging.getLogger(__name__)

# Keys that belong to the study YAML structure, not to individual experiments.
# These are stripped from base: files and excluded from the fixed dict.
# "runners" is study-level metadata (per-backend runner config) — not an experiment field.
_STUDY_ONLY_KEYS = frozenset(
    {
        "sweep",
        "experiments",
        "study_execution",
        "base",
        "study_name",
        "version",
        "runners",
        "output",
    }
)


class ExperimentOrder(StrEnum):
    SEQUENTIAL = "sequential"
    INTERLEAVE = "interleave"
    SHUFFLE = "shuffle"
    REVERSE = "reverse"
    LATIN_SQUARE = "latin_square"


@dataclass
class SkippedConfig:
    """An ExperimentConfig that failed Pydantic validation during grid expansion."""

    raw_config: dict[str, Any]
    reason: str
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def short_label(self) -> str:
        """Short label for display: 'backend, dtype'."""
        backend = self.raw_config.get("backend", "unknown")
        dtype = self.raw_config.get("dtype", "?")
        return f"{backend}, {dtype}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for StudyConfig.skipped_configs."""
        return {
            "raw_config": self.raw_config,
            "reason": self.reason,
            "short_label": self.short_label,
            "errors": self.errors,
        }


# =============================================================================
# Public API
# =============================================================================


def expand_grid(
    raw_study: dict[str, Any],
    study_yaml_path: Path | None = None,
) -> tuple[list[ExperimentConfig], list[SkippedConfig]]:
    """Expand sweep dimensions into a flat list of ExperimentConfig.

    Resolution order:
    1. Load base: file (optional DRY inheritance)
    2. Build fixed dict from non-sweep/non-experiments/non-study_execution/non-base/non-study_name keys
    3. Expand sweep: block into raw config dicts
    4. Append explicit experiments: list entries
    5. Pydantic-validate each raw dict, collecting valid + skipped

    Returns (valid_experiments, skipped_configs).
    Raises ConfigError if all configs are invalid or no experiments produced.
    """
    # Step 1: Load base: inheritance
    base_dict = _load_base(raw_study.get("base"), study_yaml_path)

    # Step 2: Fixed dict — experiment-level fields shared across all grid points
    fixed = _extract_fixed(raw_study)
    merged_fixed = {**base_dict, **fixed}  # inline fields override base

    # Step 3: Expand sweep: block into raw config dicts
    sweep = raw_study.get("sweep", {})
    sweep_raw_configs = _expand_sweep(sweep, merged_fixed)

    # Step 4: Append explicit experiments: list entries
    # Strip non-matching backend sections *inherited from fixed*, but preserve
    # any the user wrote directly in the experiment entry (those are genuine
    # misconfigurations and should fail Pydantic validation).
    explicit_entries = raw_study.get("experiments", [])
    explicit_raw_configs = []
    for exp in explicit_entries:
        merged = {**merged_fixed, **exp}
        backend = merged.get("backend", merged_fixed.get("backend", "pytorch"))
        for key in _BACKEND_SECTION_KEYS:
            if key != backend and key in merged and key not in exp:
                del merged[key]
        explicit_raw_configs.append(merged)

    all_raw_configs = sweep_raw_configs + explicit_raw_configs

    # Guard: no experiments produced at all
    if not all_raw_configs:
        raise ConfigError(
            "Study produced no experiments. "
            "Add a 'sweep:' block, 'experiments:' list, or inline 'model:' field."
        )

    # Step 5: Pydantic-validate each raw dict
    valid: list[ExperimentConfig] = []
    skipped: list[SkippedConfig] = []

    for raw_config in all_raw_configs:
        try:
            valid.append(ExperimentConfig(**raw_config))
        except (ValidationError, TypeError) as exc:
            reason = str(exc)
            errors: list[dict[str, Any]] = []
            if isinstance(exc, ValidationError):
                errors = [dict(e) for e in exc.errors()]
            skipped.append(SkippedConfig(raw_config=raw_config, reason=reason, errors=errors))

    total = len(valid) + len(skipped)

    # Guard: all configs invalid
    if len(valid) == 0:
        first_reasons = "; ".join(s.reason[:120] for s in skipped[:5])
        raise ConfigError(
            f"nothing to run — all {total} generated config(s) are invalid. "
            f"First failures: {first_reasons}"
        )

    # Warning: >50% skip rate
    if total > 0 and len(skipped) / total > 0.5:
        logger.warning(
            "Most of your sweep is invalid (%d/%d configs skipped). "
            "Check your config combinations.",
            len(skipped),
            total,
        )
        for s in skipped:
            logger.warning("  Skipped (%s): %s", s.short_label, s.reason[:200])

    return valid, skipped


def compute_study_design_hash(experiments: list[ExperimentConfig]) -> str:
    """SHA-256[:16] of the resolved experiment list (execution block excluded).

    Deterministic: uses json.dumps with sort_keys=True. Identical experiment lists
    produce the same hash across calls and interpreter restarts.
    """
    canonical = json.dumps([exp.model_dump() for exp in experiments], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def apply_cycles(
    experiments: list[ExperimentConfig],
    n_cycles: int,
    experiment_order: ExperimentOrder,
    study_design_hash: str,
    shuffle_seed: int | None = None,
) -> list[ExperimentConfig]:
    """Return the ordered execution sequence for n_cycles repetitions.

    sequential:    [A, A, A, B, B, B]  — all cycles of each experiment together
    interleave:    [A, B, A, B, A, B]  — one cycle of each experiment, repeated
    shuffle:       random per-cycle order, seeded from study_design_hash by default
    reverse:       alternating forward/backward per cycle — [A, B, B, A, A, B]
    latin_square:  Williams balanced latin square (counterbalances carryover effects)
    """
    if experiment_order == ExperimentOrder.SEQUENTIAL:
        return [exp for exp in experiments for _ in range(n_cycles)]

    if experiment_order == ExperimentOrder.INTERLEAVE:
        return experiments * n_cycles

    if experiment_order == ExperimentOrder.REVERSE:
        result: list[ExperimentConfig] = []
        for i in range(n_cycles):
            cycle = list(experiments) if i % 2 == 0 else list(reversed(experiments))
            result.extend(cycle)
        return result

    if experiment_order == ExperimentOrder.LATIN_SQUARE:
        return _williams_latin_square(experiments, n_cycles)

    # shuffle
    seed = shuffle_seed if shuffle_seed is not None else int(study_design_hash, 16) & 0xFFFFFFFF
    rng = random.Random(seed)
    result = []
    for _ in range(n_cycles):
        cycle = list(experiments)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result


def _williams_latin_square(
    experiments: list[ExperimentConfig],
    n_cycles: int,
) -> list[ExperimentConfig]:
    """Generate a Williams balanced latin square ordering.

    A Williams design is a latin square where each condition follows every other
    condition exactly once across rows, balancing first-order carryover effects.
    When n_cycles > k (number of experiments), cycles repeat the square rows.
    When n_cycles < k, the first n_cycles rows are used.
    """
    k = len(experiments)
    if k == 0:
        return []

    # Build Williams square rows (works for both even and odd k)
    rows: list[list[int]] = []
    for i in range(k):
        row: list[int] = [0] * k
        for j in range(k):
            if j == 0:
                row[j] = i
            elif j % 2 == 1:
                row[j] = (i + (j + 1) // 2) % k
            else:
                row[j] = (i - j // 2) % k
        rows.append(row)

    result: list[ExperimentConfig] = []
    for cycle_idx in range(n_cycles):
        row = rows[cycle_idx % k]
        result.extend(experiments[idx] for idx in row)
    return result


def format_preflight_summary(
    study_config: StudyConfig,
    skipped: list[SkippedConfig] | None = None,
) -> str:
    """Return pre-flight display string for terminal output.

    Format (CONTEXT.md locked):
        Study [abc123de]: 4 configs x 3 cycles = 12 runs
        Order: interleave
        Skipping 2/6: (per-skip log line with reason)
          - pytorch, fp32: [Pydantic message]
        WARNING: 67% of sweep configs are invalid — check your sweep dimensions.

    Args:
        study_config: Resolved StudyConfig (after load_study_config).
        skipped: Optional list of SkippedConfig from expand_grid.
            If None, derives skip info from study_config.skipped_configs.

    Returns:
        Multi-line string for terminal display.
    """
    n_cycles = study_config.study_execution.n_cycles
    n_runs = len(study_config.experiments)
    # n_configs is the unique config count (before cycle multiplication)
    n_configs = n_runs // n_cycles if n_cycles > 0 else n_runs
    hash_display = study_config.study_design_hash or "unknown"

    lines = [
        f"Study [{hash_display}]: {n_configs} configs x {n_cycles} cycles = {n_runs} runs",
        f"Order: {study_config.study_execution.experiment_order}",
    ]

    # Handle skipped configs display
    skipped_dicts = (
        study_config.skipped_configs if skipped is None else [s.to_dict() for s in skipped]
    )
    if skipped_dicts:
        total_generated = n_configs + len(skipped_dicts)
        lines.append(f"Skipping {len(skipped_dicts)}/{total_generated}:")
        for s in skipped_dicts:
            label = s.get("short_label", "unknown")
            reason = s.get("reason", "unknown error")
            lines.append(f"  - {label}: {reason}")

        skip_rate = len(skipped_dicts) / total_generated if total_generated > 0 else 0
        if skip_rate > 0.5:
            lines.append(
                f"  WARNING: {skip_rate:.0%} of sweep configs are invalid "
                "-- check your sweep dimensions."
            )

    return "\n".join(lines)


def build_preflight_panel(
    study_config: StudyConfig,
    runner_specs: dict[str, RunnerSpec] | None = None,
    study_dir: Path | None = None,
    probed_energy_sampler: str | None = None,
) -> Panel:
    """Return a Rich Panel with study metadata, sweep dimensions, and design hash.

    The panel shows:
    - Border title: "Study: <name>"
    - Execution Controls: experiments, cycle order, gaps, shuffle seed, skip preflight
    - Runners: per-backend runner mode with auto-elevation annotation
    - Study-wide constants: model, n, dataset, energy sampler (when not swept)
    - Sweep dimensions: varying fields with nested sub-config grouping
    - Dimmed design hash at the bottom

    When ``runner_specs`` is provided (resolved by pre-flight), the panel shows
    effective runner modes. Otherwise falls back to YAML-declared runners.

    When ``study_dir`` is provided, the expected results path is shown below
    the hash. Skipped configs are NOT included; callers display them separately.
    """
    exec_cfg = study_config.study_execution
    n_cycles = exec_cfg.n_cycles
    n_runs = len(study_config.experiments)
    n_configs = n_runs // n_cycles if n_cycles > 0 else n_runs
    hash_display = study_config.study_design_hash or "unknown"

    # --- Pluralisation helper ---
    def _pl(n: int, singular: str, plural: str | None = None) -> str:
        if n == 1:
            return f"{n} {singular}"
        return f"{n} {plural or singular + 's'}"

    # --- Helpers ---
    def _section(body: Text, title: str) -> None:
        body.append("\n  ")
        body.append(title, style="bold")
        body.append("\n")

    def _line(body: Text, label: str, value: str, indent: int = 4) -> None:
        body.append(f"{' ' * indent}{label:<18}")
        body.append(f"{value}\n")

    # --- Unique values across experiments (single pass) ---
    _backends: set[str] = set()
    _models: set[str] = set()
    _ns: set[int] = set()
    _datasets: set[str] = set()
    _max_ins: set[int | None] = set()
    _max_outs: set[int | None] = set()
    _energy: set[str] = set()
    for exp in study_config.experiments:
        _backends.add(exp.backend)
        _models.add(exp.model)
        _ns.add(exp.dataset.n_prompts)
        _datasets.add(exp.dataset.source)
        _max_ins.add(exp.max_input_tokens)
        _max_outs.add(exp.max_output_tokens)
        _energy.add(str(exp.energy_sampler) if exp.energy_sampler is not None else "disabled")
    unique_backends = sorted(_backends)
    unique_models = sorted(_models)
    unique_n = sorted(_ns)
    unique_datasets = sorted(_datasets)
    unique_max_in = sorted(v for v in _max_ins if v is not None)
    unique_max_out = sorted(v for v in _max_outs if v is not None)
    unique_energy = sorted(_energy)

    experiments_line = (
        f"{_pl(n_configs, 'config')} x {_pl(n_cycles, 'cycle')} = {_pl(n_runs, 'run')}"
    )

    # --- Resolve energy sampler display ---
    # Skip host probe when all runners are Docker (container has different samplers).
    all_docker = runner_specs is not None and all(
        spec.mode == "docker" for spec in runner_specs.values()
    )
    energy_display = _resolve_energy_display(
        unique_energy, probed_sampler=probed_energy_sampler, skip_probe=all_docker
    )

    # --- Sweep dimensions ---
    sweep_dimensions = _collect_sweep_dimensions(study_config.experiments)

    # --- Assemble body ---
    body = Text()

    # -- Execution Controls --
    _section(body, "Execution Controls")
    _line(body, "Experiments", experiments_line)
    _line(body, "Cycle order", str(exec_cfg.experiment_order))
    exp_gap = (
        f"{exec_cfg.experiment_gap_seconds}s"
        if exec_cfg.experiment_gap_seconds is not None
        else "0s"
    )
    cyc_gap = f"{exec_cfg.cycle_gap_seconds}s" if exec_cfg.cycle_gap_seconds is not None else "0s"
    _line(body, "Experiment gap", exp_gap)
    _line(body, "Cycle gap", cyc_gap)
    shuffle_val = str(exec_cfg.shuffle_seed) if exec_cfg.shuffle_seed is not None else "auto"
    _line(body, "Shuffle seed", shuffle_val)
    skip_val = "yes" if exec_cfg.skip_preflight else "no"
    _line(body, "Skip preflight", skip_val)

    # -- Runners --
    _section(body, "Runners")
    yaml_runners = study_config.runners or {}
    for b in unique_backends:
        if runner_specs and b in runner_specs:
            spec = runner_specs[b]
            body.append(f"    {b:<18}{spec.mode}")
            if getattr(spec, "source", None) == SOURCE_MULTI_BACKEND_ELEVATION:
                body.append(" (auto-elevated)", style="yellow")
            body.append("\n")
            # Show image resolution for Docker backends
            if spec.mode == "docker" and spec.image:
                src = _IMAGE_SOURCE_SHORT.get(spec.image_source or "", spec.image_source or "")
                body.append(f"    {'':18}", style="dim")
                body.append(f"↳ {spec.image}", style="dim")
                body.append(f"  ({src})\n", style="dim")
        else:
            mode_str = str(yaml_runners.get(b, "local"))
            _line(body, b, mode_str)

    # -- Study-wide constants (shown when NOT varying across experiments) --
    constants: list[tuple[str, str]] = []
    if len(unique_models) == 1:
        constants.append(("Model", unique_models[0]))
    if len(unique_n) == 1:
        constants.append(("n_prompts", str(unique_n[0])))
    if len(unique_max_in) == 1:
        constants.append(("max_input_tokens", str(unique_max_in[0])))
    if len(unique_max_out) == 1:
        constants.append(("max_output_tokens", str(unique_max_out[0])))
    if len(unique_datasets) == 1:
        constants.append(("Dataset", unique_datasets[0]))
    constants.append(("Energy sampler", energy_display))
    if not study_config.output.save_timeseries:
        constants.append(("Save timeseries", "off"))

    if constants:
        _section(body, "Constants")
        for label, value in constants:
            _line(body, label, value)

    # -- Split sweep dimensions into shared (universal) and backend-scoped --
    shared_dims: list[tuple[str, str, int]] = []
    backend_dims: list[tuple[str, str, int]] = []

    in_backend_section = False
    for label, value, depth in sweep_dimensions:
        if depth == 0 and not value and label in _BACKEND_SECTION_KEYS:
            in_backend_section = True
        elif depth == 0 and label not in _BACKEND_SECTION_KEYS:
            in_backend_section = False

        if in_backend_section:
            backend_dims.append((label, value, depth))
        else:
            shared_dims.append((label, value, depth))

    def _render_dims(body: Text, dims: list[tuple[str, str, int]]) -> None:
        for label, value, depth in dims:
            indent = "    " + "  " * depth
            if value:
                body.append(f"{indent}{label:<18}")
                body.append(f"{value}\n", style="dim")
            else:
                # Sub-config header (e.g. "pytorch", "decoder")
                body.append(f"{indent}")
                body.append(f"{label}\n", style="bold dim")

    if shared_dims:
        _section(body, "Sweep Dimensions")
        _render_dims(body, shared_dims)

    if backend_dims:
        _section(body, "Backend Sweep Dimensions")
        _render_dims(body, backend_dims)

    body.append("\n")
    # Hash (dimmed)
    body.append("Study design hasH:\n ", style="dim")
    body.append(f"  {hash_display}\n", style="dim")
    # Results path (bold cyan)
    if study_dir is not None:
        body.append("\n")
        body.append("Study results path:\n", style="bold cyan")
        body.append(f"  {study_dir}/\n", style="bold cyan")

    return Panel(
        body,
        title=f"[bold cyan]Study: {study_config.study_name or 'unnamed'}[/]",
        title_align="left",
        padding=(0, 1),
    )


_IMAGE_SOURCE_SHORT: dict[str, str] = {
    "local_build": "local build",
    "registry": "registry",
    "env": "env var",
    "yaml": "study YAML",
    "runner_override": "runner override",
    "user_config": "user config",
}

_ENERGY_SAMPLER_NAMES: dict[str, str] = {
    "nvml": "NVMLSampler",
    "zeus": "ZeusSampler",
    "codecarbon": "CodeCarbonSampler",
}


def _resolve_energy_display(
    unique_energy: list[str],
    *,
    probed_sampler: str | None = None,
    skip_probe: bool = False,
) -> str:
    """Build the energy sampler display string, resolving 'auto' when possible.

    When ``skip_probe`` is True (all runners are Docker), the host probe is
    skipped because the container may have different energy samplers available.
    When ``probed_sampler`` is provided it is used to annotate 'auto' entries.
    """
    parts: list[str] = []
    for e in unique_energy:
        if e == "auto":
            if skip_probe or probed_sampler is None:
                parts.append("auto")
            else:
                parts.append(f"{probed_sampler} (auto)")
        elif e in _ENERGY_SAMPLER_NAMES:
            parts.append(_ENERGY_SAMPLER_NAMES[e])
        else:
            parts.append(e)
    return ", ".join(parts)


def _collect_sweep_dimensions(
    experiments: list[ExperimentConfig],
) -> list[tuple[str, str, int]]:
    """Return (label, value_str, depth) triples for sweep display.

    depth=0  top-level fields and sub-config headers
    depth=1  sub-config fields and nested sub-config headers
    depth=2+ nested sub-config fields (recursion)

    A triple with an empty value string is a header; non-empty is a field.
    """
    if not experiments:
        return []

    result: list[tuple[str, str, int]] = []

    # --- Top-level scalar fields (always show if they vary OR are key sweep fields) ---
    KEY_SCALAR_FIELDS = (
        "model",
        "backend",
        "dtype",
        "max_input_tokens",
        "max_output_tokens",
    )
    SCALAR_FIELDS = (*KEY_SCALAR_FIELDS, "random_seed")

    for field_name in SCALAR_FIELDS:
        values = [getattr(exp, field_name) for exp in experiments]
        unique_vals = set(str(v) for v in values)
        if len(unique_vals) > 1:
            result.append((field_name, ", ".join(sorted(unique_vals)), 0))

    # --- Dataset sub-config fields ---
    DATASET_FIELDS = ("source", "n_prompts", "order")
    for field_name in DATASET_FIELDS:
        values = [getattr(exp.dataset, field_name) for exp in experiments]
        unique_vals = set(str(v) for v in values)
        if len(unique_vals) > 1:
            result.append((f"dataset.{field_name}", ", ".join(sorted(unique_vals)), 0))

    # --- Sub-config fields (recursive) ---
    SUB_CONFIGS_MANDATORY = ("decoder", "warmup", "baseline")
    SUB_CONFIGS_OPTIONAL = ("pytorch", "vllm", "tensorrt", "lora")

    def _get_sub_config_defaults(sub_config_name: str) -> dict[str, Any]:
        """Get default field values for a sub-config class."""
        from llenergymeasure.config.models import (
            BaselineConfig,
            DecoderConfig,
            WarmupConfig,
        )

        defaults_map = {
            "decoder": DecoderConfig(),
            "warmup": WarmupConfig(),
            "baseline": BaselineConfig(),
        }
        if sub_config_name in defaults_map:
            obj = defaults_map[sub_config_name]
            return {k: getattr(obj, k) for k in obj.model_fields}
        return {}

    def _collect_fields(
        objs: list[Any],
        defaults: dict[str, Any],
        field_depth: int,
    ) -> list[tuple[str, str, int]]:
        """Collect varying/non-default fields from a list of config objects.

        When a field's values are Pydantic models (nested sub-configs), recurse
        and emit a header + nested fields instead of the full repr string.

        Returns (label, value_str, depth) triples. Empty value_str = header.
        The caller is responsible for prepending the parent header.
        """
        first = objs[0]
        if not hasattr(first, "model_fields"):
            return []

        entries: list[tuple[str, str, int]] = []

        for field_name in first.model_fields:
            vals = [getattr(o, field_name) for o in objs]
            non_none = [v for v in vals if v is not None]

            # Detect nested Pydantic models - recurse instead of str()
            if non_none and hasattr(non_none[0], "model_fields"):
                nested = _collect_fields(non_none, {}, field_depth + 1)
                if nested:
                    entries.append((field_name, "", field_depth))  # nested header
                    entries.extend(nested)
                continue

            unique_vals = set(str(v) for v in vals)

            # Only show fields that actually vary across experiments.
            # Constant non-default values belong in the panel metadata, not sweep dims.
            if len(unique_vals) > 1:
                entries.append((field_name, ", ".join(sorted(unique_vals)), field_depth))

        return entries

    def _process_sub_config(sub_config_name: str, is_optional: bool) -> None:
        """Process a sub-config and add varying/non-default fields to result."""
        if is_optional:
            sub_configs = [getattr(exp, sub_config_name) for exp in experiments]
            active = [sc for sc in sub_configs if sc is not None]
            if not active:
                return
        else:
            active = [getattr(exp, sub_config_name) for exp in experiments]

        defaults = _get_sub_config_defaults(sub_config_name)
        entries = _collect_fields(active, defaults, field_depth=1)

        if entries:
            result.append((sub_config_name, "", 0))  # parent header
            result.extend(entries)

    for sc_name in SUB_CONFIGS_MANDATORY:
        _process_sub_config(sc_name, is_optional=False)

    for sc_name in SUB_CONFIGS_OPTIONAL:
        _process_sub_config(sc_name, is_optional=True)

    return result


# =============================================================================
# Private helpers
# =============================================================================


def _extract_fixed(raw_study: dict[str, Any]) -> dict[str, Any]:
    """Return experiment-level fields from raw_study (all keys except study-only ones)."""
    return {k: v for k, v in raw_study.items() if k not in _STUDY_ONLY_KEYS}


def _load_base(base_path_str: str | None, study_yaml_path: Path | None) -> dict[str, Any]:
    """Load a base experiment config file, stripping study-only keys.

    Path is resolved relative to the study YAML file's directory.
    Hard error (ConfigError) if the file does not exist.
    """
    if base_path_str is None:
        return {}

    base_path = Path(base_path_str)
    if not base_path.is_absolute() and study_yaml_path is not None:
        base_path = study_yaml_path.parent / base_path

    if not base_path.exists():
        raise ConfigError(
            f"base: file not found: {base_path}. "
            "Path is resolved relative to the study YAML file's directory."
        )

    with base_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    # Strip study-only keys — base: accepts experiment config files only
    return {k: v for k, v in raw.items() if k not in _STUDY_ONLY_KEYS}


_BACKEND_SECTION_KEYS = frozenset({"pytorch", "vllm", "tensorrt"})


def _strip_other_backend_sections(config_dict: dict[str, Any], backend: str) -> dict[str, Any]:
    """Remove backend-specific sections that don't match *backend*.

    In a multi-backend study, top-level backend sections (e.g. ``tensorrt:``)
    are shared defaults for that backend's experiments.  When the grid expander
    assigns a different backend, those sections must be stripped before Pydantic
    validation - otherwise ``validate_backend_section_match`` rejects the config.
    """
    return {k: v for k, v in config_dict.items() if k not in _BACKEND_SECTION_KEYS or k == backend}


def _expand_sweep(sweep: dict[str, Any], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a sweep: block into a flat list of raw experiment config dicts.

    Dotted keys (e.g. pytorch.batch_size) are backend-scoped dimensions.
    Non-dotted keys are universal dimensions applied to all backends.

    If the sweep is empty and fixed has a 'model' key, return [fixed] (single experiment).
    If the sweep is empty and no model, return [] (no experiments).
    """
    if not sweep:
        if fixed.get("model"):
            backend = fixed.get("backend", "pytorch")
            return [_strip_other_backend_sections(dict(fixed), backend)]
        return []

    # Separate universal dims from backend-scoped dims
    universal_dims: dict[str, list[Any]] = {}
    scoped_dims: dict[str, dict[str, list[Any]]] = {}  # {backend: {param: [values]}}

    for key, values in sweep.items():
        if not isinstance(values, list):
            values = [values]
        if "." in key:
            prefix, _param = key.split(".", 1)
            if prefix in _BACKEND_SECTION_KEYS:
                # Backend-scoped parameter: pytorch.batch_size, vllm.engine.block_size
                scoped_dims.setdefault(prefix, {})[_param] = values
            else:
                # Nested experiment config field: dataset.n_prompts, dataset.order
                universal_dims[key] = values
        else:
            universal_dims[key] = values

    # Determine which backends to iterate over
    fixed_backend = fixed.get("backend", "pytorch")
    if isinstance(fixed_backend, list):
        backends = list(fixed_backend)
    elif scoped_dims:
        backends = list(scoped_dims.keys())
    else:
        backends = [fixed_backend]

    results: list[dict[str, Any]] = []

    for backend in backends:
        # Combine universal dims with this backend's scoped dims
        backend_scoped = scoped_dims.get(backend, {})
        all_dim_keys = list(universal_dims.keys()) + list(backend_scoped.keys())
        all_dim_values = list(universal_dims.values()) + list(backend_scoped.values())

        if not all_dim_keys:
            # No dimensions for this backend — produce one config
            config_dict: dict[str, Any] = _strip_other_backend_sections(dict(fixed), backend)
            # Remove list-valued backend field
            config_dict["backend"] = backend
            results.append(config_dict)
            continue

        for combo in itertools.product(*all_dim_values):
            config_dict = _strip_other_backend_sections(dict(fixed), backend)
            # Override list-valued backend with the specific backend string
            config_dict["backend"] = backend

            for dim_key, value in zip(all_dim_keys, combo, strict=True):
                if dim_key in backend_scoped:
                    # Backend-scoped parameter: recursively unflatten to handle multi-level paths.
                    # "engine.block_size" -> {"engine": {"block_size": value}}
                    backend_dict = config_dict.get(backend, {})
                    nested_update = _unflatten({dim_key: value})
                    config_dict[backend] = deep_merge(backend_dict, nested_update)
                elif "." in dim_key:
                    # Nested universal parameter: dataset.n_prompts -> {"dataset": {"n_prompts": value}}
                    nested_update = _unflatten({dim_key: value})
                    config_dict = deep_merge(config_dict, nested_update)
                else:
                    # Simple universal parameter: goes at top level
                    config_dict[dim_key] = value

            results.append(config_dict)

    return results
