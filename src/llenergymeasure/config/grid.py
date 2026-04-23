"""Sweep grid expansion and cycle ordering for study configurations."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError
from rich.panel import Panel
from rich.text import Text

from llenergymeasure.config._dict_utils import _unflatten, deep_merge
from llenergymeasure.config.introspection import (
    get_display_label,
    get_field_role,
    get_swept_field_paths,
)
from llenergymeasure.config.models import (
    DatasetConfig,
    ExperimentConfig,
    MeasurementConfig,
    TaskConfig,
)
from llenergymeasure.config.ssot import ALL_ENGINES, SOURCE_MULTI_ENGINE_ELEVATION
from llenergymeasure.utils.exceptions import ConfigError

if TYPE_CHECKING:
    from llenergymeasure.config.models import StudyConfig
    from llenergymeasure.infra.runner_resolution import RunnerSpec

from llenergymeasure.utils.compat import StrEnum

logger = logging.getLogger(__name__)

# Keys that belong to the study YAML structure, not to individual experiments.
# These are stripped from base: files and excluded from the fixed dict.
# "runners" is study-level metadata (per-engine runner config) — not an experiment field.
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
        """Short label for display: 'engine, dtype'. dtype lives under the engine section."""
        engine = self.raw_config.get("engine", "unknown")
        section = self.raw_config.get(engine) if isinstance(engine, str) else None
        dtype = section.get("dtype", "?") if isinstance(section, dict) else "?"
        return f"{engine}, {dtype}"

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
    # Strip non-matching engine sections *inherited from fixed*, but preserve
    # any the user wrote directly in the experiment entry (those are genuine
    # misconfigurations and should fail Pydantic validation).
    explicit_entries = raw_study.get("experiments", [])
    explicit_raw_configs = []
    for exp in explicit_entries:
        merged = {**merged_fixed, **exp}
        engine = merged.get("engine", merged_fixed.get("engine", "transformers"))
        for key in ALL_ENGINES:
            if key != engine and key in merged and key not in exp:
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

    # Combinatorial explosion warnings (tiered)
    n_valid = len(valid)
    exec_cfg = raw_study.get("study_execution", {})
    n_cycles = exec_cfg.get("n_cycles", 1) if isinstance(exec_cfg, dict) else 1
    total_runs = n_valid * n_cycles
    gap_seconds = (
        exec_cfg.get("experiment_gap_seconds", 0) if isinstance(exec_cfg, dict) else 0
    ) or 0

    if n_valid > 2000:
        min_hours = total_runs * gap_seconds / 3600
        logger.warning(
            "Extremely large study: %d experiments (%d total runs). "
            "Minimum runtime: ~%.0fh (gap time only). "
            "Consider reducing sweep dimensions or groups.",
            n_valid,
            total_runs,
            min_hours,
        )
    elif n_valid > 500:
        min_hours = total_runs * gap_seconds / 3600
        logger.warning(
            "Very large study: %d experiments (%d total runs with %d cycles). "
            "Minimum runtime: ~%.0fh (gap time only). "
            "Consider reducing sweep dimensions or groups.",
            n_valid,
            total_runs,
            n_cycles,
            min_hours,
        )
    elif n_valid > 100:
        logger.info("Large study: %d experiments.", n_valid)

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
          - transformers, fp32: [Pydantic message]
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
    sweep_axes: int | None = None,
    sweep_groups: int | None = None,
    n_explicit: int = 0,
) -> Panel:
    """Return a Rich Panel with study preflight summary.

    The panel shows:
    - Border title: "Study: <name>"
    - Execution Controls: experiments, experiment order, gaps, shuffle seed
    - Workload: all workload fields; swept fields annotated with "+"
    - Engines: per-engine runner mode with auto-elevation annotation
    - Sweep: summary line with axis/group counts and unique configs
    - Dimmed design hash and results path at the bottom

    Field labels come from json_schema_extra display_label metadata (SSOT).
    Declared values display normally; defaulted values are dimmed.

    When ``runner_specs`` is provided (resolved by pre-flight), the panel shows
    effective runner modes. Otherwise falls back to YAML-declared runners.

    When ``sweep_axes`` / ``sweep_groups`` are provided (from the raw YAML
    sweep dict via ``count_sweep_structure``), the Sweep section shows the
    breakdown.  Otherwise falls back to counting varying field paths.

    Skipped configs are NOT included; callers display them separately.
    """
    exec_cfg = study_config.study_execution
    n_cycles = exec_cfg.n_cycles
    n_runs = len(study_config.experiments)
    n_configs = n_runs // n_cycles if n_cycles > 0 else n_runs
    hash_display = study_config.study_design_hash or "unknown"
    experiments = study_config.experiments

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

    def _line(
        body: Text,
        label: str,
        value: str,
        indent: int = 4,
        value_style: str = "dim",
    ) -> None:
        body.append(f"{' ' * indent}")
        body.append(f"{label:<18}", style="white")
        body.append(f"{value}\n", style=value_style)

    # --- Unique engines (for Engines section) ---
    unique_engines = sorted({exp.engine for exp in experiments})

    # --- Resolve energy sampler display ---
    unique_energy = sorted(
        {
            str(exp.measurement.energy_sampler)
            if exp.measurement.energy_sampler is not None
            else "disabled"
            for exp in experiments
        }
    )
    all_docker = runner_specs is not None and all(
        spec.mode == "docker" for spec in runner_specs.values()
    )
    energy_display = _resolve_energy_display(
        unique_energy, probed_sampler=probed_energy_sampler, skip_probe=all_docker
    )

    # --- Swept field paths (for annotation) ---
    swept_paths = get_swept_field_paths(experiments)

    # --- Assemble body ---
    body = Text()
    experiments_line = (
        f"{_pl(n_configs, 'config')} x {_pl(n_cycles, 'cycle')} = {_pl(n_runs, 'run')}"
    )

    # -- Execution Controls --
    _section(body, "Execution Controls")
    _line(body, "Experiments", experiments_line)
    _line(body, "Experiment order", str(exec_cfg.experiment_order))
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

    # -- Engines --
    _section(body, "Engines")
    yaml_runners = study_config.runners or {}
    for b in unique_engines:
        if runner_specs and b in runner_specs:
            spec = runner_specs[b]
            body.append("    ")
            body.append(f"{b:<18}", style="white")
            body.append(f"{spec.mode}", style="dim")
            if getattr(spec, "source", None) == SOURCE_MULTI_ENGINE_ELEVATION:
                body.append(" (auto-elevated)", style="yellow")
            body.append("\n")
            # Show image resolution for Docker engines
            if spec.mode == "docker" and spec.image:
                body.append("    ", style="dim")
                body.append(f"\u21b3 {spec.image}\n", style="dim")
        else:
            mode_str = str(yaml_runners.get(b, "local"))
            _line(body, b, mode_str)

    # -- Workload section --
    # Task fields + energy sampler. Swept fields annotated with "+" and bold.
    workload_rows: list[tuple[str, str, bool, bool]] = []  # (label, value, is_declared, is_swept)

    first_exp = experiments[0]
    task_declared = first_exp.task.model_fields_set

    for field_name, fi in TaskConfig.model_fields.items():
        if field_name == "dataset":
            dataset_first = first_exp.task.dataset
            dataset_declared = dataset_first.model_fields_set
            for ds_field, ds_fi in DatasetConfig.model_fields.items():
                ds_role = get_field_role(ds_fi)
                if ds_role != "workload":
                    continue
                ds_path = f"task.dataset.{ds_field}"
                is_swept = ds_path in swept_paths
                unique_vals = sorted(
                    {str(getattr(exp.task.dataset, ds_field)) for exp in experiments}
                )
                val_str = ", ".join(unique_vals)
                is_decl = ds_field in dataset_declared
                label = get_display_label(ds_fi, ds_field)
                workload_rows.append((label, val_str, is_decl, is_swept))
            continue

        task_path = f"task.{field_name}"
        is_swept = task_path in swept_paths
        unique_vals = sorted({str(getattr(exp.task, field_name)) for exp in experiments})
        val_str = ", ".join(unique_vals)
        label = get_display_label(fi, field_name)
        is_decl = field_name in task_declared
        workload_rows.append((label, val_str, is_decl, is_swept))

    # Energy sampler (from measurement)
    energy_fi = MeasurementConfig.model_fields["energy_sampler"]
    is_swept = "measurement.energy_sampler" in swept_paths
    label = get_display_label(energy_fi, "energy_sampler")
    workload_rows.append((label, energy_display, True, is_swept))

    if workload_rows:
        _section(body, "Workload")
        for label, val_str, is_decl, is_swept in workload_rows:
            annotation = " +" if is_swept else ""
            if is_swept:
                _line(body, label, f"{val_str}{annotation}", value_style="bold")
            elif is_decl:
                _line(body, label, val_str)
            else:
                _line(body, label, val_str, value_style="dim")

    # -- Sweep summary (only when multiple configs) --
    if n_configs > 1:
        _section(body, "Sweep")
        n_from_sweep = n_configs - n_explicit
        has_both = n_from_sweep > 0 and n_explicit > 0
        # Sweep-generated line
        if n_from_sweep > 0 and sweep_axes is not None and sweep_groups is not None:
            if sweep_groups > 0:
                sweep_line = (
                    f"{sweep_axes} axes . {sweep_groups} groups "
                    f"-> {_pl(n_from_sweep, 'config')} from sweep"
                )
            else:
                sweep_line = f"{sweep_axes} axes -> {_pl(n_from_sweep, 'config')} from sweep"
            body.append(f"    {sweep_line}\n")
        elif n_from_sweep > 0:
            n_dims = len(swept_paths)
            sweep_line = f"{_pl(n_dims, 'dimension')} -> {_pl(n_from_sweep, 'config')} from sweep"
            body.append(f"    {sweep_line}\n")
        # Explicit experiments line
        if n_explicit > 0:
            body.append(f"    {_pl(n_explicit, 'explicit experiment')}\n")
        # Total line (only when configs come from both sources)
        if has_both:
            body.append(f"    {_pl(n_configs, 'unique config')} total\n")

    # -- Dedup summary (sweep-dedup.md §2) --
    # Present whenever the canonicaliser collapsed configs. Shows the
    # N declared -> K canonical reduction that researchers use to
    # reason about measurement-equivalent runs.
    pre_run_groups = study_config.pre_run_equivalence_groups or []
    if pre_run_groups:
        declared = sum(int(g.get("member_count", 0)) for g in pre_run_groups)
        canonical = len(pre_run_groups)
        would_dedup = sum(1 for g in pre_run_groups if g.get("would_dedup"))
        if declared != canonical or would_dedup:
            _section(body, "Dedup")
            suffix = (
                f"{would_dedup} collapsed group(s)"
                if study_config.dedup_mode == "h1"
                else f"dedup OFF; {would_dedup} group(s) would collapse"
            )
            body.append(f"    {declared} declared -> {canonical} canonical ({suffix})\n")

    body.append("\n")
    # Hash (dimmed)
    body.append("Study design hash:\n ", style="dim")
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


def count_sweep_structure(raw_sweep: dict[str, Any]) -> tuple[int, int]:
    """Count independent axes and dependent groups in a raw sweep dict.

    An independent axis is a key mapping to a list of scalars (Cartesian product).
    A dependent group is a key mapping to a list of dicts (union of variants).

    Returns (n_axes, n_groups).
    """
    if not raw_sweep:
        return 0, 0

    n_axes = 0
    n_groups = 0

    for _key, values in raw_sweep.items():
        if _is_group(values):
            n_groups += 1
        else:
            n_axes += 1

    return n_axes, n_groups


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


def _strip_other_engine_sections(config_dict: dict[str, Any], engine: str) -> dict[str, Any]:
    """Remove engine-specific sections that don't match *engine*.

    In a multi-engine study, top-level engine sections (e.g. ``tensorrt:``)
    are shared defaults for that engine's experiments.  When the grid expander
    assigns a different engine, those sections must be stripped before Pydantic
    validation - otherwise ``validate_engine_section_match`` rejects the config.
    """
    return {k: v for k, v in config_dict.items() if k not in ALL_ENGINES or k == engine}


# =============================================================================
# Sweep group helpers
# =============================================================================


def _is_group(value: object) -> bool:
    """True if a sweep entry is a group (list of dicts), not an independent axis.

    Disambiguation: a list of scalars is an independent axis (Cartesian product);
    a list of dicts (or containing ``{}``) is a dependent group (union of variants).

    Raises ``ConfigError`` for mixed lists (some dicts, some scalars).
    """
    if not isinstance(value, list) or len(value) == 0:
        return False
    has_dicts = any(isinstance(e, dict) for e in value)
    if not has_dicts:
        return False
    all_dicts = all(isinstance(e, dict) for e in value)
    if not all_dicts:
        raise ConfigError(
            "Sweep entry mixes dicts and scalars. Group entries must all be "
            "dicts; independent axes must all be scalars."
        )
    return True


def _group_engine_scope(group_key: str) -> str | None:
    """Return engine name if a group key is engine-scoped, else None (universal)."""
    if "." in group_key:
        prefix = group_key.split(".", 1)[0]
        if prefix in ALL_ENGINES:
            return prefix
    return None


def _expand_group_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a single group entry into one or more flat dicts (mini-grid).

    Scalar-valued fields pass through unchanged. List-valued fields (list of
    scalars) produce a Cartesian product within the entry. Nested lists like
    ``[[0, 1]]`` are treated as literal list values (not expanded).
    """
    scalar_fields: dict[str, Any] = {}
    grid_keys: list[str] = []
    grid_values: list[list[Any]] = []

    for key, value in entry.items():
        if isinstance(value, list) and len(value) > 0 and not isinstance(value[0], (list, dict)):
            # List of scalars -> mini-grid axis
            grid_keys.append(key)
            grid_values.append(value)
        else:
            scalar_fields[key] = value

    if not grid_keys:
        return [entry]

    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(*grid_values):
        variant = dict(scalar_fields)
        for key, value in zip(grid_keys, combo, strict=True):
            variant[key] = value
        expanded.append(variant)
    return expanded


def _expand_group(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand all entries in a group, flattening mini-grids into a union of variants."""
    variants: list[dict[str, Any]] = []
    for entry in entries:
        variants.extend(_expand_group_entry(entry))
    return variants


def _route_key_value(
    config_dict: dict[str, Any],
    key: str,
    value: Any,
) -> dict[str, Any]:
    """Route a single fully-qualified key into *config_dict*.

    Routing rules:
    - Engine-prefixed dotted key (``transformers.batch_size``) → merge into engine section.
    - Other dotted key (``task.dataset.source``) → unflatten at top level.
    - Simple key → direct assignment.

    Returns the (possibly replaced) config_dict reference.
    """
    if "." in key:
        prefix, param = key.split(".", 1)
        if prefix in ALL_ENGINES:
            engine_dict = config_dict.get(prefix, {})
            nested_update = _unflatten({param: value})
            config_dict[prefix] = deep_merge(engine_dict, nested_update)
        else:
            nested_update = _unflatten({key: value})
            config_dict = deep_merge(config_dict, nested_update)
    else:
        config_dict[key] = value
    return config_dict


def _apply_group_overlay(
    config_dict: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Apply a group variant's fully-qualified keys onto a config dict."""
    for fq_key, value in overlay.items():
        config_dict = _route_key_value(config_dict, fq_key, value)
    return config_dict


def _validate_sweep_groups(
    groups: dict[str, list[dict[str, Any]]],
    axis_keys: set[str],
) -> None:
    """Raise ConfigError if a group name collides with an independent axis key."""
    collisions = set(groups.keys()) & axis_keys
    if collisions:
        raise ConfigError(
            f"Sweep group name(s) collide with independent axis key(s): "
            f"{', '.join(sorted(collisions))}. "
            f"Use abstract names for groups (e.g. 'transformers.compilation' not 'transformers.torch_compile')."
        )


# =============================================================================
# Sweep expansion
# =============================================================================


def _expand_sweep(sweep: dict[str, Any], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a sweep: block into a flat list of raw experiment config dicts.

    Supports two entry types under ``sweep:``:

    - **Independent axes** (list of scalars): Cartesian product across all axes.
    - **Dependent groups** (list of dicts): Union of variant dicts. Groups are
      crossed with each other and with independent axes, but entries *within*
      a group are alternatives (unioned, not crossed).

    Type-based disambiguation: ``list[scalar]`` = axis, ``list[dict]`` = group.
    """
    if not sweep:
        task = fixed.get("task")
        has_model = isinstance(task, dict) and task.get("model")
        if has_model:
            engine = fixed.get("engine", "transformers")
            return [_strip_other_engine_sections(dict(fixed), engine)]
        return []

    # ── Step 1: Partition sweep into axes and groups ──
    universal_dims: dict[str, list[Any]] = {}
    scoped_dims: dict[str, dict[str, list[Any]]] = {}  # {engine: {param: [values]}}
    groups: dict[str, list[dict[str, Any]]] = {}  # {group_name: [variant_dicts]}

    for key, values in sweep.items():
        if _is_group(values):
            groups[key] = _expand_group(values)
            continue

        if not isinstance(values, list):
            values = [values]

        if "." in key:
            prefix, _param = key.split(".", 1)
            if prefix in ALL_ENGINES:
                scoped_dims.setdefault(prefix, {})[_param] = values
            else:
                universal_dims[key] = values
        else:
            universal_dims[key] = values

    # Derive flat axis key set for collision detection
    axis_keys = set(universal_dims.keys()) | {
        f"{b}.{p}" for b, params in scoped_dims.items() for p in params
    }
    _validate_sweep_groups(groups, axis_keys)

    # ── Step 2: Separate groups by engine scope ──
    universal_groups: dict[str, list[dict[str, Any]]] = {}
    scoped_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}  # {engine: {name: variants}}

    for group_name, variants in groups.items():
        engine_scope = _group_engine_scope(group_name)
        if engine_scope:
            scoped_groups.setdefault(engine_scope, {})[group_name] = variants
        else:
            universal_groups[group_name] = variants

    # ── Step 3: Determine engines ──
    fixed_engine = fixed.get("engine", "transformers")
    if isinstance(fixed_engine, list):
        engines = list(fixed_engine)
    elif scoped_dims or scoped_groups:
        # Engines implied by scoped axes or scoped groups
        engines = sorted(set(scoped_dims.keys()) | set(scoped_groups.keys()))
    else:
        engines = [fixed_engine]

    # ── Step 4: Per-engine expansion ──
    results: list[dict[str, Any]] = []

    for engine in engines:
        # Collect applicable groups (universal + this engine's scoped)
        applicable_groups: dict[str, list[dict[str, Any]]] = dict(universal_groups)
        applicable_groups.update(scoped_groups.get(engine, {}))

        # Collect applicable axes — reconstruct fully-qualified keys for routing
        engine_scoped = scoped_dims.get(engine, {})
        fq_dim_keys = list(universal_dims.keys()) + [f"{engine}.{p}" for p in engine_scoped]
        all_dim_values = list(universal_dims.values()) + list(engine_scoped.values())

        # Cross all group variant lists with each other (lazy — iterated once)
        group_combos: Iterable[tuple[Any, ...]]
        if applicable_groups:
            group_names = list(applicable_groups.keys())
            group_variant_lists = [applicable_groups[n] for n in group_names]
            group_combos = itertools.product(*group_variant_lists)
        else:
            group_combos = [()]  # single empty combo → no group overlays

        if not fq_dim_keys and not applicable_groups:
            # No dimensions or groups for this engine — produce one config
            config_dict: dict[str, Any] = _strip_other_engine_sections(dict(fixed), engine)
            config_dict["engine"] = engine
            results.append(config_dict)
            continue

        # Pre-compute stripped base config once per engine
        base_config = _strip_other_engine_sections(dict(fixed), engine)
        base_config["engine"] = engine

        # axis_combos materialised — reused across group combos
        axis_combos = list(itertools.product(*all_dim_values)) if fq_dim_keys else [()]

        for group_combo in group_combos:
            for axis_combo in axis_combos:
                config_dict = dict(base_config)

                # Apply independent axis values
                for dim_key, value in zip(fq_dim_keys, axis_combo, strict=True):
                    config_dict = _route_key_value(config_dict, dim_key, value)

                # Apply group overlays (each group_combo entry is one variant dict)
                for variant in group_combo:
                    if variant:  # skip empty dicts ({} = baseline, no overlay)
                        config_dict = _apply_group_overlay(config_dict, variant)

                results.append(config_dict)

    return results
