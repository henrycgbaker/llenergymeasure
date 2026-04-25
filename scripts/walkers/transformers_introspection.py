"""Transformers library-API introspection walker — combinatorial-probe edition.

Derives validation rules from HF's runtime machinery by combinatorially
sweeping related kwargs and inferring predicates from the raise/no-raise
pattern. Replaces the old single-kwarg-perturbation extractor; the
single-pass approach can't see cross-field invariants (modulo, comparison,
range-vs-other-field) because those only fire on combinations.

Three extraction paths, all observe the library at walk time:

1. **Mode-gated dormancy auto-enumeration** (kept from prior implementation).
   For each known trigger class (greedy, single-beam, no-return-dict),
   enumerate every public ``GenerationConfig()`` field, synthesise a non-default
   probe value for its Python type, invoke ``validate(strict=True)``, and
   record any field HF reports as dormant in the composed raise.

2. **Combinatorial cluster probing** (new). For each named cluster of
   related kwargs (e.g. beam-search: num_beams, num_beam_groups,
   diversity_penalty, early_stopping, length_penalty), generate the
   Cartesian product of representative values per field, instantiate
   ``GenerationConfig(**kwargs)`` (and call ``.validate(strict=True)`` if
   construction succeeds), and tabulate which combinations raise / which
   normalise / which pass. A predicate-inference pass then groups error
   rows by message-class and emits one rule per inferred predicate.

3. **Validate-time self-triggered dormant probes** (kept). The
   ``pad_token_id < 0`` family — dormancy gated by the field's own value
   rather than a mode flag.

Predicate inference (cluster path) covers, in order of preference:

- **Cross-field divisibility** — error rows align with ``a % b != 0``.
- **Cross-field comparison** — error rows align with ``a > b`` (or any of
  ``<, <=, >=, ==, !=``).
- **Cross-field equality predicate** — error rows correlate with
  ``kwargs[a] == V`` AND ``kwargs[b] == W`` (multi-field gate).
- **Type allowlist** — error rows correlate with the type of one kwarg's
  value not being in an allowed set.
- **Single-field range** — error rows correlate with one kwarg crossing a
  threshold (``< 0``, ``<= 0``, etc.).
- **Single-field equality** — error rows correlate with one kwarg taking
  one specific value.

Recall over precision: when multiple predicates fit, the inferrer emits
ALL plausible candidates with ``walker_confidence: low``. The vendor CI
pipeline downstream re-runs each emitted rule's ``kwargs_positive`` /
``kwargs_negative`` against the live library; rules that misfire fail CI
and are pruned. False positives are cheap; missed invariants are not.

Every rule this walker emits carries ``added_by="introspection"``.

BitsAndBytesConfig coverage gap
-------------------------------
This extractor is scoped to ``GenerationConfig`` (and its depth-1
``WatermarkingConfig`` / ``SynthIDTextWatermarkingConfig`` helpers). BNB
``post_init`` type-check raises are NOT emitted here. The pre-pipeline
walker (:mod:`scripts.walkers.transformers`, deregistered) hand-curated
nine BNB rules; that path was lost in the refactor.

Coverage is restored structurally by :mod:`scripts.walkers.transformers_ast`,
which AST-walks ``BitsAndBytesConfig.post_init`` directly — the
``if not isinstance(self.X, T): raise`` pattern is exactly what its
``type_is_not`` predicate path already handles. The AST walker reads
``transformers.utils.quantization_config`` source via
``inspect.getsourcefile`` rather than importing the library, which
keeps the walker fast and dependency-free.

**Re. CUDA**: an earlier inherited comment claimed BNB introspection
"touches CUDA". That's overstated. ``import bitsandbytes`` discovers
CUDA libs at module load and emits a warning on CPU-only hosts but
returns successfully; ``BitsAndBytesConfig(**kwargs).post_init()`` is
pure Python type-checking with no CUDA calls. Probing BNB construction
is therefore CI-runner-agnostic — fits cleanly on the GH-hosted
``requires_gpu: false`` runner pool. The reason this extractor doesn't
yet probe BNB is **scope** (the cluster wasn't added in the refactor),
not CUDA.

If/when BNB probing becomes useful (cross-validation against the AST
walker, or to catch ``__init__``-time gates the AST walker doesn't
currently walk like ``load_in_4bit AND load_in_8bit``), add a
``bitsandbytes_quant`` cluster: the ``_Cluster`` pattern below
generalises directly to ``BitsAndBytesConfig`` — swap the constructor
in :func:`_probe_cluster` and supply representative kwargs per BNB
field.
"""

from __future__ import annotations

import datetime as dt
import itertools
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WALKERS_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# When run as a script (``python scripts/walkers/transformers_introspection.py``),
# Python prepends the script's directory to ``sys.path`` — that directory contains
# ``transformers.py`` (the sibling walker module), which shadows the third-party
# ``transformers`` package import. Drop the walkers dir so HF's ``transformers``
# resolves correctly. Module-style invocation
# (``python -m scripts.walkers.transformers_introspection``) avoids this trap
# but the verification command in the task brief uses script-style.
if str(_WALKERS_DIR) in sys.path:
    sys.path.remove(str(_WALKERS_DIR))

from scripts.walkers._base import RuleCandidate, WalkerSource  # noqa: E402

# ---------------------------------------------------------------------------
# Trigger classes (dormancy auto-enumeration — unchanged from prior impl)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DormancyTrigger:
    """One mode-gated dormancy trigger in HF's ``validate()``.

    HF 4.49-4.56 exposes three trigger classes: greedy (``do_sample=False``),
    single-beam (``num_beams=1``), and scalar-output
    (``return_dict_in_generate=False``). Each trigger ships an
    ``isolation_kwargs`` payload — values for the OTHER triggers that DON'T
    activate them — so the auto-enumerator can attribute a firing rule to
    exactly one trigger class even though the three categories overlap.
    """

    id_prefix: str
    trigger_field: str
    trigger_positive: Any
    trigger_negative: Any
    isolation_kwargs: dict[str, Any]
    rule_under_test_template: str


_GREEDY_TRIGGER = _DormancyTrigger(
    id_prefix="transformers_greedy_strips_",
    trigger_field="do_sample",
    trigger_positive=False,
    trigger_negative=True,
    isolation_kwargs={"num_beams": 4, "return_dict_in_generate": True},
    rule_under_test_template=(
        "GenerationConfig.validate() records dormant `{field}` when "
        "do_sample=False and `{field}` is set to a non-default value"
    ),
)


_BEAM_TRIGGER = _DormancyTrigger(
    id_prefix="transformers_single_beam_strips_",
    trigger_field="num_beams",
    trigger_positive=1,
    trigger_negative=4,
    isolation_kwargs={"do_sample": True, "return_dict_in_generate": True},
    rule_under_test_template=(
        "GenerationConfig.validate() records dormant `{field}` when "
        "num_beams=1 and `{field}` is set"
    ),
)


_RETURN_DICT_TRIGGER = _DormancyTrigger(
    id_prefix="transformers_no_return_dict_strips_",
    trigger_field="return_dict_in_generate",
    trigger_positive=False,
    trigger_negative=True,
    isolation_kwargs={"do_sample": True, "num_beams": 4},
    rule_under_test_template=(
        "GenerationConfig.validate() records dormant `{field}` when "
        "return_dict_in_generate=False and `{field}` is set"
    ),
)


TRIGGERS: tuple[_DormancyTrigger, ...] = (
    _GREEDY_TRIGGER,
    _BEAM_TRIGGER,
    _RETURN_DICT_TRIGGER,
)
"""Public: the full trigger-class partition. Tests parametrise over this."""


# Fields the dormancy enumerator skips. Categories: trigger fields themselves,
# class-instance fields whose probe value would need a specific type, meta
# / structural fields without dormancy semantics.
_DORMANCY_SKIP_FIELDS: frozenset[str] = frozenset(
    {
        "do_sample",
        "num_beams",
        "return_dict_in_generate",
        "compile_config",
        "watermarking_config",
        "generation_kwargs",
        "transformers_version",
    }
)


def _synthesise_probe_value(default: Any) -> Any | None:
    """Return a non-default value whose type matches ``default``.

    ``None`` when the type is too complex to probe mechanically.
    """
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 1
    if isinstance(default, float):
        if default in (0.0, 1.0):
            return 0.5
        return round(default + 0.1, 3)
    if default is None:
        return 0.5
    if isinstance(default, list):
        return ["probe"]
    if isinstance(default, str):
        return "__probe__"
    return None


# ---------------------------------------------------------------------------
# Library-message parsing
# ---------------------------------------------------------------------------


_ISSUE_LINE_RE = re.compile(r"^- `([^`]+)`: (.+)$")


def _parse_strict_raise(composed_message: str) -> dict[str, str]:
    """Split HF's ``validate(strict=True)`` raise into ``{field: message}``."""
    issues: dict[str, str] = {}
    for line in composed_message.splitlines():
        match = _ISSUE_LINE_RE.match(line.strip())
        if match:
            issues[match.group(1)] = match.group(2).strip()
    return issues


def _substitute_declared_value(message: str, probed_field: str | None, probe_value: Any) -> str:
    """Replace HF's ``\\`{field}\\` is set to \\`{value}\\``` with a ``{declared_value}`` slot."""
    if probed_field is None:
        return message
    pattern = f"`{probed_field}` is set to `{probe_value}`"
    if pattern in message:
        return message.replace(pattern, f"`{probed_field}` is set to `{{declared_value}}`")
    return message


_BUILTIN_TYPE_NAMES = frozenset(
    {
        "int",
        "str",
        "list",
        "tuple",
        "dict",
        "set",
        "float",
        "bool",
        "bytes",
        "frozenset",
        "complex",
    }
)


def _normalise_message_class(msg: str) -> str:
    """Collapse a library error message to its template form for grouping.

    Strips:
    - backtick-wrapped value tokens (``\\`num_beams\\``` literals stay; their
      contained values get genericised)
    - concrete numbers
    - trailing builtin-type-name tokens (``not int`` / ``not list`` /
      ``not str``) so that messages that differ ONLY in the offending
      value's type still hash to one class. This is what enables
      type-allowlist inference: HF's ``WatermarkingConfig() ... not int``
      and ``... not list`` are the same rule on the type-allowlist axis.

    Used to group probe rows by error class before predicate inference.
    """
    # Drop backtick-wrapped value tokens that vary per probe.
    msg = re.sub(r"`[^`]*`", "`X`", msg)
    # Drop bare numbers.
    msg = re.sub(r"-?\d+\.?\d*", "N", msg)
    # Genericise trailing builtin-type-name tokens after "not " so type-
    # allowlist messages (one per offending type) collapse into one class.
    type_alt = "|".join(sorted(_BUILTIN_TYPE_NAMES))
    msg = re.sub(rf"\b(not|got|is)\s+({type_alt})\b", r"\1 T", msg)
    # Collapse whitespace.
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg


# ---------------------------------------------------------------------------
# Dormancy auto-enumeration (unchanged from prior implementation)
# ---------------------------------------------------------------------------


def _enumerate_dormancy_candidates() -> list[tuple[_DormancyTrigger, str, Any, Any, str]]:
    """Probe every public ``GenerationConfig`` field under each trigger class.

    Returns ``(trigger, field, default, probe_value, per_field_message)``
    tuples for fields HF reports as dormant. Filters out: private fields,
    ``_DORMANCY_SKIP_FIELDS``, types with no synthesised probe, and fields
    whose construction trips an error under the trigger kwargs.

    Field iteration is byte-stable (sorted) so the corpus output is
    deterministic across runs.
    """
    import logging

    from transformers import GenerationConfig  # type: ignore

    hf_logger = logging.getLogger("transformers.generation.configuration_utils")
    prev_level = hf_logger.level
    hf_logger.setLevel(logging.ERROR)
    try:
        return _enumerate_dormancy_candidates_inner(GenerationConfig)
    finally:
        hf_logger.setLevel(prev_level)


def _enumerate_dormancy_candidates_inner(
    GenerationConfig: Any,
) -> list[tuple[_DormancyTrigger, str, Any, Any, str]]:
    baseline = GenerationConfig()
    discovered: list[tuple[_DormancyTrigger, str, Any, Any, str]] = []

    for field_name, default in sorted(vars(baseline).items()):
        if field_name.startswith("_"):
            continue
        if field_name in _DORMANCY_SKIP_FIELDS:
            continue
        probe = _synthesise_probe_value(default)
        if probe is None:
            continue

        for trigger in TRIGGERS:
            kwargs = {
                **trigger.isolation_kwargs,
                trigger.trigger_field: trigger.trigger_positive,
                field_name: probe,
            }
            try:
                gc = GenerationConfig(**kwargs)
            except Exception:
                continue
            try:
                gc.validate(strict=True)
            except ValueError as exc:
                issues = _parse_strict_raise(str(exc))
                if field_name in issues:
                    discovered.append((trigger, field_name, default, probe, issues[field_name]))

    return discovered


# ---------------------------------------------------------------------------
# Combinatorial cluster probing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cluster:
    """One cluster of related kwargs for combinatorial probing.

    ``values_per_field`` maps each kwarg in the cluster to the list of
    representative values to sweep. The probe runner takes the Cartesian
    product. Keep matrix size sensible — three fields x five values each
    = 125 trials, well under the 200-per-cluster cap.

    ``preconditions`` are kwargs held constant across every trial in the
    cluster (e.g. ``do_sample=True`` to suppress the greedy-mode dormancy
    layer). The pre-condition kwargs are NOT varied by the probe runner;
    they're applied to every trial.

    ``probe_class_match_prefix`` is the engine-config field-path prefix
    inserted when emitting rules from this cluster
    (``transformers.sampling`` for most clusters).
    """

    name: str
    values_per_field: dict[str, list[Any]]
    preconditions: dict[str, Any] = field(default_factory=dict)
    probe_class_match_prefix: str = "transformers.sampling"


def _watermarking_probe_values() -> list[Any]:
    """Probe values for ``watermarking_config`` — includes a real instance.

    Without a real ``WatermarkingConfig`` instance, the type-allowlist
    inference has no positive example to anchor against (every "wrong type"
    triggers a TypeError; every None just skips the field). The valid
    instance is the only OK present-value, so it must be in the matrix.
    """
    try:
        from transformers.generation.configuration_utils import WatermarkingConfig  # type: ignore

        valid_instance: Any = WatermarkingConfig()
    except Exception:
        valid_instance = None
    return [valid_instance, None, 42, "oops", [1, 2], (1, 2), 3.14]


def _compile_config_probe_values() -> list[Any]:
    """Probe values for ``compile_config`` — includes a real ``CompileConfig``."""
    try:
        from transformers.generation.configuration_utils import CompileConfig  # type: ignore

        valid_instance: Any = CompileConfig()
    except Exception:
        valid_instance = None
    return [valid_instance, None, "oops", 42, [1, 2]]


# Cluster definitions — small, representative value sets per kwarg. Adding a
# new cluster is a one-liner; tests can parametrise over CLUSTERS so probe
# coverage is auditable.
#
# Sizing rationale: keep each cluster's Cartesian product ≤ ~200 trials.
CLUSTERS: tuple[_Cluster, ...] = (
    _Cluster(
        name="beam_search",
        values_per_field={
            "num_beams": [1, 2, 4, 6],
            "num_beam_groups": [1, 2, 3],
            "diversity_penalty": [0.0, 0.5, -0.5],
            "early_stopping": [False, True, "never"],
        },
        # No preconditions: beam-search constraints fire at construction
        # regardless of do_sample.
    ),
    _Cluster(
        name="num_return_vs_beams",
        values_per_field={
            "num_beams": [1, 2, 4],
            "num_return_sequences": [1, 2, 4, 6],
            "do_sample": [False, True],
        },
    ),
    _Cluster(
        name="length_constraints",
        values_per_field={
            "min_new_tokens": [None, 5, 10],
            "max_new_tokens": [None, 4, 8, 16],
            "min_length": [0, 5],
            "max_length": [10, 20],
        },
    ),
    _Cluster(
        name="output_token_ids",
        values_per_field={
            "max_new_tokens": [-1, 0, 1, 16],
            "min_new_tokens": [-1, 0, 5],
            "pad_token_id": [-1, 0, 50256],
        },
    ),
    _Cluster(
        name="watermarking_type",
        values_per_field={
            # Probe values whose types vary across the type-allowlist axis.
            # First element is a real ``WatermarkingConfig`` instance — the
            # only present-value that doesn't error, anchoring the allowlist.
            "watermarking_config": _watermarking_probe_values(),
        },
    ),
    _Cluster(
        name="cache_choice",
        values_per_field={
            "cache_implementation": [
                None,
                "static",
                "dynamic",
                "nonsense",
                "another_bogus",
            ],
            "use_cache": [True, False],
        },
    ),
    _Cluster(
        name="early_stopping_type",
        values_per_field={
            "early_stopping": [False, True, "never", "sometimes", 0, 1.5],
            "num_beams": [1, 4],
        },
    ),
    _Cluster(
        name="compile_config_type",
        values_per_field={
            "compile_config": _compile_config_probe_values(),
        },
    ),
)


@dataclass
class _ProbeRow:
    """One trial row from the probe matrix."""

    kwargs: dict[str, Any]
    construct_error: str | None
    construct_message_class: str | None
    validate_minor_issues: dict[str, str]
    validate_strict_error: str | None
    state_after: dict[str, Any] | None  # vars(gc) after validate, for normalisation diff


def _run_cluster_probes(cluster: _Cluster) -> list[_ProbeRow]:
    """Run the Cartesian product for a cluster; return one row per trial.

    Each trial: instantiate ``GenerationConfig(**preconditions, **kwargs)``,
    capture construction-time errors; if construction succeeds, call
    ``validate(strict=True)`` and capture any composed raise.

    State diffs are not yet tabulated (would catch silent normalisation but
    silent normalisations don't surface as introspection rules in HF — they'd
    need vLLM-style detection, out of scope here).
    """
    import logging

    from transformers import GenerationConfig  # type: ignore

    # HF emits ``logger.warning`` lines on every dormant kwarg in __init__;
    # we capture validate(strict=True) raises programmatically and the warnings
    # spam the walker output. Silence them at probe-time.
    hf_logger = logging.getLogger("transformers.generation.configuration_utils")
    prev_level = hf_logger.level
    hf_logger.setLevel(logging.ERROR)

    field_names = list(cluster.values_per_field.keys())
    value_grids = [cluster.values_per_field[name] for name in field_names]
    rows: list[_ProbeRow] = []

    try:
        rows = _run_cartesian(cluster, field_names, value_grids, GenerationConfig)
    finally:
        hf_logger.setLevel(prev_level)
    return rows


def _run_cartesian(
    cluster: _Cluster,
    field_names: list[str],
    value_grids: list[list[Any]],
    GenerationConfig: Any,
) -> list[_ProbeRow]:
    """Inner Cartesian-product loop. Split out so ``finally`` restores the logger."""
    rows: list[_ProbeRow] = []
    for combo in itertools.product(*value_grids):
        kwargs = dict(zip(field_names, combo, strict=True))
        full_kwargs = {**cluster.preconditions, **kwargs}

        try:
            gc = GenerationConfig(**full_kwargs)
        except Exception as exc:
            rows.append(
                _ProbeRow(
                    kwargs=kwargs,
                    construct_error=str(exc),
                    construct_message_class=_normalise_message_class(str(exc)),
                    validate_minor_issues={},
                    validate_strict_error=None,
                    state_after=None,
                )
            )
            continue

        try:
            gc.validate(strict=True)
        except ValueError as exc:
            issues = _parse_strict_raise(str(exc))
            rows.append(
                _ProbeRow(
                    kwargs=kwargs,
                    construct_error=None,
                    construct_message_class=None,
                    validate_minor_issues=issues,
                    validate_strict_error=str(exc),
                    state_after=dict(vars(gc)),
                )
            )
        else:
            rows.append(
                _ProbeRow(
                    kwargs=kwargs,
                    construct_error=None,
                    construct_message_class=None,
                    validate_minor_issues={},
                    validate_strict_error=None,
                    state_after=dict(vars(gc)),
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Predicate inference
# ---------------------------------------------------------------------------


@dataclass
class _InferredRule:
    """Result of predicate inference over a probe-row group."""

    id_suffix: str
    rule_under_test: str
    severity: str  # "error" or "dormant"
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    message_template: str
    confidence: str  # "high" / "medium" / "low"
    method: str  # "construct" or "validate"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _row_field_value(row: _ProbeRow, field_name: str) -> Any:
    return row.kwargs.get(field_name)


def _split_error_rows(rows: list[_ProbeRow]) -> tuple[list[_ProbeRow], list[_ProbeRow]]:
    """Split rows into (errored, ok) — ok = no construct_error AND no validate_strict_error."""
    errored: list[_ProbeRow] = []
    ok: list[_ProbeRow] = []
    for row in rows:
        if row.construct_error or row.validate_strict_error:
            errored.append(row)
        else:
            ok.append(row)
    return errored, ok


def _group_construct_errors_by_class(rows: list[_ProbeRow]) -> dict[str, list[_ProbeRow]]:
    """Group construction-error rows by normalised message class."""
    groups: dict[str, list[_ProbeRow]] = defaultdict(list)
    for row in rows:
        if row.construct_message_class:
            groups[row.construct_message_class].append(row)
    return groups


def _group_validate_errors_by_field(rows: list[_ProbeRow]) -> dict[str, list[_ProbeRow]]:
    """Group validate-error rows by the FIRST minor-issues key.

    HF emits one ``minor_issues`` entry per offending field; rows are
    grouped by the field name so each affected field gets its own
    inference pass.
    """
    groups: dict[str, list[_ProbeRow]] = defaultdict(list)
    for row in rows:
        if row.validate_minor_issues:
            for key in sorted(row.validate_minor_issues.keys()):
                groups[key].append(row)
    return groups


def _check_predicate_explains_errors(
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
    predicate: Callable[[_ProbeRow], bool],
) -> bool:
    """Predicate is consistent with the error pattern.

    A predicate "explains" the errors when:
      - every row in ``error_rows`` satisfies the predicate
      - every "clean" row (no error of any kind) does NOT satisfy the predicate

    Rows that errored in a *different* error class are skipped from the
    consistency check — they contribute neither evidence for nor against
    the predicate, since the predicate is only claiming to explain THIS
    class's errors. Without this, divisibility predicates and diversity
    predicates contaminate each other inside the beam-search cluster.

    The predicate is applied only to rows where it's *defined* (the kwargs
    referenced by the predicate are present and the right types). Rows where
    the predicate is undefined are skipped.
    """
    error_set = {id(r) for r in error_rows}
    saw_error = False
    saw_ok = False
    for row in all_rows:
        is_this_class_error = id(row) in error_set
        is_any_error = bool(row.construct_error or row.validate_strict_error)
        is_other_class_error = is_any_error and not is_this_class_error
        if is_other_class_error:
            # Don't penalise the predicate for other-class errors; we make
            # no claim about them.
            continue
        try:
            holds = predicate(row)
        except (TypeError, KeyError):
            continue
        if holds and not is_this_class_error:
            return False  # predicate fires on a clean row — not explaining errors
        if not holds and is_this_class_error:
            return False  # predicate fails to fire on a known error row
        if holds and is_this_class_error:
            saw_error = True
        if not holds and not is_this_class_error:
            saw_ok = True
    # Need at least one positive and one negative example for the predicate
    # to be a meaningful rule (rule discrimination).
    return saw_error and saw_ok


def _infer_divisibility(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
    gate: Callable[[_ProbeRow], bool] | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    """Find a pair (a, b) where errors fire iff ``kwargs[a] % kwargs[b] != 0``.

    When ``gate`` is supplied, only rows satisfying the gate participate in
    the consistency check. This lets the inferrer find divisibility under
    a precondition (e.g. ``num_beam_groups > 1``).

    Returns ``(a, b, predicate_dict)`` or ``None``.
    """
    int_fields = [
        name
        for name, vals in cluster.values_per_field.items()
        if any(_is_int(v) and v > 0 for v in vals)
    ]
    if gate is None:
        gated_rows = all_rows
        gated_error_rows = error_rows
    else:
        gated_rows = [r for r in all_rows if _safe_gate(gate, r)]
        gated_error_rows = [r for r in error_rows if _safe_gate(gate, r)]
        if not gated_error_rows:
            return None
    for a, b in itertools.permutations(int_fields, 2):

        def pred(row: _ProbeRow, _a: str = a, _b: str = b) -> bool:
            va, vb = _row_field_value(row, _a), _row_field_value(row, _b)
            if not (_is_int(va) and _is_int(vb)):
                raise TypeError
            if vb <= 0:
                raise TypeError
            return va % vb != 0

        if _check_predicate_explains_errors(gated_rows, gated_error_rows, pred):
            return a, b, {"not_divisible_by": f"@{b}"}
    return None


_COMPARATORS: tuple[tuple[str, Callable[[Any, Any], bool]], ...] = (
    (">", lambda a, b: a > b),
    ("<", lambda a, b: a < b),
    (">=", lambda a, b: a >= b),
    ("<=", lambda a, b: a <= b),
)


def _infer_cross_field_comparison(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
    gate: Callable[[_ProbeRow], bool] | None = None,
) -> tuple[str, str, str, dict[str, Any]] | None:
    """Find a pair (a, b) and op where errors fire iff ``kwargs[a] op kwargs[b]``.

    When ``gate`` is supplied, the consistency check only considers rows
    where ``gate(row)`` holds. This lets the caller layer in a multi-field
    gate (``num_beams > 1``) so the cross-field comparison
    (``num_return_sequences > num_beams``) explains the residual error
    pattern after the gate is applied.

    Returns ``(a, b, op, predicate_dict)`` or ``None``.
    """
    int_fields = [
        name for name, vals in cluster.values_per_field.items() if any(_is_int(v) for v in vals)
    ]
    if gate is None:
        gated_rows = all_rows
        gated_error_rows = error_rows
    else:
        gated_rows = [r for r in all_rows if _safe_gate(gate, r)]
        gated_error_rows = [r for r in error_rows if _safe_gate(gate, r)]
        if not gated_error_rows:
            return None
    for a, b in itertools.permutations(int_fields, 2):
        for op_name, op_fn in _COMPARATORS:

            def pred(
                row: _ProbeRow, _a: str = a, _b: str = b, _op: Callable[..., bool] = op_fn
            ) -> bool:
                va, vb = _row_field_value(row, _a), _row_field_value(row, _b)
                if not (_is_int(va) and _is_int(vb)):
                    raise TypeError
                return _op(va, vb)

            if _check_predicate_explains_errors(gated_rows, gated_error_rows, pred):
                return a, b, op_name, {op_name: f"@{b}"}
    return None


def _safe_gate(gate: Callable[[_ProbeRow], bool], row: _ProbeRow) -> bool:
    """Apply ``gate(row)``; treat undefined rows (missing kwarg) as out of gate."""
    try:
        return gate(row)
    except (TypeError, KeyError):
        return False


def _infer_single_field_threshold(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
) -> tuple[str, str, Any, dict[str, Any]] | None:
    """Find a kwarg ``a`` and threshold ``v`` where errors fire iff ``kwargs[a] op v``.

    Tries representative thresholds: 0 (for ``< 0`` and ``<= 0``), the
    middle of the value grid for ``<`` / ``>``. Returns
    ``(field, op, value, predicate_dict)`` or ``None``.
    """
    for a, vals in cluster.values_per_field.items():
        # Numeric thresholds — try 0 first as the most common HF guard.
        thresholds: list[tuple[str, Any, Callable[[Any], bool]]] = []
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            thresholds.extend(
                [
                    (
                        "<",
                        0,
                        lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and x < 0,
                    ),
                    (
                        "<=",
                        0,
                        lambda x: (
                            isinstance(x, (int, float)) and not isinstance(x, bool) and x <= 0
                        ),
                    ),
                    (
                        ">",
                        0,
                        lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0,
                    ),
                ]
            )

        for op, threshold, op_fn in thresholds:

            def pred(row: _ProbeRow, _a: str = a, _fn: Callable[[Any], bool] = op_fn) -> bool:
                va = _row_field_value(row, _a)
                if va is None:
                    raise TypeError
                return _fn(va)

            if _check_predicate_explains_errors(all_rows, error_rows, pred):
                return a, op, threshold, {op: threshold}
    return None


def _infer_type_allowlist(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
) -> tuple[str, list[str], dict[str, Any]] | None:
    """Find a field where errors correlate with type-of-value being out of an allowlist.

    Returns ``(field, ok_type_names, predicate_dict)``.
    """
    error_ids = {id(r) for r in error_rows}
    for a, vals in cluster.values_per_field.items():
        type_names = sorted({type(v).__name__ for v in vals if v is not None})
        if len(type_names) < 2:
            continue
        ok_types: set[str] = set()
        err_types: set[str] = set()
        for row in all_rows:
            v = _row_field_value(row, a)
            if v is None:
                continue
            t = type(v).__name__
            is_this_class_error = id(row) in error_ids
            is_any_error = bool(row.construct_error or row.validate_strict_error)
            is_other_class_error = is_any_error and not is_this_class_error
            if is_other_class_error:
                # Skip — neutral for this class's predicate inference.
                continue
            if is_this_class_error:
                err_types.add(t)
            else:
                ok_types.add(t)
        if ok_types and err_types and not (ok_types & err_types):
            ok_list = sorted(ok_types)
            spec = {"present": True, "type_is_not": ok_list}

            def pred(row: _ProbeRow, _a: str = a, _ok: set[str] = ok_types) -> bool:
                v = _row_field_value(row, _a)
                if v is None:
                    raise TypeError
                return type(v).__name__ not in _ok

            if _check_predicate_explains_errors(all_rows, error_rows, pred):
                return a, ok_list, spec
    return None


def _infer_value_allowlist(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
) -> tuple[str, list[Any], dict[str, Any]] | None:
    """Find a field whose errors correlate with the value being out of an allowlist.

    Returns ``(field, ok_values, predicate_dict)`` where the predicate is
    ``{"not_in": ok_values}``. Distinct from ``type_allowlist`` because
    string-enum fields (``cache_implementation``) are typed-uniformly but
    fail on specific string values.
    """
    error_ids = {id(r) for r in error_rows}
    for a, vals in cluster.values_per_field.items():
        # Only consider hashable values.
        try:
            distinct_vals = {v for v in vals if v is not None}
        except TypeError:
            continue
        if len(distinct_vals) < 2:
            continue
        ok_vals: set[Any] = set()
        err_vals: set[Any] = set()
        for row in all_rows:
            v = _row_field_value(row, a)
            if v is None:
                continue
            try:
                hash(v)
            except TypeError:
                continue
            is_this_class_error = id(row) in error_ids
            is_any_error = bool(row.construct_error or row.validate_strict_error)
            is_other_class_error = is_any_error and not is_this_class_error
            if is_other_class_error:
                continue
            if is_this_class_error:
                err_vals.add(v)
            else:
                ok_vals.add(v)
        if ok_vals and err_vals and not (ok_vals & err_vals):
            ok_sorted = sorted(ok_vals, key=str)
            spec = {"present": True, "not_in": ok_sorted}

            def pred(row: _ProbeRow, _a: str = a, _ok: set[Any] = ok_vals) -> bool:
                v = _row_field_value(row, _a)
                if v is None:
                    raise TypeError
                return v not in _ok

            if _check_predicate_explains_errors(all_rows, error_rows, pred):
                return a, ok_sorted, spec
    return None


def _infer_multi_field_equality_gate(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Find a multi-field equality / range gate that explains errors.

    Two cases per field:
    1. **Equality** — all error rows share one value for the field.
    2. **Range comparator** — all error rows have ``field op V`` for the
       same threshold. Only emitted when the gate is non-trivial (the
       threshold actually discriminates ok rows from error rows). Useful
       when an error class is gated by ``num_beam_groups > 1`` even though
       beam-groups itself varies across the error rows (2, 3, ...).

    Returns ``(common_assignment, match_fields_addition)`` where
    ``match_fields_addition`` is a dict ready to merge into ``match_fields``
    (values may be predicate dicts like ``{">": 1}`` for range gates), or
    ``None`` if no useful gate exists.
    """
    if not error_rows:
        return None
    field_names = list(cluster.values_per_field.keys())
    common_assignment: dict[str, Any] = {}
    common_match: dict[str, Any] = {}
    ok_rows = [r for r in all_rows if not (r.construct_error or r.validate_strict_error)]
    for name in field_names:
        try:
            err_value_keys = {repr(_row_field_value(r, name)) for r in error_rows}
        except TypeError:
            continue
        if len(err_value_keys) == 1:
            v = _row_field_value(error_rows[0], name)
            common_assignment[name] = v
            common_match[name] = v
            continue
        # Range gate: try ``> V``, ``>= V``, ``< V``, ``<= V`` for V in
        # the union of cluster values for this field.
        err_vals = [_row_field_value(r, name) for r in error_rows]
        if all(_is_int(v) or isinstance(v, float) for v in err_vals if v is not None):
            for op_name, op_fn in _COMPARATORS:
                for threshold in cluster.values_per_field[name]:
                    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
                        continue
                    err_holds = all(v is not None and op_fn(v, threshold) for v in err_vals)
                    if not err_holds:
                        continue
                    # The gate must DISCRIMINATE: at least one ok row has
                    # value not satisfying the gate. Otherwise it's not
                    # information-bearing.
                    ok_breaks = any(
                        _row_field_value(r, name) is not None
                        and not op_fn(_row_field_value(r, name), threshold)
                        for r in ok_rows
                    )
                    if ok_breaks:
                        common_match[name] = {op_name: threshold}
                        break
                if name in common_match:
                    break
    if not common_match:
        return None
    match_addition = {f"{cluster.probe_class_match_prefix}.{k}": v for k, v in common_match.items()}
    return common_assignment, match_addition


# ---------------------------------------------------------------------------
# Cluster-level rule extraction
# ---------------------------------------------------------------------------


_SAFE_YAML_TYPES: tuple[type, ...] = (str, int, float, bool, type(None), list, tuple, dict)


def _is_yaml_safe(value: Any) -> bool:
    """True iff ``value`` round-trips through ``yaml.safe_dump`` / ``yaml.safe_load``.

    Pure-Python literals (str/int/float/bool/None/list/dict/tuple) are safe.
    Library class instances (``WatermarkingConfig``, ``CompileConfig``) are
    not — they need a synthesised replacement value (``None`` or a literal
    dict that round-trips into the same kwarg semantics).
    """
    if not isinstance(value, _SAFE_YAML_TYPES):
        return False
    if isinstance(value, dict):
        return all(_is_yaml_safe(k) and _is_yaml_safe(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_yaml_safe(v) for v in value)
    return True


def _yaml_safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Replace yaml-unsafe values (library class instances) with ``None``.

    The corpus needs to round-trip through ``yaml.safe_dump`` so the merger
    and the loader can re-parse it. Dropping the live instance to ``None``
    is lossy for the kwargs-positive / kwargs-negative pair, but the rule's
    *match predicate* still expresses the correct invariant — the consumer
    re-runs kwargs_positive to confirm the rule fires, kwargs_negative to
    confirm it doesn't, and library classes are typically representable as
    ``{}`` (empty kwargs) or ``None`` (absent) for the negative case.
    """
    return {k: (v if _is_yaml_safe(v) else None) for k, v in kwargs.items()}


def _pick_positive_negative_for_predicate(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
    affected_fields: list[str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Pick representative kwargs_positive / kwargs_negative pair for a rule.

    Positive: an error row's kwargs (subset to fields in the cluster).
    Negative: a non-error row's kwargs that differs MINIMALLY from the
    positive — to keep the test plausible.

    Yaml-unsafe values (library class instances) are replaced with ``None``;
    callers re-run the kwargs against the live library and library classes
    typically accept ``None`` as the absent-field state.

    Returns ``None`` if no clean pair exists.
    """
    if not error_rows:
        return None
    ok_rows = [r for r in all_rows if not (r.construct_error or r.validate_strict_error)]
    if not ok_rows:
        return None
    pos_row = error_rows[0]
    pos_kwargs = {k: v for k, v in pos_row.kwargs.items() if k in cluster.values_per_field}

    def overlap(row: _ProbeRow) -> int:
        # Compare via repr so unhashable values still count overlaps.
        return sum(
            1
            for k in affected_fields
            if k in row.kwargs
            and k in pos_kwargs
            and repr(row.kwargs.get(k)) == repr(pos_kwargs.get(k))
        )

    ok_rows_sorted = sorted(ok_rows, key=lambda r: -overlap(r))
    neg_row = ok_rows_sorted[0]
    neg_kwargs = {k: v for k, v in neg_row.kwargs.items() if k in cluster.values_per_field}
    return _yaml_safe_kwargs(pos_kwargs), _yaml_safe_kwargs(neg_kwargs)


def _representative_message(error_rows: list[_ProbeRow]) -> str:
    """Pick the canonical message string for an error group.

    For construct-error rows: the construct_error string.
    For validate-error rows: the per-field minor_issues body of the first row.
    """
    if not error_rows:
        return ""
    head = error_rows[0]
    if head.construct_error:
        return head.construct_error
    if head.validate_minor_issues:
        # Pick first key alphabetically for determinism.
        first_key = sorted(head.validate_minor_issues.keys())[0]
        return head.validate_minor_issues[first_key]
    return ""


def _extract_construct_error_rules(
    cluster: _Cluster,
    rows: list[_ProbeRow],
) -> list[_InferredRule]:
    """Run predicate inference for each construct-error message-class."""
    rules: list[_InferredRule] = []
    error_groups = _group_construct_errors_by_class(rows)
    for group_rows in error_groups.values():
        rules.extend(
            _infer_rules_for_group(cluster, rows, group_rows, severity="error", method="construct")
        )
    return rules


def _extract_validate_error_rules(
    cluster: _Cluster,
    rows: list[_ProbeRow],
) -> list[_InferredRule]:
    """Run predicate inference for each validate-error per-field group.

    These rows ARE dormancy, but cross-field gates (num_return_sequences > 1
    AND do_sample=False AND num_beams=1) often surface here.
    """
    rules: list[_InferredRule] = []
    field_groups = _group_validate_errors_by_field(rows)
    for affected_field, group_rows in field_groups.items():
        if affected_field in _DORMANCY_SKIP_FIELDS:
            # The dormancy enumerator handles these.
            continue
        rules.extend(
            _infer_rules_for_group(
                cluster,
                rows,
                group_rows,
                severity="dormant",
                method="validate",
                affected_field_hint=affected_field,
            )
        )
    return rules


def _infer_rules_for_group(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    error_rows: list[_ProbeRow],
    severity: str,
    method: str,
    affected_field_hint: str | None = None,
) -> list[_InferredRule]:
    """Try every predicate-inference shape; emit ALL fits with confidence ranking.

    Inference order is by specificity (most informative first):
    divisibility → cross-field comparison → type allowlist → value allowlist
    → single-field threshold. Results are emitted with confidence:

    - ``high``  — exactly one inference shape fits.
    - ``medium`` — two shapes fit (likely overlap, vendor CI prunes losers).
    - ``low``   — three or more shapes fit (ambiguous).

    Inference-shape order is also the rule-id suffix order, which keeps
    rule IDs stable across reruns.
    """
    if not error_rows:
        return []

    rep_msg = _representative_message(error_rows)
    gate_result = _infer_multi_field_equality_gate(cluster, all_rows, error_rows)
    common_assignment, common_match_addition = gate_result if gate_result else ({}, {})
    prefix = cluster.probe_class_match_prefix

    def _layer_gate(match: dict[str, Any], skip_fields: set[str]) -> dict[str, Any]:
        """Add gate predicates to ``match`` for fields not in ``skip_fields``."""
        for full_path, spec in common_match_addition.items():
            field_name = full_path.split(".")[-1]
            if field_name not in skip_fields:
                match[full_path] = spec
        return match

    # Build a callable form of the gate so cross-field inference can pre-filter
    # rows by it. The gate holds when every gate-field's predicate is satisfied.
    gate_fn: Callable[[_ProbeRow], bool] | None = None
    if common_match_addition:

        def gate_fn(row: _ProbeRow) -> bool:  # type: ignore[no-redef]
            for full_path, spec in common_match_addition.items():
                field_name = full_path.split(".")[-1]
                v = _row_field_value(row, field_name)
                if v is None:
                    return False
                if isinstance(spec, dict):
                    for op_name, threshold in spec.items():
                        op_fn = dict(_COMPARATORS).get(op_name)
                        if op_fn is None:
                            # Other operator types — give up gating for this row.
                            return False
                        if not op_fn(v, threshold):
                            return False
                else:
                    if v != spec:
                        return False
            return True

    inferences: list[tuple[str, str, dict[str, Any], list[str], str]] = []
    # tuple shape: (id_suffix, method_label, match_fields_dict, affected_field_names, declared_field)

    div_result = _infer_divisibility(cluster, all_rows, error_rows, gate=gate_fn)
    if div_result:
        a, b, spec = div_result
        match: dict[str, Any] = {f"{prefix}.{a}": spec}
        _layer_gate(match, {a, b})
        inferences.append((f"{a}_not_divisible_by_{b}", "divisibility", match, [a, b], a))

    cmp_result = _infer_cross_field_comparison(cluster, all_rows, error_rows, gate=gate_fn)
    if cmp_result:
        a, b, op, spec = cmp_result
        match = {f"{prefix}.{a}": spec}
        _layer_gate(match, {a, b})
        inferences.append((f"{a}_{_OP_NAMES[op]}_{b}", f"cmp{op}", match, [a, b], a))

    type_result = _infer_type_allowlist(cluster, all_rows, error_rows)
    if type_result:
        a, ok_list, spec = type_result
        match = {f"{prefix}.{a}": spec}
        _layer_gate(match, {a})
        type_label = "_or_".join(ok_list) if ok_list else "any"
        inferences.append((f"{a}_type_not_in_{type_label}", "type_allowlist", match, [a], a))

    val_result = _infer_value_allowlist(cluster, all_rows, error_rows)
    if val_result:
        a, ok_list, spec = val_result
        match = {f"{prefix}.{a}": spec}
        _layer_gate(match, {a})
        inferences.append((f"{a}_not_in_allowlist", "value_allowlist", match, [a], a))

    thr_result = _infer_single_field_threshold(cluster, all_rows, error_rows)
    if thr_result:
        a, op, threshold, spec = thr_result
        match = {f"{prefix}.{a}": spec}
        _layer_gate(match, {a})
        inferences.append(
            (
                f"{a}_{_OP_NAMES[op]}_{_value_label(threshold)}",
                f"threshold{op}",
                match,
                [a],
                a,
            )
        )

    # If nothing more specific fits but we have a multi-field gate, emit it
    # as a pure gate-equality / range rule. Lowest confidence — vendor CI
    # is the safety net.
    if not inferences and common_match_addition:
        if common_assignment:
            keys = sorted(common_assignment.keys())
            subject = keys[-1]
            id_suffix = "_and_".join(f"{k}_eq_{_value_label(common_assignment[k])}" for k in keys)
            inferences.append(
                (
                    id_suffix,
                    "common_assignment",
                    common_match_addition,
                    list(common_assignment.keys()),
                    subject,
                )
            )
        else:
            # Range-only gate (no equality). Build a descriptive id.
            keys = sorted({p.split(".")[-1] for p in common_match_addition})
            subject = keys[-1] if keys else "unknown"
            id_suffix = "_and_".join(f"{k}_gated" for k in keys)
            inferences.append((id_suffix, "range_gate", common_match_addition, keys, subject))

    if not inferences:
        return []

    # Confidence depends on count of plausible fits.
    if len(inferences) == 1:
        confidence = "high"
    elif len(inferences) == 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Build _InferredRule for each fit.
    out: list[_InferredRule] = []
    for id_suffix, _label, match_fields, affected_fields, declared_field in inferences:
        pn = _pick_positive_negative_for_predicate(cluster, all_rows, error_rows, affected_fields)
        if pn is None:
            continue
        kwargs_positive, kwargs_negative = pn
        # Inject preconditions into kwargs so the consumer can re-run them
        # against a clean GenerationConfig.
        for k, v in cluster.preconditions.items():
            kwargs_positive.setdefault(k, v)
            kwargs_negative.setdefault(k, v)

        # Substitute declared-value template placeholder where possible.
        probe_value = kwargs_positive.get(declared_field)
        message_template = _substitute_declared_value(rep_msg, declared_field, probe_value)

        rule_under_test = _rule_under_test_for(method, declared_field, id_suffix)

        out.append(
            _InferredRule(
                id_suffix=id_suffix,
                rule_under_test=rule_under_test,
                severity=severity,
                match_fields=match_fields,
                kwargs_positive=kwargs_positive,
                kwargs_negative=kwargs_negative,
                message_template=message_template,
                confidence=confidence,
                method=method,
            )
        )
    return out


_OP_NAMES: dict[str, str] = {
    ">": "exceeds",
    "<": "lt",
    ">=": "ge",
    "<=": "le",
    "==": "eq",
    "!=": "ne",
}


def _value_label(v: Any) -> str:
    """Short rule-id-safe label for a value (``-1`` → ``neg1``, ``0`` → ``zero``)."""
    if v is None:
        return "none"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        if v < 0:
            return f"neg{abs(v)}"
        if v == 0:
            return "zero"
        return str(v)
    if isinstance(v, float):
        return str(v).replace(".", "p").replace("-", "neg")
    return re.sub(r"\W+", "_", str(v))


def _rule_under_test_for(method: str, declared_field: str, id_suffix: str) -> str:
    """Compose a human-readable rule_under_test."""
    site = "GenerationConfig.__init__" if method == "construct" else "GenerationConfig.validate"
    return f"{site} flags `{declared_field}` ({id_suffix.replace('_', ' ')})"


# ---------------------------------------------------------------------------
# Validate-time self-triggered dormant probes (kept from prior implementation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DormantProbe:
    """A validate-time dormancy rule whose trigger is field-self-driven."""

    id: str
    rule_under_test: str
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    probed_field: str


_VALIDATE_DORMANT_PROBES: tuple[_DormantProbe, ...] = (
    _DormantProbe(
        id="transformers_negative_pad_token_id",
        rule_under_test="GenerationConfig.validate() records dormant pad_token_id < 0",
        match_fields={"transformers.sampling.pad_token_id": {"<": 0}},
        kwargs_positive={"pad_token_id": -1},
        kwargs_negative={"pad_token_id": 0},
        probed_field="pad_token_id",
    ),
)


# ---------------------------------------------------------------------------
# Rule-candidate factories
# ---------------------------------------------------------------------------


def _default_predicate(default: Any) -> dict[str, Any]:
    """Predicate that fires when the field is set to a non-default value."""
    if default is None:
        return {"present": True}
    return {"present": True, "not_equal": default}


@cache
def _read_source_lines(source_file: str) -> tuple[str, ...]:
    try:
        return tuple(Path(source_file).read_text().splitlines())
    except OSError:
        return ()


def _find_line(source_file: str, needle: str) -> int:
    for i, line in enumerate(_read_source_lines(source_file), start=1):
        if needle in line:
            return i
    return 0


def _make_dormancy_candidate(
    trigger: _DormancyTrigger,
    field_name: str,
    default: Any,
    probe: Any,
    library_message: str,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    template = _substitute_declared_value(library_message, field_name, probe)
    line = _find_line(abs_source_path, f"self.{field_name}")
    return RuleCandidate(
        id=f"{trigger.id_prefix}{field_name}",
        engine="transformers",
        library="transformers",
        rule_under_test=trigger.rule_under_test_template.format(field=field_name),
        severity="dormant",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate",
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields={
            f"transformers.sampling.{trigger.trigger_field}": trigger.trigger_positive,
            f"transformers.sampling.{field_name}": _default_predicate(default),
        },
        kwargs_positive={trigger.trigger_field: trigger.trigger_positive, field_name: probe},
        kwargs_negative={trigger.trigger_field: trigger.trigger_negative, field_name: probe},
        expected_outcome={
            "outcome": "dormant_announced",
            "emission_channel": "logger_warning_once",
            "normalised_fields": [],
        },
        message_template=template,
        references=[f"transformers.GenerationConfig.validate() (line ~{line})"],
        added_by="introspection",
        added_at=today,
    )


_OUTCOME_BY_SEVERITY: dict[str, dict[str, Any]] = {
    "error": {
        "outcome": "error",
        "emission_channel": "none",
        "normalised_fields": [],
    },
    "dormant": {
        "outcome": "dormant_announced",
        "emission_channel": "logger_warning_once",
        "normalised_fields": [],
    },
}


_REFERENCE_BY_METHOD: dict[str, str] = {
    "construct": "transformers.GenerationConfig — observed via construction-time ValueError",
    "validate": "transformers.GenerationConfig.validate() — observed via validate(strict=True)",
}


def _make_inferred_candidate(
    cluster: _Cluster,
    rule: _InferredRule,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    """Compose a ``RuleCandidate`` from a combinatorial-inference rule."""
    line = _find_line(abs_source_path, f"self.{_subject_field_from_id(rule.id_suffix)}")
    return RuleCandidate(
        id=f"transformers_{cluster.name}_{rule.id_suffix}",
        engine="transformers",
        library="transformers",
        rule_under_test=rule.rule_under_test,
        severity=rule.severity,
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate" if rule.method == "validate" else "__init__",
            line_at_scan=line,
            walker_confidence=rule.confidence,
        ),
        match_fields=rule.match_fields,
        kwargs_positive=rule.kwargs_positive,
        kwargs_negative=rule.kwargs_negative,
        expected_outcome=_OUTCOME_BY_SEVERITY[rule.severity],
        message_template=rule.message_template,
        references=[_REFERENCE_BY_METHOD[rule.method]],
        added_by="introspection",
        added_at=today,
    )


def _subject_field_from_id(id_suffix: str) -> str:
    """Best-effort: pull the leading kwarg name from the rule-id suffix."""
    return id_suffix.split("_")[0]


def _make_dormant_probe_candidate(
    probe: _DormantProbe,
    library_message: str,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    """Compose a ``RuleCandidate`` from a hardcoded validate-time dormant probe."""
    template = _substitute_declared_value(
        library_message, probe.probed_field, probe.kwargs_positive[probe.probed_field]
    )
    line = _find_line(abs_source_path, f"self.{probe.probed_field}")
    return RuleCandidate(
        id=probe.id,
        engine="transformers",
        library="transformers",
        rule_under_test=probe.rule_under_test,
        severity="dormant",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate",
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields=probe.match_fields,
        kwargs_positive=probe.kwargs_positive,
        kwargs_negative=probe.kwargs_negative,
        expected_outcome=_OUTCOME_BY_SEVERITY["dormant"],
        message_template=template,
        references=[
            "transformers.GenerationConfig.validate() — observed via validate(strict=True)"
        ],
        added_by="introspection",
        added_at=today,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class IntrospectionProbeDisappeared(RuntimeError):
    """A hardcoded validate-dormant probe stopped firing — library drift signal."""


def _walk_combinatorial(
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> tuple[list[RuleCandidate], int]:
    """Run all clusters and return ``(candidates, total_probe_rows)``."""
    candidates: list[RuleCandidate] = []
    total_rows = 0
    seen_ids: set[str] = set()
    for cluster in CLUSTERS:
        rows = _run_cluster_probes(cluster)
        total_rows += len(rows)
        construct_rules = _extract_construct_error_rules(cluster, rows)
        validate_rules = _extract_validate_error_rules(cluster, rows)
        for rule in (*construct_rules, *validate_rules):
            cand = _make_inferred_candidate(cluster, rule, abs_source_path, rel_source_path, today)
            if cand.id in seen_ids:
                # Same rule discovered in multiple clusters — keep the
                # first (deterministic by CLUSTERS iteration order).
                continue
            seen_ids.add(cand.id)
            candidates.append(cand)
    return candidates, total_rows


def walk_generation_config_rules(
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    """Return all introspection-derived rules for ``GenerationConfig``.

    Composes three sources in deterministic order:

    1. Mode-gated dormancy rules auto-enumerated per trigger class.
    2. Combinatorial cluster-probe rules (cross-field invariants).
    3. Validate-time self-triggered dormant rules.

    Raises :class:`IntrospectionProbeDisappeared` when a hardcoded probe
    silently stops firing — preserves the "silent coverage loss becomes a
    visible CI failure" contract for the small set of probes that aren't
    auto-discovered.
    """
    import logging

    from transformers import GenerationConfig  # type: ignore

    hf_logger = logging.getLogger("transformers.generation.configuration_utils")
    prev_level = hf_logger.level
    hf_logger.setLevel(logging.ERROR)

    candidates: list[RuleCandidate] = []

    # Path 1: dormancy auto-enumeration (preserved unchanged).
    for trigger, field_name, default, probe, lib_message in _enumerate_dormancy_candidates():
        candidates.append(
            _make_dormancy_candidate(
                trigger,
                field_name,
                default,
                probe,
                lib_message,
                abs_source_path,
                rel_source_path,
                today,
            )
        )

    # Path 2: combinatorial cluster probes.
    inferred, _row_count = _walk_combinatorial(abs_source_path, rel_source_path, today)
    candidates.extend(inferred)

    # Path 3: validate-time self-triggered dormant probes.
    for dprobe in _VALIDATE_DORMANT_PROBES:
        gc = GenerationConfig(**dprobe.kwargs_positive)
        try:
            gc.validate(strict=True)
        except ValueError as exc:
            issues = _parse_strict_raise(str(exc))
        else:
            raise IntrospectionProbeDisappeared(
                f"Validate-time dormant probe {dprobe.id!r} no longer fires."
            )
        if dprobe.probed_field not in issues:
            raise IntrospectionProbeDisappeared(
                f"Validate-time dormant probe {dprobe.id!r} fired without "
                f"mentioning {dprobe.probed_field!r}."
            )
        candidates.append(
            _make_dormant_probe_candidate(
                dprobe,
                issues[dprobe.probed_field],
                abs_source_path,
                rel_source_path,
                today,
            )
        )

    hf_logger.setLevel(prev_level)
    return candidates


def discover_dormancy_fields() -> dict[str, set[str]]:
    """Return ``{trigger.id_prefix: {field, ...}}`` — auto-discovered dormancy fields."""
    result: dict[str, set[str]] = {t.id_prefix: set() for t in TRIGGERS}
    for trigger, field_name, _default, _probe, _msg in _enumerate_dormancy_candidates():
        result[trigger.id_prefix].add(field_name)
    return result


# ---------------------------------------------------------------------------
# Standalone driver — writes the staging YAML for the corpus merger
# ---------------------------------------------------------------------------


def _relative_source_path(abs_path: str) -> str:
    """Strip host-specific prefixes so the corpus is reproducible across machines."""
    marker = "site-packages/"
    idx = abs_path.find(marker)
    if idx >= 0:
        return abs_path[idx + len(marker) :]
    return Path(abs_path).name


def _candidate_to_dict(c: RuleCandidate) -> dict[str, Any]:
    """Render a :class:`RuleCandidate` into the YAML corpus entry shape."""
    return {
        "id": c.id,
        "engine": c.engine,
        "library": c.library,
        "rule_under_test": c.rule_under_test,
        "severity": c.severity,
        "native_type": c.native_type,
        "walker_source": {
            "path": c.walker_source.path,
            "method": c.walker_source.method,
            "line_at_scan": c.walker_source.line_at_scan,
            "walker_confidence": c.walker_source.walker_confidence,
        },
        "match": {
            "engine": c.engine,
            "fields": c.match_fields,
        },
        "kwargs_positive": c.kwargs_positive,
        "kwargs_negative": c.kwargs_negative,
        "expected_outcome": c.expected_outcome,
        "message_template": c.message_template,
        "references": c.references,
        "added_by": c.added_by,
        "added_at": c.added_at,
    }


def _resolve_source_paths() -> tuple[str, str, str]:
    """Locate transformers' GenerationConfig source on disk.

    Returns ``(version, abs_path, rel_path)`` — the latter rooted at
    ``site-packages/`` for reproducibility.
    """
    import inspect

    import transformers  # type: ignore
    from transformers import GenerationConfig  # type: ignore

    abs_path = inspect.getsourcefile(GenerationConfig) or "<unknown>"
    rel_path = _relative_source_path(abs_path)
    return transformers.__version__, abs_path, rel_path


def main(argv: list[str] | None = None) -> int:
    """Run the introspection extractor end-to-end and write the staging YAML."""
    out_path = (
        Path(_PROJECT_ROOT)
        / "configs"
        / "validation_rules"
        / "_staging"
        / "transformers_introspection.yaml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    version, abs_source_path, rel_source_path = _resolve_source_paths()
    today = os.environ.get("LLENERGY_WALKER_FROZEN_AT", dt.date.today().isoformat())[:10]

    candidates = walk_generation_config_rules(
        abs_source_path=abs_source_path,
        rel_source_path=rel_source_path,
        today=today,
    )

    # Stable order: by walker_source.method, then by id.
    candidates_sorted = sorted(candidates, key=lambda c: (c.walker_source.method, c.id))

    walked_at = os.environ.get(
        "LLENERGY_WALKER_FROZEN_AT",
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )

    doc = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": version,
        "walked_at": walked_at,
        "extractor": "transformers_introspection",
        "rules": [_candidate_to_dict(c) for c in candidates_sorted],
    }
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100))

    print(
        f"Wrote {len(candidates_sorted)} introspection-derived rules to {out_path}",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "CLUSTERS",
    "TRIGGERS",
    "IntrospectionProbeDisappeared",
    "discover_dormancy_fields",
    "walk_generation_config_rules",
]


if __name__ == "__main__":
    raise SystemExit(main())
