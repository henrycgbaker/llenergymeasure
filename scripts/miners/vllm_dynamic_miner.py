"""Dynamic (introspection + probe) miner for the vLLM library.

Composes three sub-library lifts against vLLM's config classes plus a small
combinatorial probe loop for cross-field invariants the lifts cannot reach
on their own:

1. **Pydantic lift** over the 12 ``vllm.config.*`` pydantic-dataclasses
   (``CacheConfig``, ``ParallelConfig``, ``ModelConfig``, …). Each emits one
   rule per ``annotated-types`` constraint (``Gt``, ``Le``, ``MultipleOf``,
   …) and per ``Literal[...]`` allowlist annotation.
2. **msgspec lift** over ``vllm.SamplingParams``. Per the research doc,
   vLLM ships zero ``msgspec.Meta(...)`` annotations on SamplingParams, so
   this currently emits zero candidates — wired so that any future vLLM
   release adopting msgspec.Meta is captured automatically.
3. **dataclass lift** over ``vllm.engine.arg_utils.EngineArgs`` (a 175-field
   stdlib dataclass). Picks up every ``Literal[...]`` allowlist annotation
   on the user-facing ``LLM(...)`` kwargs.

Plus a runtime probe pass over ``SamplingParams`` cluster grids — small
Cartesian sweeps that surface cross-field rules (``min_tokens > max_tokens``,
``stop AND not detokenize``) the AST static miner already catches but
re-confirms via observed library behaviour.

CPU-safety
----------
``import vllm`` is CPU-safe on Ampere hardware (per the research doc),
emits stderr noise about libamd_smi but returns successfully. No CUDA
contexts are created during dynamic-miner execution; ``SamplingParams``
construction is pure Python validation. ``vllm.config.*`` classes
instantiate cleanly with no GPU; only ``EngineArgs(...).create_engine_config()``
reaches GPU-dependent code, which we deliberately avoid.

No LLM components anywhere
--------------------------
Per ``feedback_miner_pipeline_deterministic.md``: this miner is a pure
function of (vllm SHA, lift modules, probe seed). No model-as-author,
model-as-AST-parser, model-as-template-suggester. Hypothesis is not used
here; Cartesian probing only.

Output
------
Writes ``configs/validation_rules/_staging/vllm_dynamic_miner.yaml``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import itertools
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root on sys.path for direct script + module-style invocation.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Mirror the static miner's defence: strip the script directory from sys.path
# so a future ``vllm.py`` stub here couldn't shadow the installed library.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR).resolve()]
sys.path[:] = [p for p in sys.path if p != ""]

from scripts.miners._base import (  # noqa: E402
    MinerLandmarkMissingError,
    MinerSource,
    RuleCandidate,
    check_installed_version,
)
from scripts.miners._msgspec_lift import lift as _msgspec_lift  # noqa: E402
from scripts.miners._pydantic_lift import lift as _pydantic_lift  # noqa: E402
from scripts.miners.vllm_static_miner import (  # noqa: E402
    TESTED_AGAINST_VERSIONS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENGINE = "vllm"
LIBRARY = "vllm"

NS_SAMPLING = "vllm.sampling"
NS_ENGINE = "vllm.engine"


# ---------------------------------------------------------------------------
# Lift composition: vllm.config.* pydantic-dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LiftTarget:
    """One pydantic-dataclass to lift via ``_pydantic_lift.lift``.

    ``module_attr`` is the dotted path under ``vllm`` (e.g.
    ``"config.cache.CacheConfig"``). ``namespace`` is the field path-prefix
    used in emitted rules.
    """

    module_attr: str
    namespace: str

    @property
    def class_name(self) -> str:
        return self.module_attr.rsplit(".", 1)[-1]

    @property
    def module_path(self) -> str:
        return f"vllm.{self.module_attr.rsplit('.', 1)[0]}"


_PYDANTIC_LIFT_TARGETS: tuple[_LiftTarget, ...] = (
    _LiftTarget("config.cache.CacheConfig", NS_ENGINE),
    _LiftTarget("config.parallel.ParallelConfig", NS_ENGINE),
    _LiftTarget("config.parallel.EPLBConfig", NS_ENGINE),
    _LiftTarget("config.lora.LoRAConfig", NS_ENGINE),
    _LiftTarget("config.model.ModelConfig", NS_ENGINE),
    _LiftTarget("config.scheduler.SchedulerConfig", NS_ENGINE),
    _LiftTarget("config.multimodal.MultiModalConfig", NS_ENGINE),
    _LiftTarget("config.speculative.SpeculativeConfig", NS_ENGINE),
    _LiftTarget("config.compilation.CompilationConfig", NS_ENGINE),
    _LiftTarget("config.load.LoadConfig", NS_ENGINE),
    _LiftTarget("config.pooler.PoolerConfig", NS_ENGINE),
    _LiftTarget("config.structured_outputs.StructuredOutputsConfig", NS_ENGINE),
)


# ---------------------------------------------------------------------------
# Probe clusters (combinatorial)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cluster:
    """One probe cluster.

    ``probe_class_factory`` constructs the probe target — vLLM has multiple
    probe classes (SamplingParams, vllm.config.* dataclasses), so unlike
    transformers we pass a callable that takes a kwargs dict and returns
    a constructed instance (or raises).

    ``namespace`` is the field-path prefix used when emitting rules from
    this cluster.
    """

    name: str
    values_per_field: dict[str, list[Any]]
    namespace: str
    probe_class_factory: Callable[[dict[str, Any]], Any]
    native_type: str
    preconditions: dict[str, Any] = field(default_factory=dict)


def _sampling_factory(kw: dict[str, Any]) -> Any:
    from vllm import SamplingParams  # type: ignore

    return SamplingParams(**kw)


CLUSTERS: tuple[_Cluster, ...] = (
    _Cluster(
        name="sampling_token_counts",
        values_per_field={
            "max_tokens": [None, 1, 5, 10],
            "min_tokens": [-1, 0, 5, 15],
        },
        namespace=NS_SAMPLING,
        probe_class_factory=_sampling_factory,
        native_type="vllm.SamplingParams",
    ),
    _Cluster(
        name="sampling_basic_ranges",
        values_per_field={
            "temperature": [-0.5, 0.0, 0.5, 1.0, 2.0],
            "top_p": [0.0, 0.5, 1.0, 1.5],
        },
        namespace=NS_SAMPLING,
        probe_class_factory=_sampling_factory,
        native_type="vllm.SamplingParams",
    ),
    _Cluster(
        name="sampling_topk_minp",
        values_per_field={
            "top_k": [-2, -1, 0, 1, 50],
            "min_p": [-0.1, 0.0, 0.5, 1.0, 1.5],
        },
        namespace=NS_SAMPLING,
        probe_class_factory=_sampling_factory,
        native_type="vllm.SamplingParams",
    ),
    _Cluster(
        name="sampling_penalties",
        values_per_field={
            "presence_penalty": [-2.5, -2.0, 0.0, 2.0, 2.5],
            "frequency_penalty": [-2.5, -2.0, 0.0, 2.0, 2.5],
            "repetition_penalty": [-1.0, 0.0, 0.5, 1.0],
        },
        namespace=NS_SAMPLING,
        probe_class_factory=_sampling_factory,
        native_type="vllm.SamplingParams",
    ),
    _Cluster(
        name="sampling_n_greedy",
        values_per_field={
            "n": [1, 2, 3],
            "temperature": [0.0, 0.005, 0.5, 1.0],
        },
        namespace=NS_SAMPLING,
        probe_class_factory=_sampling_factory,
        native_type="vllm.SamplingParams",
    ),
    _Cluster(
        name="sampling_stop_pair",
        values_per_field={
            "stop": [None, [], ["done"]],
            "detokenize": [True, False],
        },
        namespace=NS_SAMPLING,
        probe_class_factory=_sampling_factory,
        native_type="vllm.SamplingParams",
    ),
)


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


@dataclass
class _ProbeRow:
    """One trial row from a Cartesian probe matrix."""

    kwargs: dict[str, Any]
    error_message: str | None
    error_type: str | None


def _silence_vllm_loggers() -> tuple[logging.Logger, int]:
    """Quiet the noisy vLLM init loggers during probing.

    Returns (logger, prev_level) so caller can restore — vLLM emits warnings
    on every greedy-clamp probe, which spams the miner's stderr.
    """
    lg = logging.getLogger("vllm")
    prev = lg.level
    lg.setLevel(logging.ERROR)
    return lg, prev


def _run_cluster(cluster: _Cluster) -> list[_ProbeRow]:
    """Run the Cartesian product for a cluster; return one row per trial."""
    field_names = list(cluster.values_per_field.keys())
    grids = [cluster.values_per_field[name] for name in field_names]

    lg, prev = _silence_vllm_loggers()
    rows: list[_ProbeRow] = []
    try:
        for combo in itertools.product(*grids):
            kwargs = dict(zip(field_names, combo, strict=True))
            full_kwargs = {**cluster.preconditions, **kwargs}
            try:
                cluster.probe_class_factory(full_kwargs)
            except Exception as exc:  # pragma: no cover — vLLM raises broadly
                rows.append(
                    _ProbeRow(
                        kwargs=kwargs,
                        error_message=str(exc),
                        error_type=type(exc).__name__,
                    )
                )
            else:
                rows.append(_ProbeRow(kwargs=kwargs, error_message=None, error_type=None))
    finally:
        lg.setLevel(prev)
    return rows


# ---------------------------------------------------------------------------
# Predicate inference (cross-field comparison + single-field threshold)
# ---------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _split(rows: list[_ProbeRow]) -> tuple[list[_ProbeRow], list[_ProbeRow]]:
    err = [r for r in rows if r.error_message is not None]
    ok = [r for r in rows if r.error_message is None]
    return err, ok


_COMPARATORS: tuple[tuple[str, Callable[[Any, Any], bool]], ...] = (
    (">", lambda a, b: a > b),
    ("<", lambda a, b: a < b),
    (">=", lambda a, b: a >= b),
    ("<=", lambda a, b: a <= b),
)


def _explains(
    all_rows: list[_ProbeRow],
    err_rows: list[_ProbeRow],
    pred: Callable[[_ProbeRow], bool],
) -> bool:
    """True iff ``pred`` fires on ``err_rows`` and on NO clean rows.

    Other-class errored rows (errors that aren't in ``err_rows``) are
    skipped from the consistency check — they're not evidence for or
    against this class's predicate. Without this skip, divisibility and
    range predicates inside one cluster would contaminate each other.
    """
    err_ids = {id(r) for r in err_rows}
    saw_err = False
    saw_ok = False
    for row in all_rows:
        is_this_class_error = id(row) in err_ids
        is_any_error = row.error_message is not None
        is_other_class_error = is_any_error and not is_this_class_error
        if is_other_class_error:
            continue
        try:
            holds = pred(row)
        except (TypeError, KeyError):
            continue
        if holds and not is_this_class_error:
            return False
        if not holds and is_this_class_error:
            return False
        if holds and is_this_class_error:
            saw_err = True
        if not holds and not is_this_class_error:
            saw_ok = True
    return saw_err and saw_ok


def _infer_cross_field_comparison(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    err_rows: list[_ProbeRow],
) -> tuple[str, str, str] | None:
    """Find a pair (a, b, op) where errors fire iff ``kwargs[a] op kwargs[b]``."""
    int_fields = [
        name for name, vals in cluster.values_per_field.items() if any(_is_int(v) for v in vals)
    ]
    for a, b in itertools.permutations(int_fields, 2):
        for op_name, op_fn in _COMPARATORS:

            def pred(
                row: _ProbeRow, _a: str = a, _b: str = b, _op: Callable[..., bool] = op_fn
            ) -> bool:
                va, vb = row.kwargs.get(_a), row.kwargs.get(_b)
                if not (_is_int(va) and _is_int(vb)):
                    raise TypeError
                return _op(va, vb)

            if _explains(all_rows, err_rows, pred):
                return a, b, op_name
    return None


def _infer_single_field_threshold(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    err_rows: list[_ProbeRow],
) -> tuple[str, str, Any] | None:
    """Find a kwarg ``a`` and threshold V where errors fire iff ``kwargs[a] op V``."""
    for a, vals in cluster.values_per_field.items():
        if not any(_is_num(v) for v in vals):
            continue
        for op, threshold, op_fn in [
            ("<", 0, lambda x: _is_num(x) and x < 0),
            ("<=", 0, lambda x: _is_num(x) and x <= 0),
            ("<", 1, lambda x: _is_num(x) and x < 1),
        ]:

            def pred(row: _ProbeRow, _a: str = a, _fn: Callable[[Any], bool] = op_fn) -> bool:
                va = row.kwargs.get(_a)
                if va is None:
                    raise TypeError
                return _fn(va)

            if _explains(all_rows, err_rows, pred):
                return a, op, threshold
    return None


# ---------------------------------------------------------------------------
# Cluster -> rule candidate
# ---------------------------------------------------------------------------


_OP_NAMES = {">": "exceeds", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


def _value_label(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "none"
    if isinstance(v, int):
        if v < 0:
            return f"neg{abs(v)}"
        if v == 0:
            return "zero"
        return str(v)
    if isinstance(v, float):
        s = str(v).replace(".", "p").replace("-", "neg")
        return s
    return "".join(ch for ch in str(v) if ch.isalnum() or ch == "_").lower()


def _normalise_message_class(msg: str) -> str:
    """Collapse a raised-error message to its template form for grouping.

    Strips backtick-wrapped value tokens and bare numbers so messages that
    differ only in the offending value still hash to one class. Mirrors the
    transformers dynamic miner's normalisation.
    """
    import re as _re

    msg = _re.sub(r"`[^`]*`", "`X`", msg or "")
    msg = _re.sub(r"-?\d+\.?\d*", "N", msg)
    msg = _re.sub(r"\s+", " ", msg).strip()
    return msg


def _group_errors_by_class(rows: list[_ProbeRow]) -> dict[str, list[_ProbeRow]]:
    """Group errored rows by normalised error-message class."""
    groups: dict[str, list[_ProbeRow]] = {}
    for r in rows:
        if r.error_message is None:
            continue
        key = _normalise_message_class(r.error_message)
        groups.setdefault(key, []).append(r)
    return groups


def _cluster_rules(
    cluster: _Cluster,
    rows: list[_ProbeRow],
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    """Run predicate inference per error-class and emit one RuleCandidate per fit.

    Splits errored rows by normalised message-class so each underlying rule
    in the cluster gets its own inference pass. Without grouping, a cluster
    that hits multiple validators (``temperature < 0`` AND ``top_p > 1``)
    fails to find any single predicate explaining all errors and emits zero
    rules — the symptom this grouping fixes.
    """
    err_rows, ok_rows = _split(rows)
    if not err_rows or not ok_rows:
        return []

    out: list[RuleCandidate] = []
    seen_ids: set[str] = set()
    for msg_class, group_rows in _group_errors_by_class(err_rows).items():
        del msg_class  # used only for grouping
        for rule in _infer_for_group(cluster, rows, group_rows, rel_source_path, today):
            if rule.id in seen_ids:
                continue
            seen_ids.add(rule.id)
            out.append(rule)
    return out


def _infer_for_group(
    cluster: _Cluster,
    all_rows: list[_ProbeRow],
    group_rows: list[_ProbeRow],
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    out: list[RuleCandidate] = []
    cmp_result = _infer_cross_field_comparison(cluster, all_rows, group_rows)
    if cmp_result is not None:
        a, b, op = cmp_result
        ok_rows = [r for r in all_rows if r.error_message is None]
        out.append(
            _make_cluster_rule(
                cluster=cluster,
                id_suffix=f"{a}_{_OP_NAMES[op]}_{b}",
                rule_under_test=f"{cluster.name}: error when {a} {op} {b}",
                match_fields={f"{cluster.namespace}.{a}": {op: f"@{b}"}},
                kwargs_positive=group_rows[0].kwargs,
                kwargs_negative=ok_rows[0].kwargs,
                message_template=group_rows[0].error_message,
                rel_source_path=rel_source_path,
                today=today,
            )
        )

    thr_result = _infer_single_field_threshold(cluster, all_rows, group_rows)
    if thr_result is not None:
        a, op, threshold = thr_result
        ok_rows = [r for r in all_rows if r.error_message is None]
        out.append(
            _make_cluster_rule(
                cluster=cluster,
                id_suffix=f"{a}_{_OP_NAMES[op]}_{_value_label(threshold)}",
                rule_under_test=f"{cluster.name}: error when {a} {op} {threshold}",
                match_fields={f"{cluster.namespace}.{a}": {op: threshold}},
                kwargs_positive=group_rows[0].kwargs,
                kwargs_negative=ok_rows[0].kwargs,
                message_template=group_rows[0].error_message,
                rel_source_path=rel_source_path,
                today=today,
            )
        )

    return out


def _make_cluster_rule(
    *,
    cluster: _Cluster,
    id_suffix: str,
    rule_under_test: str,
    match_fields: dict[str, Any],
    kwargs_positive: dict[str, Any],
    kwargs_negative: dict[str, Any],
    message_template: str | None,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    rid = f"vllm_{cluster.name}_{id_suffix}"
    return RuleCandidate(
        id=rid,
        engine=ENGINE,
        library=LIBRARY,
        rule_under_test=rule_under_test,
        severity="error",
        native_type=cluster.native_type,
        miner_source=MinerSource(
            path=rel_source_path,
            method="<probe>",
            line_at_scan=0,
        ),
        match_fields=match_fields,
        kwargs_positive=_yaml_safe_kwargs(kwargs_positive),
        kwargs_negative=_yaml_safe_kwargs(kwargs_negative),
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=message_template,
        references=[
            f"vllm.{cluster.native_type.rsplit('.', 1)[-1]} — observed via combinatorial probing"
        ],
        added_by="dynamic_miner",
        added_at=today,
    )


# Operator inversion: the lift modules emit ``match_fields`` shaped as the
# *constraint* the field must satisfy (e.g. ``{">": 0}`` for ``Gt(0)``). The
# corpus loader's convention is that ``match_fields`` encodes the rule's
# *firing condition* — i.e. the violation predicate. The transformers static
# miner emits violation-shape predicates because it walks ``if X: raise``
# AST. The lift modules emit constraint-shape predicates and we invert here
# at the per-engine driver level (per the brief: do not modify the lifts
# directly; engine-level adaptation is allowed).
_OP_INVERSE: dict[str, str] = {
    ">": "<=",
    ">=": "<",
    "<": ">=",
    "<=": ">",
    "==": "!=",
    "!=": "==",
    "in": "not_in",
    "not_in": "in",
}


def _invert_match_fields(match_fields: dict[str, Any]) -> dict[str, Any]:
    """Invert lift-emitted ``match_fields`` so they fire on the violation.

    For each path, replace operator keys with their loader-vocabulary
    inverse. Length operators (``min_len`` / ``max_len``) and the
    ``multiple_of`` predicate have no clean single-op inverse — we leave
    those entries as-is and let vendor-CI prune them; that's the lift's
    edge case to fix at the type-system level.
    """
    out: dict[str, Any] = {}
    for path, spec in match_fields.items():
        if not isinstance(spec, dict):
            # Bare value spec — equality. The inverted form is ``!=`` against
            # the same value, but bare-equality lift output is the Literal
            # path which already uses ``in`` not ``==``; leave alone.
            out[path] = spec
            continue
        new_spec: dict[str, Any] = {}
        for op, value in spec.items():
            inverse = _OP_INVERSE.get(op)
            if inverse is None:
                new_spec[op] = value
            else:
                new_spec[inverse] = value
        out[path] = new_spec
    return out


_SAFE_TYPES: tuple[type, ...] = (str, int, float, bool, type(None), list, tuple, dict)


def _is_yaml_safe(value: Any) -> bool:
    if not isinstance(value, _SAFE_TYPES):
        return False
    if isinstance(value, dict):
        return all(_is_yaml_safe(k) and _is_yaml_safe(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_yaml_safe(v) for v in value)
    return True


def _yaml_safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: (v if _is_yaml_safe(v) else None) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# Source-path helpers + landmark verification
# ---------------------------------------------------------------------------


def _site_packages_relative(abs_path: str) -> str:
    marker = "site-packages/"
    idx = abs_path.find(marker)
    if idx >= 0:
        return abs_path[idx + len(marker) :]
    return Path(abs_path).name


def _check_landmarks() -> str:
    """Import vLLM, verify SamplingParams + lift targets exist; return version.

    Mirrors the static miner's fail-loud contract: missing class -> raise.
    """
    try:
        import vllm  # type: ignore
    except ImportError as exc:
        raise MinerLandmarkMissingError("vllm.__init__", detail="vllm not importable") from exc

    try:
        from vllm import SamplingParams  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise MinerLandmarkMissingError(
            "vllm.SamplingParams", detail="symbol not importable"
        ) from exc

    try:
        from vllm.engine.arg_utils import EngineArgs  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise MinerLandmarkMissingError(
            "vllm.engine.arg_utils.EngineArgs", detail="symbol not importable"
        ) from exc

    for target in _PYDANTIC_LIFT_TARGETS:
        try:
            module = __import__(target.module_path, fromlist=[target.class_name])
        except ImportError as exc:
            raise MinerLandmarkMissingError(
                target.module_path, detail=f"module not importable: {exc}"
            ) from exc
        if not hasattr(module, target.class_name):
            raise MinerLandmarkMissingError(
                f"{target.module_path}.{target.class_name}",
                detail="class symbol missing",
            )

    return vllm.__version__


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def walk_vllm_dynamic() -> tuple[list[RuleCandidate], str]:
    """Return ``(candidates, vllm_version)`` for the dynamic mining pass.

    Composes (a) lift modules over SamplingParams + EngineArgs +
    vllm.config.* dataclasses, then (b) cluster probes. Emission order is
    deterministic across runs.
    """
    installed_version = _check_landmarks()
    check_installed_version("vllm", installed_version, TESTED_AGAINST_VERSIONS)

    today = dt.date.today().isoformat()
    candidates: list[RuleCandidate] = []

    # Lift 1: msgspec — SamplingParams. (``vllm.SamplingParams`` re-export
    # exists, so the lift's default native_type is dotted-importable.)
    from vllm import SamplingParams  # type: ignore

    sp_source = inspect.getsourcefile(SamplingParams) or "<unknown>"
    sp_rel = _site_packages_relative(sp_source)
    sp_lifted = _msgspec_lift(
        SamplingParams,
        namespace=NS_SAMPLING,
        today=today,
        source_path=sp_rel,
    )
    for cand in sp_lifted:
        # The lift synthesises its own message templates from
        # ``annotated-types`` operator names; vLLM's runtime messages come
        # from msgspec / pydantic and don't match. Drop the synthesised
        # template so the vendor-CI gate skips the substring check (the
        # rule's ``kwargs_positive`` raises and ``kwargs_negative`` doesn't,
        # which is sufficient signal for a lift-derived rule). Per
        # ``feedback_corpus_is_measurement_not_authoring``, the corpus
        # records what fires, not how the library worded the message.
        cand.message_template = None
        cand.match_fields = _invert_match_fields(cand.match_fields)
    candidates.extend(sp_lifted)

    # Lift 2: pydantic — vllm.config.* family. The lift sets
    # ``native_type=f"{library}.{type_name}"`` from ``cls.__module__.split('.', 1)[0]``,
    # which yields ``"vllm.CacheConfig"`` — but the actual import path is
    # ``vllm.config.cache.CacheConfig`` (re-exported as ``vllm.config.CacheConfig``).
    # Fixing this in ``_pydantic_lift`` would change every other engine's
    # behaviour (the lift is library-agnostic). Instead, rewrite the
    # native_type per-engine here so vendor-CI's ``_construct_generic`` can
    # reach the class via dotted import. ``native_type`` is the only field
    # that needs adjusting; ``library`` (used for diagnostics) stays as
    # ``"vllm"``.
    for target in _PYDANTIC_LIFT_TARGETS:
        module = __import__(target.module_path, fromlist=[target.class_name])
        cls = getattr(module, target.class_name)
        cls_source = inspect.getsourcefile(module) or "<unknown>"
        cls_rel = _site_packages_relative(cls_source)
        lifted = _pydantic_lift(
            cls,
            namespace=target.namespace,
            today=today,
            source_path=cls_rel,
        )
        # Use the ``vllm.config.<Name>`` re-export form (rather than the
        # deeper ``vllm.config.parallel.ParallelConfig`` actual location) —
        # vLLM re-exports every config class from ``vllm.config`` and this
        # is the form the static miner uses, so cross-validation /
        # fingerprint readability stays uniform.
        canonical_native = f"vllm.config.{target.class_name}"
        for cand in lifted:
            cand.native_type = canonical_native
            # Synthesised templates don't match Pydantic's runtime errors;
            # see the SamplingParams lift loop above for rationale.
            cand.message_template = None
            cand.match_fields = _invert_match_fields(cand.match_fields)
        candidates.extend(lifted)

    # Lift 3: stdlib dataclass — DELIBERATELY OMITTED for EngineArgs.
    #
    # ``EngineArgs`` is a 175-field stdlib dataclass with rich
    # ``Literal[...]`` annotations. Calling ``_dataclass_lift`` over it
    # produces ~23 candidate rules — all of which fail vendor-CI because
    # stdlib dataclass does NOT enforce Literal types at construction:
    # ``EngineArgs(runner="bogus")`` succeeds and only triggers a runtime
    # warning. Real validation lives on the depth-1 ``vllm.config.*``
    # pydantic-dataclasses (covered by the pydantic lift above) and on
    # the AST-walked validators (covered by the static miner).
    #
    # Recall-first would say "emit and let vendor-CI prune", but that
    # bloats the quarantine file with unenforceable rules. This is the
    # type-system mismatch the design's "skip dynamic-TRT-LLM" decision
    # also encountered: not every lift × class produces real rules.
    # Symmetric resolution is to omit the lift call when the class isn't
    # the actual validation site. Per-engine adaptation, no lift-module
    # change.

    # Probe path: SamplingParams cluster runs.
    for cluster in CLUSTERS:
        rows = _run_cluster(cluster)
        candidates.extend(_cluster_rules(cluster, rows, sp_rel, today))

    return candidates, installed_version


def _candidate_to_dict(c: RuleCandidate) -> dict[str, Any]:
    return {
        "id": c.id,
        "engine": c.engine,
        "library": c.library,
        "rule_under_test": c.rule_under_test,
        "severity": c.severity,
        "native_type": c.native_type,
        "miner_source": {
            "path": c.miner_source.path,
            "method": c.miner_source.method,
            "line_at_scan": c.miner_source.line_at_scan,
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


def emit_yaml(candidates: list[RuleCandidate], engine_version: str) -> str:
    import yaml

    sorted_candidates = sorted(candidates, key=lambda c: (c.miner_source.method, c.id))
    doc = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": engine_version,
        "walker_pinned_range": str(TESTED_AGAINST_VERSIONS),
        "mined_at": dt.date.today().isoformat(),
        "rules": [_candidate_to_dict(c) for c in sorted_candidates],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/validation_rules/_staging/vllm_dynamic_miner.yaml"),
        help="Where to write the staging YAML.",
    )
    args = parser.parse_args(argv)

    candidates, version = walk_vllm_dynamic()
    text = emit_yaml(candidates, version)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} vLLM dynamic-miner rule candidates to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
