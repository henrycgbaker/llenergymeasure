"""Canonical serialisation and SHA-256 hashing for H1 and H3.

Two hashes with orthogonal sources but identical shape:

- **H1** (library-resolution mechanism-output): hashed at sweep-expansion time over the
  canonical form produced by :mod:`llenergymeasure.study.library_resolution`.
- **H3** (library-observed): hashed at sidecar-write time over the native
  types the engine constructed during inference (via
  :func:`llenergymeasure.engines._helpers.extract_observed_params`).

Sharing the schema and serialisation is what makes the observed-config-hash collision invariant
meaningful (see ``.product/designs/config-deduplication-dormancy/sweep-dedup.md``
§4.1): after resolved-config-hash dedup, any observed-config-hash duplicate is a proven library-resolution mechanism gap.

The normalisation rules below are locked by sweep-dedup.md §9.Q3 — over-
normalising would hide library-enforced semantics (e.g. ``None`` vs missing in
vLLM), so the rules are strict.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from llenergymeasure.config.models import ExperimentConfig

_FLOAT_SIG_DIGITS = 12
"""Float rounding precision (significant digits) for hash stability.

Upstream float arithmetic can produce bit-level jitter in the last 1-2
digits that does not reflect an actual configuration difference. Rounding at
12 significant digits removes that jitter without compressing any two values
a researcher would write differently.
"""


# ---------------------------------------------------------------------------
# Canonical serialisation (shared by H1 and H3)
# ---------------------------------------------------------------------------


def _normalise(value: Any) -> Any:
    """Normalise a value for deterministic JSON serialisation.

    Applies the locked rules from sweep-dedup.md §9.Q3:

    - ``NaN`` → string ``"NaN"`` (NaN != NaN breaks dict hashing otherwise)
    - float → rounded to 12 significant digits (stable across minor
      arithmetic jitter)
    - tuple → list (incidental immutability choice, not semantic)
    - bool is preserved as bool (not folded into int even though
      ``True == 1`` in Python)
    - dict → dict with recursively normalised values (key sorting happens
      at ``json.dumps(sort_keys=True)`` time)
    - None and missing keys stay distinguishable (by never inserting a
      sentinel for missing)
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            # Preserve infinity as a string literal for hash stability
            return "Infinity" if value > 0 else "-Infinity"
        if value == 0.0:
            return 0.0
        # Round to sig-figs rather than decimal places
        mag = math.floor(math.log10(abs(value)))
        factor = 10 ** (_FLOAT_SIG_DIGITS - 1 - mag)
        return round(value * factor) / factor
    if isinstance(value, (set, frozenset)):
        # Sort for determinism; normalise elements recursively.
        return [_normalise(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    return value


def canonical_serialise(obj: Any) -> bytes:
    """Serialise ``obj`` to canonical-JSON bytes, ready for hashing.

    Applies :func:`_normalise` then ``json.dumps(sort_keys=True)``. Uses a
    separators tuple with no whitespace for compactness and stability
    across Python versions.
    """
    normalised = _normalise(obj)
    return json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        default=str,  # Pydantic enums and dates
        ensure_ascii=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Hashed-field schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigHashView:
    """Fixed-schema view of the fields hashed into resolved or observed config hash.

    Per sweep-dedup.md §2.4, the hashed-field set is:

    - ``task`` — model, prompt source, batch shape
    - ``observed_engine_params`` — engine state (library-resolution mechanism output for H1,
      live library observation for H3)
    - ``observed_sampling_params`` — sampling state (same sources as above)
    - ``lora`` / ``passthrough_kwargs`` — user-attached overrides

    Excluded: ``MeasurementConfig`` (observation dials), ``ExecutionConfig``
    (runner/parallelism), ``experiment_id``.
    """

    engine: str
    task: dict[str, Any]
    observed_engine_params: dict[str, Any] = field(default_factory=dict)
    observed_sampling_params: dict[str, Any] = field(default_factory=dict)
    lora: dict[str, Any] | None = None
    passthrough_kwargs: dict[str, Any] = field(default_factory=dict)


def hash_config(view: ConfigHashView) -> str:
    """Return SHA-256 hex digest of ``view`` via :func:`canonical_serialise`.

    Both H1 and H3 route through this function; they differ only in how
    the :class:`ConfigHashView` is populated.
    """
    payload = asdict(view)
    return hashlib.sha256(canonical_serialise(payload)).hexdigest()


# ---------------------------------------------------------------------------
# H1 view construction — from library-resolution mechanism output
# ---------------------------------------------------------------------------


def build_resolved_view(config: ExperimentConfig) -> ConfigHashView:
    """Project a (post-library-resolution) ``ExperimentConfig`` into an H1 view.

    Reads the active engine section's full post-normalisation state; the
    library-resolution mechanism has already applied dormant rules to fixpoint before this
    runs. Callers pass the resolved config, not the declared one — resolved_config_hash is
    meaningless on a pre-resolved config.

    Engine-specific sub-models carry a ``sampling`` attribute; it is lifted
    into its own dict so H1/H3 ordering separates "how the engine constructs"
    from "what it generates with".
    """
    engine_name = config.engine.value if hasattr(config.engine, "value") else str(config.engine)
    section: Any = getattr(config, engine_name, None)
    dump: dict[str, Any] = section.model_dump(mode="python") if section is not None else {}
    sampling = dump.pop("sampling", None) or {}

    return ConfigHashView(
        engine=engine_name,
        task=config.task.model_dump(mode="python"),
        observed_engine_params=dump,
        observed_sampling_params=sampling,
        lora=None,  # No LoRA field yet on ExperimentConfig — reserved for future.
        passthrough_kwargs=dict(config.passthrough_kwargs or {}),
    )


# ---------------------------------------------------------------------------
# H3 view construction — from library-observed effective params
# ---------------------------------------------------------------------------


def build_observed_view(
    *,
    engine: str,
    task: dict[str, Any],
    observed_engine_params: dict[str, Any],
    observed_sampling_params: dict[str, Any],
    lora: dict[str, Any] | None = None,
    passthrough_kwargs: dict[str, Any] | None = None,
) -> ConfigHashView:
    """Assemble an H3 view from per-engine ``extract_observed_params`` output.

    Callers live in the harness/sidecar path — they read ``task`` from the
    same config that ran and pair it with the native-object dumps the engine
    returned.
    """
    return ConfigHashView(
        engine=engine,
        task=task,
        observed_engine_params=dict(observed_engine_params),
        observed_sampling_params=dict(observed_sampling_params),
        lora=lora,
        passthrough_kwargs=dict(passthrough_kwargs or {}),
    )
