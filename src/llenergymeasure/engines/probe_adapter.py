"""Compose a :class:`ConfigProbe` from the vendored-rules validator and per-engine hardware checks.

The probe has two inputs with disjoint scopes:

- Dormancy (from :meth:`ExperimentConfig._apply_vendored_rules` via the
  ``_dormant_observations`` attribute on the resolved config).
- Hardware errors (from :meth:`EnginePlugin.check_hardware`).

M1 returns empty ``effective_engine_params`` / ``effective_sampling_params``;
those fields light up when the M2 introspection walker supplies the effective
kwargs surface.
"""

from __future__ import annotations

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.probe import ConfigProbe, DormantField


def build_config_probe(config: ExperimentConfig) -> ConfigProbe:
    """Assemble a :class:`ConfigProbe` for *config*.

    Never raises: hardware-check exceptions are trapped into ``errors``.
    """
    # Lazy import to avoid package-init ordering with engines/__init__.py.
    from llenergymeasure.engines import get_engine

    engine = get_engine(config.engine)

    errors: list[str] = []
    try:
        errors.extend(engine.check_hardware(config))
    except Exception as exc:
        errors.append(f"check_hardware raised {type(exc).__name__}: {exc}")

    dormant_fields: dict[str, DormantField] = getattr(config, "_dormant_observations", None) or {}

    return ConfigProbe(
        effective_engine_params={},
        effective_sampling_params={},
        dormant_fields=dormant_fields,
        errors=errors,
        warnings=[],
    )
