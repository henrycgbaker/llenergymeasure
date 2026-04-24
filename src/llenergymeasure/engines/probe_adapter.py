"""Adapter composing hardware errors and dormancy observations into a ``ConfigProbe``."""

from __future__ import annotations

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.probe import ConfigProbe, DormantField


def build_config_probe(config: ExperimentConfig) -> ConfigProbe:
    """Assemble a :class:`ConfigProbe` for *config*.

    Composes two disjoint inputs:
      - hardware errors from :meth:`EnginePlugin.check_hardware`
      - dormancy observations from :meth:`ExperimentConfig._apply_vendored_rules`
        (via the ``_dormant_observations`` attribute, a ``dict[rule_id, DormantField]``)

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
        observed_engine_params={},
        observed_sampling_params={},
        dormant_fields=dormant_fields,
        errors=errors,
        warnings=[],
    )
