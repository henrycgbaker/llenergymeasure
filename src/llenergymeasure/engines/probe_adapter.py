"""Build :class:`ConfigProbe` results from a validated :class:`ExperimentConfig`.

Replaces per-engine ``probe_config()`` methods (removed in phase 50.2c). The
adapter combines two independent observations into the :class:`ConfigProbe`
result container:

- **Framework-rule outputs** — recorded by the generic
  :meth:`ExperimentConfig._apply_vendored_rules` validator. Error-severity
  rules raise during config construction; warn-severity rules emit
  :class:`ConfigValidationWarning`; dormant rules append entries to the
  private ``_dormant_observations`` attribute this adapter reads.
- **Hardware compat** — produced by :meth:`EnginePlugin.check_hardware`.
  Unlike the vendored rules (which are declarative config x library
  predicates), hardware checks consult the live host (NVML, SM capability)
  so they must run at preflight, not at config-load.

The adapter is intentionally thin: it owns no rule matching logic and
holds no state. Effective-params population is 50.3a's canonicaliser job;
this phase returns empty dicts for those fields.
"""

from __future__ import annotations

from llenergymeasure import engines as _engines_pkg
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.protocol import ConfigProbe, DormantField


def build_config_probe(config: ExperimentConfig) -> ConfigProbe:
    """Assemble a :class:`ConfigProbe` for *config*.

    Reads dormant observations populated during :class:`ExperimentConfig`
    construction (via the generic vendored-rules validator) and runs the
    per-engine :meth:`EnginePlugin.check_hardware` to surface
    hardware-compat errors.

    Contract: never raises. Missing engine plugins (e.g. vLLM not installed
    on the host) produce a :class:`ConfigProbe` without hardware errors
    rather than propagating :class:`EngineError`. Callers that want to
    distinguish "no plugin available" from "plugin says hardware ok" should
    check :func:`llenergymeasure.engines.get_engine` directly.

    Args:
        config: A constructed :class:`ExperimentConfig`. If the config was
            built with :meth:`ExperimentConfig.model_construct` (skipping
            validators), ``_dormant_observations`` may not be populated; in
            that case the returned probe lists no dormant fields.

    Returns:
        A :class:`ConfigProbe` with:
          - ``effective_engine_params`` / ``effective_sampling_params`` —
            empty dicts (populated by the canonicaliser in 50.3a).
          - ``dormant_fields`` — dormancy observations from the vendored
            rules, keyed by ``<engine>.<rule_id>``.
          - ``errors`` — hardware-check output from the engine plugin.
          - ``warnings`` — empty list; warn-severity vendored rules emit
            ``ConfigValidationWarning`` through :func:`warnings.warn` at
            config-load time, which callers capture via
            :func:`warnings.catch_warnings` if needed.
    """
    engine_name = config.engine.value if hasattr(config.engine, "value") else str(config.engine)

    dormant_observations: list[DormantField] = list(
        getattr(config, "_dormant_observations", []) or []
    )
    dormant_fields: dict[str, DormantField] = {
        f"{engine_name}.dormant[{idx}]": obs for idx, obs in enumerate(dormant_observations)
    }

    errors = _run_check_hardware(engine_name, config)

    return ConfigProbe(
        effective_engine_params={},
        effective_sampling_params={},
        dormant_fields=dormant_fields,
        errors=errors,
        warnings=[],
    )


def _run_check_hardware(engine_name: str, config: ExperimentConfig) -> list[str]:
    """Dispatch to the engine plugin's :meth:`check_hardware`; quiet on missing plugin.

    Resolves ``get_engine`` via the module reference at call time so tests
    (and other runtime consumers) can ``monkeypatch.setattr`` the attribute
    on ``llenergymeasure.engines`` and have the adapter pick the replacement
    up on the next invocation.
    """
    try:
        plugin = _engines_pkg.get_engine(engine_name)
    except Exception:
        # Engine library not installed on the host — hardware check isn't
        # available. Callers distinguish "engine missing" via the standard
        # preflight install-check; this adapter stays silent.
        return []
    try:
        result = plugin.check_hardware(config)
    except Exception as exc:  # pragma: no cover — check_hardware must not raise
        return [f"check_hardware raised unexpectedly: {type(exc).__name__}: {exc}"]
    if not isinstance(result, list):
        return [f"check_hardware returned non-list {type(result).__name__}; treating as empty."]
    return [str(item) for item in result]


__all__ = ["build_config_probe"]
