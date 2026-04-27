"""Load vendored engine schemas discovered by ``scripts/discover_engine_schemas.py``.

The vendored JSON files in ``discovered_schemas/`` are the canonical SSOT for
"what parameters CAN be configured per engine". They are produced by running
introspection inside each engine's Docker image and committed to the repo.

This loader reads them via ``importlib.resources`` so it works in both editable
installs and installed wheels. Repeated loads are cached per-engine. Major
version mismatches (envelope schema breaking changes) raise
``UnsupportedSchemaVersionError``.

Downstream consumers (doc generators, field-name alignment, CI drift guards)
should load through this module rather than reading the JSON files directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from typing import Any

from llenergymeasure.config.ssot import Engine

SUPPORTED_MAJOR_VERSION = 1

# Engines known to ship a vendored schema. Test-patchable module attribute
# (tests monkeypatch this to inject fake engines); derived from Engine SSOT.
_KNOWN_ENGINES: tuple[str, ...] = tuple(Engine)

_PACKAGE = "llenergymeasure.config.discovered_schemas"


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a vendored schema's major version doesn't match this loader."""


@dataclass(frozen=True)
class DiscoveryLimitation:
    """A single limitation recorded by the discovery script.

    Fields that discovery could not recover (e.g. HF's None-default fields with
    no type annotations, or kwargs that don't appear in an inspected signature)
    are surfaced here rather than silently dropped.
    """

    section: str
    fields: list[str]
    reason: str


@dataclass(frozen=True)
class DiscoveredSchema:
    """A parsed vendored engine schema.

    ``engine_params`` and ``sampling_params`` are kept as raw dicts rather than
    a typed FieldDescriptor because per-engine richness varies: TRT-LLM fields
    carry ``description`` and ``deprecated`` from its Pydantic schema, while
    vLLM and Transformers fields only have ``type`` and ``default``. Consumers
    that need uniform shape should adapt at read time.
    """

    schema_version: str
    engine: str
    engine_version: str
    engine_commit_sha: str | None
    image_ref: str
    base_image_ref: str
    discovered_at: datetime
    discovery_method: str
    discovery_limitations: list[DiscoveryLimitation] = field(default_factory=list)
    engine_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    sampling_params: dict[str, dict[str, Any]] = field(default_factory=dict)


class SchemaLoader:
    """Load and cache vendored engine schemas.

    Uses a per-instance dict cache (rather than ``functools.lru_cache``) so
    multiple SchemaLoader instances don't share state — convenient for tests
    and for isolating reloads after a schema refresh.
    """

    def __init__(self) -> None:
        self._cache: dict[str, DiscoveredSchema] = {}

    def load_schema(self, engine: str) -> DiscoveredSchema:
        """Load the vendored schema for ``engine``.

        Raises:
            ValueError: ``engine`` is not a known engine name.
            FileNotFoundError: No vendored JSON exists for ``engine``.
            UnsupportedSchemaVersionError: Vendored schema major version
                doesn't match ``SUPPORTED_MAJOR_VERSION``.
            json.JSONDecodeError: Vendored file is not valid JSON.
        """
        if engine not in _KNOWN_ENGINES:
            raise ValueError(f"Unknown engine {engine!r}. Known engines: {list(_KNOWN_ENGINES)}.")

        cached = self._cache.get(engine)
        if cached is not None:
            return cached

        try:
            raw_text = (resources.files(_PACKAGE) / f"{engine}.json").read_text()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Vendored schema for engine {engine!r} not found. "
                f"Run `./scripts/refresh_discovered_schemas.sh {engine}` to generate it."
            ) from exc

        parsed = _parse_envelope(engine=engine, raw_text=raw_text)
        self._cache[engine] = parsed
        return parsed

    def load_all_schemas(self) -> dict[str, DiscoveredSchema]:
        """Load all known engines' schemas.

        Does not skip missing files — every engine in ``_KNOWN_ENGINES`` must
        have a vendored schema. Callers that need tolerance should iterate and
        catch ``FileNotFoundError`` themselves.
        """
        return {engine: self.load_schema(engine) for engine in _KNOWN_ENGINES}

    def invalidate(self, engine: str | None = None) -> None:
        """Drop cached schema(s). Useful after a schema refresh in-process."""
        if engine is None:
            self._cache.clear()
        else:
            self._cache.pop(engine, None)


def _parse_envelope(*, engine: str, raw_text: str) -> DiscoveredSchema:
    data = json.loads(raw_text)

    schema_version = data["schema_version"]
    major = _major_version(schema_version)
    if major != SUPPORTED_MAJOR_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Vendored schema for {engine!r} has schema_version={schema_version!r} "
            f"(major={major}); this SchemaLoader only supports major "
            f"{SUPPORTED_MAJOR_VERSION}. Regenerate with a matching discovery script, "
            f"or upgrade the loader."
        )

    limitations_raw = data.get("discovery_limitations", [])
    limitations = [
        DiscoveryLimitation(
            section=item.get("section", ""),
            fields=list(item.get("fields", [])),
            reason=item.get("reason", ""),
        )
        for item in limitations_raw
    ]

    return DiscoveredSchema(
        schema_version=schema_version,
        engine=data["engine"],
        engine_version=data["engine_version"],
        engine_commit_sha=data.get("engine_commit_sha"),
        image_ref=data["image_ref"],
        base_image_ref=data.get("base_image_ref", data["image_ref"]),
        discovered_at=_parse_iso(data["discovered_at"]),
        discovery_method=data.get("discovery_method", ""),
        discovery_limitations=limitations,
        engine_params=data.get("engine_params", {}),
        sampling_params=data.get("sampling_params", {}),
    )


def _major_version(version: str) -> int:
    """Parse major from a semver-ish string. ``"1.0.0"`` -> ``1``."""
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError) as exc:
        raise UnsupportedSchemaVersionError(
            f"Unparseable schema_version {version!r}: expected semver like '1.0.0'."
        ) from exc


def _parse_iso(value: str) -> datetime:
    # Accept both "...+00:00" and "...Z" terminations.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)
