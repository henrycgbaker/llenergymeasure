"""Env-var helpers for opinionated runtime defaults.

This module is the canonical location for ``LLEM_*`` env-var constants and
the thin passthrough helpers that read them. Helpers are pure — they return
``os.environ.get(...)`` (or a parsed form) and return ``None`` / ``False``
when unset, deferring to each inference library's own default.

Opinionated defaults (e.g. ``LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP=auto``) live in the
repo-root ``.env.example`` — the single reviewable source of truth for what
llem overrides out-of-the-box. Users copy ``.env.example`` to ``.env`` (auto-
loaded by the CLI) and edit to taste. Because helpers have no baked-in
defaults, removing a line from ``.env`` always restores the library default.

Layer: ``utils/`` (Layer 0). Cannot import ``config/``. This module is
consumed by engine plugins in Layer 2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

ENV_TRANSFORMERS_DEFAULT_DEVICE_MAP: Final = "LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP"
"""Override for the HuggingFace ``device_map`` argument at model load.

Unset / empty → ``None`` (caller omits the kwarg; HuggingFace's own default
applies, which is CPU-only). Any non-empty value is forwarded as-is
(e.g. ``auto``, ``balanced``, ``sequential``). The opinionated default
``auto`` is shipped via ``.env.example`` — not baked into this helper.
"""


def default_device_map() -> str | None:
    """Return the configured default ``device_map`` or ``None`` if unset.

    Pure passthrough: no opinionated default is baked in. The repo-root
    ``.env.example`` ships ``LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP=auto`` so that the
    out-of-the-box experience on multi-GPU hosts is ``device_map="auto"``;
    deleting the line from a user's ``.env`` reverts to HuggingFace's own
    default (``None`` = CPU-only).

    Callers typically apply typed-config precedence first (e.g. the
    ``transformers.device_map`` field in YAML), then fall back to this
    helper. If this returns ``None``, callers should omit the kwarg.
    """
    return os.environ.get(ENV_TRANSFORMERS_DEFAULT_DEVICE_MAP) or None


ENV_TRT_BUILD_CACHE_ENABLED: Final = "LLEM_TRT_BUILD_CACHE_ENABLED"
"""Toggle for TRT-LLM on-disk engine build cache.

Unset / empty / any falsy value (``0``, ``false``, ``no``, ``off``) → False,
matching TRT-LLM's own default. Truthy values (``1``, ``true``, ``yes``,
``on``, case-insensitive) → True. The opinionated default ``1`` is shipped
via ``.env.example`` — not baked into this helper.
"""


ENV_TRT_BUILD_CACHE_PATH: Final = "LLEM_TRT_BUILD_CACHE_PATH"
"""User-supplied cache directory for TRT-LLM engine build cache.

If set and non-empty, the engine plugin wraps it into TRT-LLM's
``BuildCacheConfig.cache_root``. Unset / empty leaves TRT-LLM's internal
default cache location in place.
"""


def trt_build_cache_enabled() -> bool:
    """Return whether TRT-LLM on-disk engine build cache should be enabled.

    Pure passthrough: no opinionated default is baked in. The repo-root
    ``.env.example`` ships ``LLEM_TRT_BUILD_CACHE_ENABLED=1`` so the
    out-of-the-box experience preserves the cache (engine compilation takes
    minutes); deleting the line reverts to TRT-LLM's disabled default.
    """
    return os.environ.get(ENV_TRT_BUILD_CACHE_ENABLED, "").lower() in {"1", "true", "yes", "on"}


def trt_build_cache_path() -> Path | None:
    """Return the user-supplied TRT-LLM build cache root, if any.

    Returns a ``Path`` when set to a non-empty value; otherwise ``None`` so
    TRT-LLM uses its internal default location (``~/.cache/tensorrt_llm/``).
    """
    raw = os.environ.get(ENV_TRT_BUILD_CACHE_PATH)
    return Path(raw) if raw else None
