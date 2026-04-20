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
