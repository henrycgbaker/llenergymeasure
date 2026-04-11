#!/usr/bin/env python3
"""Compute the ExperimentConfig schema fingerprint for Docker build stamping.

Prints a 64-character SHA-256 hex digest of
``ExperimentConfig.model_json_schema()`` serialised with
``sort_keys=True, separators=(',', ':')``. Used by:

- ``Makefile`` as ``LLEM_EXPCONF_SCHEMA_FINGERPRINT`` build-arg source.
- ``.github/workflows/docker-publish.yml`` so published images carry the same
  fingerprint that host-side ``StudyRunner._prepare_images`` will compare
  against at run time.

This script is intentionally tiny and dependency-free beyond ``pydantic``
(implicit via the package) so it can run in any environment where
llenergymeasure is installed or its source tree is on ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Allow running from a checkout without an editable install."""
    repo_src = Path(__file__).resolve().parent.parent / "src"
    if repo_src.is_dir() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))


def main() -> int:
    _ensure_src_on_path()
    from llenergymeasure.infra.version_handshake import compute_expconf_fingerprint

    print(compute_expconf_fingerprint())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
