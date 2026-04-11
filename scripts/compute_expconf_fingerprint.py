#!/usr/bin/env python3
"""Print the ExperimentConfig schema fingerprint for Docker build stamping.

Emits a 64-character SHA-256 hex digest of the ExperimentConfig JSON schema
serialised deterministically. Intended as a build-arg so Docker images
can be compared against the running host's fingerprint at study start.
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
