#!/usr/bin/env python3
"""Check that Dockerfile engine version ARGs match vendored schema engine_versions.

Covers all engines where the Dockerfile ARG directly corresponds to the engine
version tracked in the schema:
  - vllm: ARG VLLM_VERSION in docker/Dockerfile.vllm
  - tensorrt: ARG TRTLLM_VERSION in docker/Dockerfile.tensorrt
  - transformers: ARG TRANSFORMERS_VERSION in docker/Dockerfile.transformers

Exit codes:
    0 = all versions match (or non-version Dockerfile change)
    1 = mismatch detected
    2 = error (missing file, parse failure)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (engine_name, arg_name)
_ENGINE_SPECS = [
    ("vllm", "VLLM_VERSION"),
    ("tensorrt", "TRTLLM_VERSION"),
    ("transformers", "TRANSFORMERS_VERSION"),
]


def _normalize_version(version: str) -> str:
    """Strip leading 'v' prefix for comparison."""
    return version.lstrip("v")


def _parse_arg(dockerfile: Path, arg_name: str) -> str | None:
    """Extract ARG default value from a Dockerfile."""
    pattern = re.compile(rf"^ARG\s+{re.escape(arg_name)}=(\S+)")
    for line in dockerfile.read_text().splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1)
    return None


def _parse_schema_version(schema_path: Path) -> str | None:
    """Extract engine_version from a vendored schema JSON."""
    data = json.loads(schema_path.read_text())
    return data.get("engine_version")


def main(repo_root: Path | None = None) -> int:
    root = repo_root or REPO_ROOT
    schema_dir = root / "src" / "llenergymeasure" / "config" / "discovered_schemas"

    errors: list[str] = []
    mismatches: list[str] = []

    for engine, arg_name in _ENGINE_SPECS:
        dockerfile = root / "docker" / f"Dockerfile.{engine}"
        if not dockerfile.exists():
            errors.append(f"{engine}: Dockerfile not found: {dockerfile}")
            continue

        schema_path = schema_dir / f"{engine}.json"
        if not schema_path.exists():
            errors.append(f"{engine}: schema not found: {schema_path}")
            continue

        dockerfile_version = _parse_arg(dockerfile, arg_name)
        if dockerfile_version is None:
            errors.append(f"{engine}: ARG {arg_name} not found in {dockerfile.name}")
            continue

        schema_version = _parse_schema_version(schema_path)
        if schema_version is None:
            errors.append(f"{engine}: engine_version not found in {schema_path.name}")
            continue

        if _normalize_version(dockerfile_version) != _normalize_version(schema_version):
            mismatches.append(
                f"MISMATCH: {dockerfile.name} pins {arg_name}={dockerfile_version} "
                f"but schema was discovered against {schema_version}\n"
                f"  Run: ./scripts/update_engine_schema.sh {engine}"
            )

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if mismatches:
        for m in mismatches:
            print(m, file=sys.stderr)
        return 1

    print("All schema versions match Dockerfile ARGs.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
