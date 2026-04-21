"""Tests for scripts/check_schema_versions.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the script's main function directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from check_schema_versions import main


def _setup_repo(
    tmp_path: Path,
    *,
    vllm_arg: str = "v0.7.3",
    vllm_schema_version: str = "0.7.3",
    trt_arg: str = "0.21.0",
    trt_schema_version: str = "0.21.0",
    transformers_arg: str = "5.5.4",
    transformers_schema_version: str = "5.5.4",
    skip_vllm_schema: bool = False,
) -> Path:
    """Create a minimal repo structure for the version check script."""
    repo = tmp_path / "repo"
    docker = repo / "docker"
    docker.mkdir(parents=True)

    (docker / "Dockerfile.vllm").write_text(f"FROM ubuntu:22.04\nARG VLLM_VERSION={vllm_arg}\n")
    (docker / "Dockerfile.tensorrt").write_text(
        f"FROM ubuntu:22.04\nARG TRTLLM_VERSION={trt_arg}\n"
    )
    (docker / "Dockerfile.transformers").write_text(
        f"FROM ubuntu:22.04\nARG TRANSFORMERS_VERSION={transformers_arg}\n"
    )

    schema_dir = repo / "src" / "llenergymeasure" / "config" / "discovered_schemas"
    schema_dir.mkdir(parents=True)

    if not skip_vllm_schema:
        (schema_dir / "vllm.json").write_text(json.dumps({"engine_version": vllm_schema_version}))
    (schema_dir / "tensorrt.json").write_text(json.dumps({"engine_version": trt_schema_version}))
    (schema_dir / "transformers.json").write_text(
        json.dumps({"engine_version": transformers_schema_version})
    )

    return repo


class TestVersionsMatch:
    def test_all_match(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        assert main(repo_root=repo) == 0

    def test_v_prefix_normalised(self, tmp_path: Path):
        """v0.7.3 in Dockerfile should match 0.7.3 in schema."""
        repo = _setup_repo(tmp_path, vllm_arg="v0.7.3", vllm_schema_version="0.7.3")
        assert main(repo_root=repo) == 0


class TestMismatch:
    def test_version_mismatch(self, tmp_path: Path, capsys):
        repo = _setup_repo(tmp_path, vllm_arg="v0.8.0", vllm_schema_version="0.7.3")
        code = main(repo_root=repo)
        assert code == 1
        captured = capsys.readouterr()
        assert "MISMATCH" in captured.err
        assert "update_engine_schema.sh" in captured.err

    def test_transformers_mismatch(self, tmp_path: Path, capsys):
        repo = _setup_repo(
            tmp_path,
            transformers_arg="5.6.0",
            transformers_schema_version="5.5.4",
        )
        code = main(repo_root=repo)
        assert code == 1
        captured = capsys.readouterr()
        assert "MISMATCH" in captured.err
        assert "transformers" in captured.err


class TestErrors:
    def test_missing_schema_file(self, tmp_path: Path, capsys):
        repo = _setup_repo(tmp_path, skip_vllm_schema=True)
        code = main(repo_root=repo)
        assert code == 2
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
