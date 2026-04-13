#!/usr/bin/env python3
"""Runtime test orchestrator - dispatches tests to engine containers.

This orchestrator runs on the HOST and dispatches each parameter test to the
appropriate engine container (pytorch, vllm, tensorrt) using the same Docker
dispatch pattern as the campaign CLI.

Key features:
- Uses SSOT introspection to discover ALL params for ALL engines
- Routes each test to the correct Docker container based on engine
- Supports parallel execution within each engine
- Generates comprehensive test reports

Usage:
    # Run all engines (recommended)
    python scripts/runtime-test-orchestrator.py

    # Run specific engine
    python scripts/runtime-test-orchestrator.py --engine pytorch

    # Quick mode (fewer params)
    python scripts/runtime-test-orchestrator.py --quick

    # List params without running
    python scripts/runtime-test-orchestrator.py --list-params

    # Check Docker setup
    python scripts/runtime-test-orchestrator.py --check-docker
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Configuration
# =============================================================================

TEST_MODEL = "Qwen/Qwen2.5-0.5B"
TEST_SAMPLE_SIZE = 5
TEST_MAX_OUTPUT = 32
TEST_TIMEOUT_SECONDS = 300  # 5 minutes per test

ENGINES = ["transformers", "vllm", "tensorrt"]

# Quick mode: reduced param set for faster iteration
QUICK_PARAMS = {
    "transformers": ["transformers.batch_size", "decoder.temperature"],
    "vllm": ["vllm.max_num_seqs", "decoder.temperature"],
    "tensorrt": ["tensorrt.max_batch_size", "decoder.temperature"],
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TestCase:
    """A single test case to execute."""

    engine: str
    param_path: str
    param_value: Any
    config_path: Path | None = None
    container_config_path: str | None = None


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_case: TestCase
    status: str  # passed | failed | skipped
    exit_code: int
    elapsed_seconds: float
    error_summary: str | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class OrchestratorReport:
    """Complete orchestrator report."""

    run_id: str
    timestamp: str
    summary: dict[str, dict[str, int]]  # {engine: {passed, failed, skipped}}
    results: list[TestResult]
    total_elapsed_seconds: float


# =============================================================================
# SSOT Parameter Discovery
# =============================================================================


def get_all_engine_params() -> dict[str, dict[str, list[Any]]]:
    """Get all params for all engines using SSOT introspection.

    Uses the same introspection functions as tests/runtime/test_all_params.py
    (the canonical source for parameter testing). This ensures the orchestrator
    tests exactly the same params that the standalone test suite would test.

    Returns:
        {engine: {param_path: [test_values]}}
    """
    try:
        from llenergymeasure.config.introspection import (
            get_engine_params,
            get_shared_params,
        )

        all_params: dict[str, dict[str, list[Any]]] = {}

        # Get shared params (universal - apply to all engines)
        shared = get_shared_params()
        shared_test_values: dict[str, list[Any]] = {}
        for param_path, meta in shared.items():
            test_values = meta.get("test_values", [])
            if test_values:
                shared_test_values[param_path] = test_values

        # Get engine-specific params (Tier 2 - engine-native)
        for engine in ENGINES:
            engine_params = get_engine_params(engine)
            params: dict[str, list[Any]] = {}

            # Add engine-specific params
            for param_path, meta in engine_params.items():
                test_values = meta.get("test_values", [])
                if test_values:
                    params[param_path] = test_values

            # Add shared params to each engine
            params.update(shared_test_values)

            all_params[engine] = params

        return all_params

    except ImportError as e:
        print(f"Error: Could not import introspection module: {e}")
        print("Make sure the package is installed: pip install -e .")
        sys.exit(1)


def filter_quick_params(
    all_params: dict[str, dict[str, list[Any]]],
) -> dict[str, dict[str, list[Any]]]:
    """Filter to quick mode subset of params."""
    filtered: dict[str, dict[str, list[Any]]] = {}
    for engine, params in all_params.items():
        quick_list = QUICK_PARAMS.get(engine, [])
        filtered[engine] = {k: v for k, v in params.items() if k in quick_list}
    return filtered


# =============================================================================
# Docker Management
# =============================================================================


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_docker_images(engines: list[str]) -> tuple[list[str], list[str]]:
    """Check which engine images exist.

    Returns:
        (existing_images, missing_images)
    """
    existing = []
    missing = []
    for engine in engines:
        image_name = f"llenergymeasure:{engine}"
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            existing.append(engine)
        else:
            missing.append(engine)
    return existing, missing


def ensure_env_file() -> None:
    """Ensure .env file exists with PUID/PGID for Docker permissions."""
    env_file = PROJECT_ROOT / ".env"

    # Export vars regardless (for docker compose)
    os.environ.setdefault("PUID", str(os.getuid()))
    os.environ.setdefault("PGID", str(os.getgid()))

    # Skip if .env is not a regular writable file
    if env_file.exists() and not env_file.is_file():
        return

    # Create .env if it doesn't exist or is empty
    if not env_file.exists():
        with contextlib.suppress(OSError):
            env_file.write_text(f"PUID={os.getuid()}\nPGID={os.getgid()}\n")


def build_docker_images(engines: list[str]) -> bool:
    """Build Docker images for specified engines.

    Returns:
        True if all builds succeeded.
    """
    for engine in engines:
        print(f"  Building {engine} image...")
        result = subprocess.run(
            ["docker", "compose", "build", engine],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(f"  [ERROR] Failed to build {engine}")
            return False
        print(f"  [OK] {engine} built successfully")
    return True


# =============================================================================
# Config Generation
# =============================================================================


def create_base_config(engine: str) -> dict[str, Any]:
    """Create a minimal base config for testing."""
    config: dict[str, Any] = {
        "config_name": f"{engine}-test-base",
        "model_name": TEST_MODEL,
        "engine": engine,
        "gpus": [0],
        "max_input_tokens": 64,
        "max_output_tokens": TEST_MAX_OUTPUT,
        "num_input_prompts": TEST_SAMPLE_SIZE,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "dataset": {"name": "ai_energy_score", "sample_size": TEST_SAMPLE_SIZE},
    }

    # Engine-specific defaults
    if engine == "transformers":
        config["transformers"] = {
            "batch_size": 1,
            "batching_strategy": "static",
            "attn_implementation": "sdpa",
        }
    elif engine == "vllm":
        config["vllm"] = {
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 512,
        }
    elif engine == "tensorrt":
        config["tensorrt"] = {
            "max_batch_size": 4,
            "builder_opt_level": 3,
            "force_rebuild": True,
        }

    return config


def create_test_config(
    engine: str,
    param_path: str,
    param_value: Any,
    config_dir: Path,
) -> Path:
    """Create a config file with a single param variation.

    Args:
        engine: Engine name (pytorch, vllm, tensorrt)
        param_path: Dotted param path (e.g., "decoder.temperature")
        param_value: Value to set
        config_dir: Directory to write config file

    Returns:
        Path to the created config file
    """
    import yaml

    config = create_base_config(engine)

    # Apply the variation
    parts = param_path.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]

    # Set the value (handle None/null)
    target[parts[-1]] = param_value

    # Update config name
    safe_value = str(param_value).replace(".", "_").replace("/", "_").replace(" ", "_")
    config["config_name"] = f"{engine}_{parts[-1]}_{safe_value}"

    # Write config
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config['config_name']}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


# =============================================================================
# Test Execution
# =============================================================================


def run_test_in_container(
    test_case: TestCase,
    config_dir: Path,
) -> TestResult:
    """Run a single test in the appropriate engine container.

    Args:
        test_case: Test case to execute
        config_dir: Directory containing config files (bind-mounted to /app/configs/)

    Returns:
        TestResult with execution details
    """
    start_time = time.time()

    # Config path inside container
    container_config_path = f"/app/configs/test_grid/{test_case.config_path.name}"

    # Build docker compose run command
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        test_case.engine,
        "lem",
        "experiment",
        container_config_path,
        "--yes",
        "--fresh",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
        )

        elapsed = time.time() - start_time

        # Determine status from exit code
        if result.returncode == 0:
            status = "passed"
            error_summary = None
        else:
            status = "failed"
            # Extract brief error summary
            stderr_lines = result.stderr.strip().split("\n")
            error_summary = stderr_lines[-1] if stderr_lines else "Unknown error"

        return TestResult(
            test_case=test_case,
            status=status,
            exit_code=result.returncode,
            elapsed_seconds=elapsed,
            error_summary=error_summary,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return TestResult(
            test_case=test_case,
            status="failed",
            exit_code=-1,
            elapsed_seconds=elapsed,
            error_summary=f"Timeout after {TEST_TIMEOUT_SECONDS}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test_case=test_case,
            status="failed",
            exit_code=-2,
            elapsed_seconds=elapsed,
            error_summary=str(e),
        )


def run_engine_tests(
    engine: str,
    params: dict[str, list[Any]],
    config_dir: Path,
) -> list[TestResult]:
    """Run all tests for a single engine.

    Args:
        engine: Engine name
        params: {param_path: [test_values]}
        config_dir: Directory for config files

    Returns:
        List of test results
    """
    results: list[TestResult] = []
    total_tests = sum(len(values) for values in params.values())

    print(f"\n{'=' * 60}")
    print(f"  Engine: {engine.upper()}")
    print(f"  Tests: {total_tests} ({len(params)} params)")
    print(f"{'=' * 60}\n")

    test_idx = 0
    for param_path, values in params.items():
        for value in values:
            test_idx += 1
            print(f"  [{test_idx}/{total_tests}] {param_path}={value}", end=" ... ", flush=True)

            # Create config
            config_path = create_test_config(engine, param_path, value, config_dir)

            # Create test case
            test_case = TestCase(
                engine=engine,
                param_path=param_path,
                param_value=value,
                config_path=config_path,
            )

            # Run test
            result = run_test_in_container(test_case, config_dir)
            results.append(result)

            # Report result
            if result.status == "passed":
                print(f"✓ ({result.elapsed_seconds:.1f}s)")
            else:
                print(f"✗ ({result.elapsed_seconds:.1f}s)")
                if result.error_summary:
                    print(f"      Error: {result.error_summary[:80]}")

    return results


# =============================================================================
# Reporting
# =============================================================================


def generate_report(
    results: list[TestResult],
    total_elapsed: float,
) -> OrchestratorReport:
    """Generate comprehensive test report."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate summary by engine
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        engine = result.test_case.engine
        if engine not in summary:
            summary[engine] = {"passed": 0, "failed": 0, "skipped": 0}
        summary[engine][result.status] += 1

    return OrchestratorReport(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        summary=summary,
        results=results,
        total_elapsed_seconds=total_elapsed,
    )


def print_summary(report: OrchestratorReport) -> None:
    """Print human-readable summary."""
    print("\n")
    print("=" * 60)
    print("  RUNTIME TEST SUMMARY")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for engine, counts in report.summary.items():
        passed = counts.get("passed", 0)
        failed = counts.get("failed", 0)
        skipped = counts.get("skipped", 0)
        total = passed + failed + skipped

        total_passed += passed
        total_failed += failed
        total_skipped += skipped

        status_icon = "✓" if failed == 0 else "✗"
        print(f"\n  {status_icon} {engine.upper()}: {passed}/{total} passed", end="")
        if failed > 0:
            print(f" ({failed} failed)", end="")
        if skipped > 0:
            print(f" ({skipped} skipped)", end="")
        print()

    total = total_passed + total_failed + total_skipped
    print(f"\n  {'─' * 50}")
    print(f"  TOTAL: {total_passed}/{total} passed", end="")
    if total_failed > 0:
        print(f" | {total_failed} failed", end="")
    print(f" | {report.total_elapsed_seconds:.1f}s")
    print("=" * 60)

    # List failed tests
    failed_results = [r for r in report.results if r.status == "failed"]
    if failed_results:
        print("\n  FAILED TESTS:")
        for result in failed_results[:10]:  # Show first 10
            tc = result.test_case
            print(f"    - {tc.engine}/{tc.param_path}={tc.param_value}")
            if result.error_summary:
                print(f"      {result.error_summary[:60]}")
        if len(failed_results) > 10:
            print(f"    ... and {len(failed_results) - 10} more")


def save_report(report: OrchestratorReport, output_dir: Path) -> Path:
    """Save report to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"runtime_test_report_{report.run_id}.json"

    # Convert to serializable format
    data = {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "summary": report.summary,
        "total_elapsed_seconds": report.total_elapsed_seconds,
        "results": [
            {
                "engine": r.test_case.engine,
                "param_path": r.test_case.param_path,
                "param_value": r.test_case.param_value,
                "status": r.status,
                "exit_code": r.exit_code,
                "elapsed_seconds": r.elapsed_seconds,
                "error_summary": r.error_summary,
            }
            for r in report.results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return report_path


# =============================================================================
# Main
# =============================================================================


def list_params(all_params: dict[str, dict[str, list[Any]]]) -> None:
    """Print all discovered params."""
    print("\n" + "=" * 60)
    print("  DISCOVERED PARAMETERS (SSOT)")
    print("=" * 60)

    total = 0
    for engine in ENGINES:
        params = all_params.get(engine, {})
        test_count = sum(len(v) for v in params.values())
        total += test_count

        print(f"\n  {engine.upper()} ({len(params)} params, {test_count} tests):")
        for param_path, values in sorted(params.items()):
            values_str = ", ".join(str(v) for v in values[:5])
            if len(values) > 5:
                values_str += f", ... ({len(values)} total)"
            print(f"    {param_path}: [{values_str}]")

    print(f"\n  {'─' * 50}")
    print(f"  TOTAL: {total} tests across {len(ENGINES)} engines")
    print("=" * 60)


def check_docker_setup() -> bool:
    """Check Docker setup and report status."""
    print("\n" + "=" * 60)
    print("  DOCKER SETUP CHECK")
    print("=" * 60)

    # Check Docker available
    if not check_docker_available():
        print("\n  [ERROR] Docker is not available or not running")
        print("  Please start Docker and try again.")
        return False
    print("\n  [OK] Docker is running")

    # Check images
    existing, missing = check_docker_images(ENGINES)

    if existing:
        print(f"  [OK] Images ready: {', '.join(existing)}")
    if missing:
        print(f"  [WARN] Images missing: {', '.join(missing)}")
        print(f"\n  Build with: docker compose build {' '.join(missing)}")

    print("=" * 60)
    return len(missing) == 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Runtime test orchestrator - dispatches tests to engine containers"
    )
    parser.add_argument(
        "--engine",
        choices=[*ENGINES, "all"],
        default="all",
        help="Engine to test (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick subset of params",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List all params without running tests",
    )
    parser.add_argument(
        "--check-docker",
        action="store_true",
        help="Check Docker setup and exit",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build missing Docker images before running",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "runtime_tests",
        help="Directory for test reports",
    )

    args = parser.parse_args()

    # Check Docker setup
    if args.check_docker:
        return 0 if check_docker_setup() else 1

    # Discover params
    print("\n[INFO] Discovering params via SSOT introspection...")
    all_params = get_all_engine_params()

    if args.quick:
        print("[INFO] Quick mode: using reduced param set")
        all_params = filter_quick_params(all_params)

    # Filter to specific engine if requested
    if args.engine != "all":
        all_params = {args.engine: all_params.get(args.engine, {})}

    # List params mode
    if args.list_params:
        list_params(all_params)
        return 0

    # Check Docker is available
    if not check_docker_available():
        print("[ERROR] Docker is not available or not running")
        return 1

    # Check/build images
    engines_needed = list(all_params.keys())
    _existing, missing = check_docker_images(engines_needed)

    if missing:
        if args.build:
            print(f"[INFO] Building missing images: {', '.join(missing)}")
            if not build_docker_images(missing):
                return 1
        else:
            print(f"[ERROR] Missing Docker images: {', '.join(missing)}")
            print(f"Run: docker compose build {' '.join(missing)}")
            print("Or use --build flag to build automatically")
            return 1

    # Ensure .env file
    ensure_env_file()

    # Create config directory (configs/test_grid is bind-mounted to /app/configs/test_grid)
    config_dir = PROJECT_ROOT / "configs" / "test_grid"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    print("\n" + "=" * 60)
    print("  RUNTIME PARAMETER TESTS")
    print("=" * 60)

    total_tests = sum(sum(len(v) for v in params.values()) for params in all_params.values())
    print(f"\n  Engines: {', '.join(engines_needed)}")
    print(f"  Total tests: {total_tests}")
    print(f"  Output: {args.output_dir}")

    start_time = time.time()
    all_results: list[TestResult] = []

    for engine in engines_needed:
        params = all_params.get(engine, {})
        if not params:
            print(f"\n[WARN] No params found for {engine}")
            continue

        results = run_engine_tests(engine, params, config_dir)
        all_results.extend(results)

    total_elapsed = time.time() - start_time

    # Generate and display report
    report = generate_report(all_results, total_elapsed)
    print_summary(report)

    # Save report
    report_path = save_report(report, args.output_dir)
    print(f"\n  Report saved: {report_path}")

    # Return exit code based on failures
    total_failed = sum(counts.get("failed", 0) for counts in report.summary.values())
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
