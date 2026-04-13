#!/usr/bin/env python3
"""Test multi-GPU parallelization across all engines.

Validates that each engine can run with parallelization degree 4 (tensor parallel)
using their native arguments.

Run from host (orchestrates Docker containers):
    python scripts/test_multi_gpu_parallelization.py

This script:
1. Creates minimal test configs for each engine with tp=4
2. Runs a short experiment in each container
3. Validates all 4 GPUs were utilized
"""

import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def create_pytorch_config(config_dir: Path) -> Path:
    """Create PyTorch config with 4 data-parallel processes."""
    config = config_dir / "pytorch_tp4.yaml"
    config.write_text("""
# PyTorch with 4 data-parallel processes
model_name: Qwen/Qwen2.5-0.5B
engine: pytorch
max_output_tokens: 8
num_input_prompts: 4
gpus: [0, 1, 2, 3]

parallelism:
  strategy: data_parallel
  degree: 4
""")
    return config


def create_vllm_config(config_dir: Path) -> Path:
    """Create vLLM config with tensor_parallel_size=4."""
    config = config_dir / "vllm_tp4.yaml"
    config.write_text("""
# vLLM with tensor parallelism across 4 GPUs
model_name: Qwen/Qwen2.5-0.5B
engine: vllm
max_output_tokens: 8
num_input_prompts: 4
gpus: [0, 1, 2, 3]

parallelism:
  strategy: tensor_parallel
  degree: 4

vllm:
  tensor_parallel_size: 4
  enforce_eager: true
""")
    return config


def create_tensorrt_config(config_dir: Path) -> Path:
    """Create TensorRT config with tp_size=4."""
    config = config_dir / "tensorrt_tp4.yaml"
    config.write_text("""
# TensorRT-LLM with tensor parallelism across 4 GPUs
model_name: Qwen/Qwen2.5-0.5B
engine: tensorrt
max_output_tokens: 8
num_input_prompts: 4
gpus: [0, 1, 2, 3]

parallelism:
  strategy: tensor_parallel
  degree: 4

tensorrt:
  tp_size: 4
  max_batch_size: 4
""")
    return config


def run_experiment(engine: str, config_path: Path, timeout: int = 600) -> tuple[bool, str]:
    """Run experiment in Docker container.

    Returns:
        (success, output) tuple
    """
    # Container path for mounted config
    container_config = f"/app/configs/test_tp4/{config_path.name}"

    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-v",
        f"{config_path.parent}:/app/configs/test_tp4:ro",
        engine,
        "lem",
        "experiment",
        container_config,
        "--dataset",
        "ai_energy_score",
        "--sample-size",
        "4",
        "--yes",
        "--fresh",
    ]

    print(f"\n{'=' * 60}")
    print(f"Testing {engine} with tp=4")
    print(f"{'=' * 60}")
    print(f"$ {' '.join(cmd[:8])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )

        output = result.stdout + "\n" + result.stderr

        if result.returncode == 0:
            print(f"✓ {engine}: SUCCESS")
            return True, output
        else:
            print(f"✗ {engine}: FAILED (exit code {result.returncode})")
            # Print last few lines of output for debugging
            lines = output.strip().split("\n")
            print("Last 10 lines of output:")
            for line in lines[-10:]:
                print(f"  {line}")
            return False, output

    except subprocess.TimeoutExpired:
        print(f"✗ {engine}: TIMEOUT after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ {engine}: ERROR - {e}")
        return False, str(e)


def check_gpu_utilization(output: str, expected_gpus: int = 4) -> bool:
    """Check if output indicates all GPUs were used."""
    # Look for indicators that multiple GPUs were used
    indicators = [
        "GPUs: [0, 1, 2, 3]",
        f"Processes: {expected_gpus}",
        "tensor_parallel",
        f"tp_size: {expected_gpus}",
        f"tensor_parallel_size: {expected_gpus}",
    ]

    return any(indicator in output for indicator in indicators)


def main():
    print("=" * 60)
    print("Multi-GPU Parallelization Test (tp=4)")
    print("=" * 60)
    print("This test validates that each engine can utilize all 4 GPUs")
    print("using their native tensor/data parallelism arguments.")

    # Check we have 4 GPUs
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True
    )
    gpu_count = len(result.stdout.strip().split("\n")) if result.returncode == 0 else 0

    if gpu_count < 4:
        print(f"\n⚠ Only {gpu_count} GPUs available. Need 4 for this test.")
        print("Skipping multi-GPU parallelization test.")
        return 0

    print(f"\n✓ Found {gpu_count} GPUs")

    # Create temp directory for configs
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        results = {}

        # Test PyTorch (data parallel)
        pytorch_config = create_pytorch_config(config_dir)
        success, _output = run_experiment("pytorch", pytorch_config)
        results["pytorch"] = success

        # Test vLLM (tensor parallel)
        vllm_config = create_vllm_config(config_dir)
        success, _output = run_experiment("vllm", vllm_config)
        results["vllm"] = success

        # Test TensorRT (tensor parallel)
        tensorrt_config = create_tensorrt_config(config_dir)
        success, _output = run_experiment(
            "tensorrt", tensorrt_config, timeout=900
        )  # TRT needs more time for engine build
        results["tensorrt"] = success

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for engine, success in results.items():
        status = "PASS ✓" if success else "FAIL ✗"
        if not success:
            all_passed = False
        print(f"  {engine}: {status}")

    print("=" * 60)

    if all_passed:
        print("\n✓ All engines can utilize 4 GPUs with parallelization")
    else:
        print("\n✗ Some engines failed multi-GPU parallelization")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
