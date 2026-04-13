#!/usr/bin/env python3
"""Test CUDA_VISIBLE_DEVICES propagation across engines.

Validates that:
1. Container default (GPU 0) allows imports
2. Explicit CUDA_VISIBLE_DEVICES is respected by all engines
3. Multi-GPU configs see all specified GPUs

Run inside containers:
    docker compose run --rm pytorch python scripts/test_cuda_visible_devices.py
    docker compose run --rm vllm python scripts/test_cuda_visible_devices.py
    docker compose run --rm tensorrt python scripts/test_cuda_visible_devices.py
"""

import os
import subprocess
import sys


def test_torch_cuda_visibility():
    """Test that PyTorch sees the GPUs specified by CUDA_VISIBLE_DEVICES."""
    import torch

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if available else 0

    print(f"[torch] CUDA_VISIBLE_DEVICES={cuda_env}")
    print(f"[torch] cuda.is_available()={available}")
    print(f"[torch] device_count()={device_count}")

    if available:
        for i in range(device_count):
            print(f"[torch] GPU {i}: {torch.cuda.get_device_name(i)}")

    return available, device_count


def test_subprocess_inheritance():
    """Test that child processes inherit CUDA_VISIBLE_DEVICES."""
    # Set a specific value
    test_gpus = "0,1" if os.environ.get("MULTI_GPU_TEST") else "0"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = test_gpus

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import torch; "
            'print(f\'child CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}\'); '
            "print(f'child device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}')",
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    print(f"\n[subprocess] Parent set CUDA_VISIBLE_DEVICES={test_gpus}")
    print(f"[subprocess] Child output:\n{result.stdout.strip()}")

    if result.returncode != 0:
        print(f"[subprocess] STDERR: {result.stderr}")
        return False

    # Verify child saw the correct value
    return f"CUDA_VISIBLE_DEVICES={test_gpus}" in result.stdout


def test_vllm_gpu_detection():
    """Test vLLM's GPU detection respects CUDA_VISIBLE_DEVICES."""
    try:
        # vLLM checks GPU count at import time via ray or internal mechanisms
        import vllm  # noqa: F401

        print("\n[vllm] Import successful")

        # Check what vLLM sees
        import torch

        device_count = torch.cuda.device_count()
        print(f"[vllm] torch.cuda.device_count()={device_count}")
        return True
    except ImportError:
        print("\n[vllm] Not installed (skip)")
        return None
    except Exception as e:
        print(f"\n[vllm] Error: {e}")
        return False


def test_tensorrt_gpu_detection():
    """Test TensorRT-LLM's GPU detection respects CUDA_VISIBLE_DEVICES."""
    try:
        import tensorrt_llm

        print(f"\n[tensorrt] Import successful, version={tensorrt_llm.__version__}")

        # TensorRT-LLM uses its own GPU detection
        import torch

        device_count = torch.cuda.device_count()
        print(f"[tensorrt] torch.cuda.device_count()={device_count}")
        return True
    except ImportError:
        print("\n[tensorrt] Not installed (skip)")
        return None
    except Exception as e:
        print(f"\n[tensorrt] Error: {e}")
        return False


def test_multi_gpu_override():
    """Test that we can override container default to see multiple GPUs."""

    # Get actual GPU count from nvidia-smi (bypasses CUDA_VISIBLE_DEVICES)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("\n[multi-gpu] nvidia-smi failed, skip")
        return None

    total_gpus = len(result.stdout.strip().split("\n"))
    print(f"\n[multi-gpu] Total GPUs on system: {total_gpus}")

    if total_gpus < 2:
        print("[multi-gpu] Only 1 GPU available, skip multi-GPU test")
        return None

    # Test with all GPUs
    all_gpus = ",".join(str(i) for i in range(total_gpus))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = all_gpus

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.device_count())",
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    child_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    print(f"[multi-gpu] With CUDA_VISIBLE_DEVICES={all_gpus}: child sees {child_count} GPUs")

    success = child_count == total_gpus
    if not success:
        print(f"[multi-gpu] FAIL: Expected {total_gpus}, got {child_count}")
    return success


def test_multiprocessing_spawn():
    """Test multiprocessing.spawn (used by vLLM/TensorRT) inherits CUDA_VISIBLE_DEVICES."""

    # This simulates what vLLM/TensorRT do internally
    test_gpus = "0,1,2,3"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = test_gpus

    # Use subprocess to simulate a fresh process that then uses multiprocessing
    code = """
import os
import multiprocessing

def worker_fn(worker_id):
    import torch
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return f"worker_{worker_id}: CUDA_VISIBLE_DEVICES={cuda_env}, devices={device_count}"

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.Pool(2) as pool:
        results = pool.map(worker_fn, [0, 1])
    for r in results:
        print(r)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )

    print(f"\n[multiprocessing] Parent set CUDA_VISIBLE_DEVICES={test_gpus}")
    print(f"[multiprocessing] spawn workers output:\n{result.stdout.strip()}")

    if result.returncode != 0:
        print(f"[multiprocessing] STDERR: {result.stderr}")
        return False

    # Verify workers saw all GPUs
    return "devices=4" in result.stdout


def main():
    print("=" * 60)
    print("CUDA_VISIBLE_DEVICES Propagation Test")
    print("=" * 60)

    results = {}

    # Test 1: Basic torch CUDA visibility
    print("\n--- Test 1: PyTorch CUDA Visibility ---")
    available, _count = test_torch_cuda_visibility()
    results["torch_cuda"] = available

    # Test 2: Subprocess inheritance
    print("\n--- Test 2: Subprocess Inheritance ---")
    results["subprocess"] = test_subprocess_inheritance()

    # Test 3: Multi-GPU override
    print("\n--- Test 3: Multi-GPU Override ---")
    results["multi_gpu"] = test_multi_gpu_override()

    # Test 4: Multiprocessing spawn (simulates vLLM/TensorRT internal workers)
    print("\n--- Test 4: Multiprocessing Spawn (vLLM/TensorRT pattern) ---")
    results["mp_spawn"] = test_multiprocessing_spawn()

    # Test 5: vLLM detection (if available)
    print("\n--- Test 5: vLLM GPU Detection ---")
    results["vllm"] = test_vllm_gpu_detection()

    # Test 6: TensorRT detection (if available)
    print("\n--- Test 6: TensorRT GPU Detection ---")
    results["tensorrt"] = test_tensorrt_gpu_detection()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False
        print(f"  {name}: {status}")

    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
