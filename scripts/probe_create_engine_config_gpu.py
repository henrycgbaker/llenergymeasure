"""PoC-G: Does EngineArgs.create_engine_config() actually require GPU?

Hypothesis
----------
§4.3 of runtime-config-validation.md proposes splitting vendor CI cases
between GH-hosted (CPU) and self-hosted (GPU) runners based on a
`requires_gpu: bool` field per rule. The classification rests on the
assumption that `EngineArgs.create_engine_config()` requires CUDA while
pure `SamplingParams.__post_init__` does not. That assumption was never
empirically tested on this host.

This PoC:
  1. Constructs EngineArgs with a minimal kwargs set.
  2. Tries calling create_engine_config().
  3. Reports whether it fails without CUDA (confirms GPU-required) or
     succeeds (invalidates the split assumption; more cases go CPU).
  4. Repeats for TRT-LLM LlmArgs as the parallel case.

Decision criteria (pre-committed)
---------------------------------
- create_engine_config() raises CUDA / NVML error without GPU
    -> §4.3's CPU/GPU split is justified; requires_gpu=true for
       resolution_call cases.
- create_engine_config() succeeds without GPU (as long as we mask
  tensor_parallel_size and GPU count)
    -> More cases can go to GH-hosted CPU runner. Only cases explicitly
       invoking CUDA queries need self-hosted.
- create_engine_config() raises something else (import error, missing
  class, API mismatch)
    -> Report and move on; may indicate a vLLM version incompatibility
       with the host install.

Run
---
  /usr/bin/python3.10 scripts/probe_create_engine_config_gpu.py

Run context: host has NO CUDA runtime (torch.cuda.is_available() = False),
but pynvml CAN see 4x A100. This lets us distinguish:
  - Failures from CUDA runtime absence (PASS on self-hosted GPU CI)
  - Failures from pure-Python pre-conditions (FAIL regardless of runner)
"""

from __future__ import annotations

import traceback

import torch

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print("Python environment reports no CUDA runtime on host (expected).")
print()


# --- Test 1: vLLM EngineArgs construction + create_engine_config ---
print("=" * 78)
print("vLLM EngineArgs.create_engine_config()")
print("=" * 78)

try:
    from vllm import EngineArgs
except ImportError as e:
    print(f"vllm not importable: {e}")
    print()
else:
    # Minimal kwargs — avoid anything that needs GPU
    test_cases = [
        {
            "label": "minimal (local model path, bfloat16, tp=1)",
            "kwargs": {
                "model": "Qwen/Qwen2.5-0.5B",
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "max_model_len": 256,
                "gpu_memory_utilization": 0.5,
                "enforce_eager": True,
            },
        },
        {
            "label": "float32 (should be rejected by vLLM)",
            "kwargs": {
                "model": "Qwen/Qwen2.5-0.5B",
                "dtype": "float32",
                "tensor_parallel_size": 1,
                "max_model_len": 256,
            },
        },
        {
            "label": "tp_size=8 (exceeds available GPUs even on GPU runner)",
            "kwargs": {
                "model": "Qwen/Qwen2.5-0.5B",
                "dtype": "bfloat16",
                "tensor_parallel_size": 8,
                "max_model_len": 256,
            },
        },
    ]

    for tc in test_cases:
        label = tc["label"]
        kwargs = tc["kwargs"]
        print(f"\n--- CASE: {label} ---")
        print(f"    kwargs: {kwargs}")
        try:
            engine_args = EngineArgs(**kwargs)
            print("  EngineArgs construction: OK (no CUDA needed)")
        except Exception as e:
            print(f"  EngineArgs construction RAISED: {type(e).__name__}: {str(e)[:150]}")
            continue

        try:
            cfg = engine_args.create_engine_config()
            print(f"  create_engine_config: **SUCCEEDED without CUDA** (obj: {type(cfg).__name__})")
            print("    => This case could run on GH-hosted CPU runner")
        except ImportError as e:
            print(f"  create_engine_config RAISED ImportError (missing CUDA): {str(e)[:100]}")
            print("    => GPU runner needed for this case")
        except AssertionError as e:
            print(f"  create_engine_config RAISED AssertionError: {str(e)[:120]}")
            # Could be either "AssertionError: CUDA not available" or a pre-CUDA check
            if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                print("    => GPU runner needed (CUDA/GPU assertion)")
            else:
                print("    => Pure-Python assertion; fails regardless of runner")
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  create_engine_config RAISED: {err}")
            lowered = err.lower()
            if "cuda" in lowered or "gpu" in lowered or "device" in lowered:
                print("    => CUDA/GPU-related; GPU runner needed")
            elif "tensor_parallel" in lowered or "available_gpu" in lowered:
                print("    => NVML-queryable runtime state; GPU runner needed")
            else:
                print("    => Pure-Python failure; fails regardless of runner")
                print("    TRACEBACK:")
                traceback.print_exc()

print()
print("=" * 78)
print("TRT-LLM LlmArgs (parallel case)")
print("=" * 78)

try:
    from tensorrt_llm.llmapi import BuildConfig, LlmArgs  # noqa: F401
except Exception as e:
    print(f"tensorrt_llm not importable on host: {e}")
    print("-> Skip; rerun inside NGC container for validation")
else:
    print("TRT-LLM is importable — would run test cases here.")
    # Not exercised; fill in if/when TRT-LLM is usable on host.

print()
print("=" * 78)
print("DECISION CRITERIA EVALUATION")
print("=" * 78)
print()
print("Look at each CASE above:")
print("  - 'SUCCEEDED without CUDA' -> case is CPU-runnable; requires_gpu=false")
print("    This shifts cases from self-hosted to GH-hosted, reducing CI cost.")
print("  - 'GPU runner needed' -> requires_gpu=true; assigned to self-hosted.")
print("  - 'Pure-Python failure' -> the rule being tested doesn't need runtime")
print("    at all; the failure IS the rule's expected outcome. Mark as CPU.")
print()
print("Doc implication for §4.3:")
print("  If minimal-kwargs create_engine_config() succeeds on CPU, revise the")
print("  rule table: requires_gpu is true only for cases specifically invoking")
print("  hardware-dependent paths (FP8 SM check, TP size vs available GPUs).")
