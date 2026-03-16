#!/bin/bash
# killpg verification: confirms that sending SIGINT to a running vLLM study
# releases GPU memory after the process group is killed via os.killpg().
#
# Validates the os.killpg fix (runner.py _kill_process_group) end-to-end:
#   1. Records baseline GPU memory
#   2. Launches llem run with a vLLM config (facebook/opt-125m)
#   3. Polls until GPU memory exceeds baseline + threshold (model loaded)
#   4. Sends SIGINT — triggers _sigint_handler -> _kill_process_group(SIGTERM)
#   5. Asserts GPU memory returns to baseline (all vLLM workers released GPU)
#
# Requires: GPU hardware (CUDA), facebook/opt-125m cached, llem installed
# Runs in: Tier 2 GPU CI inside vLLM Docker container (ds01-gpu)
# Must be run from repository root (or /app inside container).
set -euo pipefail

STUDY_YAML="tests/fixtures/killpg_vllm_study.yaml"
RESULTS_DIR="results"
# Seconds to wait for GPU memory to increase (vLLM loads slower than PyTorch)
POLL_TIMEOUT=180
# Minimum GPU memory increase in MB to confirm model has loaded
MEMORY_THRESHOLD_MB=50
# Seconds to wait for GPU memory to return to baseline after kill
MEMORY_RETURN_TIMEOUT=30
# Delta tolerance in MB: current <= baseline + tolerance counts as "returned"
MEMORY_TOLERANCE_MB=100

# Force local runner — inside a container, we already have CUDA.
export LLEM_RUNNER_VLLM=local

# Use llem directly if available (container), else uv run (host)
if command -v llem &>/dev/null; then
    LLEM_CMD="llem"
else
    LLEM_CMD="uv run llem"
fi

echo "=== killpg Verification ==="
echo "Config: $STUDY_YAML"
echo "Results dir: $RESULTS_DIR"
echo "Runner: $LLEM_CMD"

# Helper: read GPU 0 memory usage in MB via pynvml
_gpu_memory_mb() {
    python3 -c "
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(h)
print(info.used // (1024 * 1024))
pynvml.nvmlShutdown()
"
}

# --- Step 1: Record baseline GPU memory ---
BASELINE_MB=$(_gpu_memory_mb)
echo "Baseline GPU memory: ${BASELINE_MB} MB"

mkdir -p "$RESULTS_DIR"

# --- Step 2: Launch llem run in background ---
$LLEM_CMD run "$STUDY_YAML" &
LLEM_PID=$!
echo "Started llem (PID $LLEM_PID)"

# Cleanup trap: kill llem if script exits unexpectedly before we do so ourselves
_cleanup() {
    kill -9 "$LLEM_PID" 2>/dev/null || true
    wait "$LLEM_PID" 2>/dev/null || true
    rm -rf results/killpg-test* 2>/dev/null || true
}
trap _cleanup EXIT

# --- Step 3: Poll until GPU memory exceeds baseline + threshold ---
POLL_ELAPSED=0
MODEL_LOADED=false
while [ $POLL_ELAPSED -lt $POLL_TIMEOUT ]; do
    # Check if llem has already exited (unexpected early exit)
    if ! kill -0 "$LLEM_PID" 2>/dev/null; then
        echo "FAIL: llem process exited unexpectedly before model loaded (elapsed ${POLL_ELAPSED}s)"
        trap - EXIT
        exit 1
    fi

    CURRENT_MB=$(_gpu_memory_mb)
    INCREASE=$((CURRENT_MB - BASELINE_MB))
    if [ $INCREASE -gt $MEMORY_THRESHOLD_MB ]; then
        echo "Model loaded at ${POLL_ELAPSED}s (memory +${INCREASE} MB, total ${CURRENT_MB} MB)"
        MODEL_LOADED=true
        break
    fi
    echo "  Waiting for model load... (memory +${INCREASE} MB at ${POLL_ELAPSED}s)"
    sleep 2
    POLL_ELAPSED=$((POLL_ELAPSED + 2))
done

if [ "$MODEL_LOADED" = "false" ]; then
    CURRENT_MB=$(_gpu_memory_mb)
    echo "FAIL: Model did not load within ${POLL_TIMEOUT}s (current: ${CURRENT_MB} MB, baseline: ${BASELINE_MB} MB)"
    # cleanup trap will kill llem
    exit 1
fi

# --- Step 4: Send SIGINT to trigger os.killpg via _sigint_handler ---
# This exercises the exact code path: _sigint_handler -> _kill_process_group(pid, SIGTERM)
# The worker called os.setpgrp(), so killpg sends SIGTERM to the whole process group
# (vLLM workers, ray workers, etc.) — not just the parent.
echo "Sending SIGINT to PID $LLEM_PID"
kill -INT "$LLEM_PID"

# Wait for process to exit and capture exit code
set +e
wait "$LLEM_PID"
EXIT_CODE=$?
set -e

echo "Exit code: $EXIT_CODE"

# Disarm the EXIT trap — llem is already gone
trap - EXIT

# Assertion 1: Exit code must be 130 (SIGINT convention: 128 + 2)
if [ "$EXIT_CODE" -ne 130 ]; then
    echo "FAIL: Expected exit code 130, got $EXIT_CODE"
    exit 1
fi
echo "PASS: Exit code is 130"

# --- Step 5: Assert GPU memory returns to baseline ---
# Give vLLM workers a moment to release GPU memory after SIGTERM/SIGKILL.
RETURN_ELAPSED=0
MEMORY_RETURNED=false
while [ $RETURN_ELAPSED -lt $MEMORY_RETURN_TIMEOUT ]; do
    CURRENT_MB=$(_gpu_memory_mb)
    DELTA=$((CURRENT_MB - BASELINE_MB))
    if [ $DELTA -le $MEMORY_TOLERANCE_MB ]; then
        echo "PASS: GPU memory returned to baseline at ${RETURN_ELAPSED}s (delta ${DELTA} MB, current ${CURRENT_MB} MB)"
        MEMORY_RETURNED=true
        break
    fi
    echo "  Waiting for memory release... (delta ${DELTA} MB at ${RETURN_ELAPSED}s)"
    sleep 2
    RETURN_ELAPSED=$((RETURN_ELAPSED + 2))
done

if [ "$MEMORY_RETURNED" = "false" ]; then
    CURRENT_MB=$(_gpu_memory_mb)
    DELTA=$((CURRENT_MB - BASELINE_MB))
    echo "FAIL: GPU memory did not return to baseline within ${MEMORY_RETURN_TIMEOUT}s"
    echo "  Baseline: ${BASELINE_MB} MB, Current: ${CURRENT_MB} MB, Delta: ${DELTA} MB (tolerance: ${MEMORY_TOLERANCE_MB} MB)"
    echo "  os.killpg may not have cleaned up all vLLM worker processes."
    exit 1
fi

# --- Cleanup ---
rm -rf results/killpg-test* 2>/dev/null || true

echo "=== killpg Verification PASSED ==="
