#!/bin/bash
# SIGINT verification: confirms that sending SIGINT to a running study
# preserves the manifest file with status 'interrupted' and exits with code 130.
#
# Requires: GPU hardware (CUDA), gpt2 model cached, llem installed
# Runs in: Tier 2 GPU CI inside Docker container (ds01-gpu)
# Must be run from repository root (or /app inside container).
set -euo pipefail

STUDY_YAML="tests/fixtures/sigint_study.yaml"
RESULTS_DIR="results"
# Seconds to wait for manifest to appear before sending SIGINT
POLL_TIMEOUT=120

# Force local runner — inside a container, we already have CUDA.
export LLEM_RUNNER_PYTORCH=local

# Use llem directly if available (container), else uv run (host)
if command -v llem &>/dev/null; then
    LLEM_CMD="llem"
else
    LLEM_CMD="uv run llem"
fi

echo "=== SIGINT Verification ==="
echo "Config: $STUDY_YAML"
echo "Results dir: $RESULTS_DIR"
echo "Runner: $LLEM_CMD"

# Record start time for finding the manifest created by this run
START_EPOCH=$(date +%s)

# Launch llem run in background (study path: manifest written to results/)
$LLEM_CMD run "$STUDY_YAML" &
LLEM_PID=$!
echo "Started llem (PID $LLEM_PID)"

# Poll for manifest.json creation in results/, looking for files newer than start time.
# The manifest is created immediately when the study runner initialises (before any
# experiment runs), so a short poll timeout is sufficient.
POLL_ELAPSED=0
MANIFEST=""
while [ $POLL_ELAPSED -lt $POLL_TIMEOUT ]; do
    # Find manifest.json files newer than our start marker
    MANIFEST=$(find "$RESULTS_DIR" -name "manifest.json" -newer /proc/$$/status 2>/dev/null | head -1 || true)
    if [ -n "$MANIFEST" ]; then
        echo "Manifest detected at ${POLL_ELAPSED}s: $MANIFEST"
        # Give a few more seconds for the first experiment to start
        sleep 5
        break
    fi
    sleep 2
    POLL_ELAPSED=$((POLL_ELAPSED + 2))
done

if [ -z "$MANIFEST" ]; then
    echo "FAIL: No manifest.json found in $RESULTS_DIR after ${POLL_TIMEOUT}s"
    kill "$LLEM_PID" 2>/dev/null || true
    wait "$LLEM_PID" 2>/dev/null || true
    exit 1
fi

# Send SIGINT to the llem process
echo "Sending SIGINT to PID $LLEM_PID"
kill -INT "$LLEM_PID"

# Wait for process to exit and capture exit code
set +e
wait "$LLEM_PID"
EXIT_CODE=$?
set -e

echo "Exit code: $EXIT_CODE"

# Assertion 1: Exit code must be 130
if [ "$EXIT_CODE" -ne 130 ]; then
    echo "FAIL: Expected exit code 130, got $EXIT_CODE"
    exit 1
fi
echo "PASS: Exit code is 130"

# Assertion 2: Manifest file must still exist
if [ ! -f "$MANIFEST" ]; then
    echo "FAIL: manifest.json not found after SIGINT at $MANIFEST"
    exit 1
fi
echo "PASS: manifest.json exists at $MANIFEST"

# Assertion 3: Manifest status must be 'interrupted'
python3 -c "
import json, sys
with open('$MANIFEST') as f:
    m = json.load(f)
status = m.get('status', 'MISSING')
if status != 'interrupted':
    print(f'FAIL: Expected status=interrupted, got status={status}')
    sys.exit(1)
print(f'PASS: Manifest status is interrupted')
"

# Cleanup: remove the study directory created by this run
STUDY_DIR=$(dirname "$MANIFEST")
echo "Cleaning up study directory: $STUDY_DIR"
rm -rf "$STUDY_DIR"

echo "=== SIGINT Verification PASSED ==="
