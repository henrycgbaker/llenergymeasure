#!/usr/bin/env bash
#
# DEPRECATED: This script has been superseded by runtime-test-orchestrator.py
#
# The new orchestrator uses SSOT introspection to discover ALL params and
# dispatches each test to the correct engine container (pytorch, vllm, tensorrt).
#
# Usage:
#   python scripts/runtime-test-orchestrator.py               # All engines
#   python scripts/runtime-test-orchestrator.py --engine pytorch
#   python scripts/runtime-test-orchestrator.py --quick       # Quick mode
#   python scripts/runtime-test-orchestrator.py --list-params # List params
#   python scripts/runtime-test-orchestrator.py --check-docker
#
# Or via make:
#   make test-runtime       # PyTorch engine
#   make test-runtime-all   # All engines
#   make test-runtime-quick # Quick mode
#
# This wrapper script is kept for backwards compatibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "NOTE: run-runtime-tests.sh is deprecated. Using runtime-test-orchestrator.py"
echo ""

# Forward all arguments to the Python orchestrator
python "${SCRIPT_DIR}/runtime-test-orchestrator.py" "$@"
