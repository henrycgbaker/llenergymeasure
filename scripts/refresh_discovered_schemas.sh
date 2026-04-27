#!/usr/bin/env bash
# Rediscover a vendored engine schema by running discovery inside the
# appropriate Docker image.
#
# Usage: ./scripts/refresh_discovered_schemas.sh {vllm|tensorrt|transformers}
#
# Always writes to src/llenergymeasure/config/discovered_schemas/<engine>.json
# and prints the resulting `git diff`. Does NOT commit. The vendored JSON
# file IS the canonical SSOT — authority comes from `git commit`, not from
# who ran discovery.
#
# Legitimate refresh (e.g. you bumped a Dockerfile FROM tag):
#   review the diff, `git add`, and open a PR.
# Exploring a fork or stale image:
#   `git checkout src/llenergymeasure/config/discovered_schemas/<engine>.json`
#
# Discovery image selection:
#   vllm         -> pristine vllm/vllm-openai:<tag> (vllm pre-installed)
#   tensorrt     -> pristine nvcr.io/nvidia/tensorrt-llm/release:<tag>
#                   (works around llenergymeasure:tensorrt's cuKernelGetName bug)
#   transformers -> llenergymeasure:transformers-<ver> (base pytorch image has no
#                   transformers package; our Dockerfile pip-installs it at the
#                   version pinned by ARG TRANSFORMERS_VERSION)
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/refresh_discovered_schemas.sh {vllm|tensorrt|transformers}

Builds or pulls the engine's discovery image, runs discovery inside it,
writes src/llenergymeasure/config/discovered_schemas/<engine>.json, and
prints the git diff. Does NOT commit.
EOF
}

if [[ $# -ne 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage >&2
    exit 1
fi

ENGINE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Extract ARG default value from a Dockerfile: _arg_default <file> <NAME>
_arg_default() {
    grep -oE "^ARG[[:space:]]+${2}=[^[:space:]]+" "$1" | head -1 | cut -d= -f2-
}

case "$ENGINE" in
    vllm)
        VER="$(_arg_default "$REPO_ROOT/docker/Dockerfile.vllm" VLLM_VERSION)"
        IMAGE="vllm/vllm-openai:${VER}"
        ;;
    tensorrt)
        VER="$(_arg_default "$REPO_ROOT/docker/Dockerfile.tensorrt" TRTLLM_VERSION)"
        IMAGE="nvcr.io/nvidia/tensorrt-llm/release:${VER}"
        ;;
    transformers)
        VER="$(_arg_default "$REPO_ROOT/docker/Dockerfile.transformers" TRANSFORMERS_VERSION)"
        IMAGE="llenergymeasure:transformers-${VER}"
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            echo "[$ENGINE] Image $IMAGE not found; building from docker/Dockerfile.transformers..." >&2
            docker build -f "$REPO_ROOT/docker/Dockerfile.transformers" -t "$IMAGE" "$REPO_ROOT"
        fi
        ;;
    *)
        echo "Unknown engine: $ENGINE" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ -z "${IMAGE:-}" ]]; then
    echo "Failed to resolve image for engine '$ENGINE'" >&2
    exit 1
fi

OUTPUT_REL="src/llenergymeasure/config/discovered_schemas/${ENGINE}.json"

echo "[$ENGINE] Running discovery inside $IMAGE..." >&2
# Forward LLENERGY_DISCOVERY_FROZEN_AT into the container if the caller (CI)
# set it. discover_engine_schemas.py uses it to pin `discovered_at` to a
# stable anchor, breaking the writeback->resync loop on unchanged source.
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --entrypoint python3 \
    -e LLENERGY_DISCOVERY_FROZEN_AT="${LLENERGY_DISCOVERY_FROZEN_AT:-}" \
    -v "$REPO_ROOT:/repo" \
    -w /repo \
    "$IMAGE" \
    scripts/discover_engine_schemas.py "$ENGINE" \
    --image-ref "$IMAGE" \
    --output "/repo/$OUTPUT_REL"

cd "$REPO_ROOT"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "[$ENGINE] Not inside a git repo — skipping diff output." >&2
    exit 0
fi

if git diff --quiet -- "$OUTPUT_REL" 2>/dev/null; then
    if [[ -z "$(git status --porcelain -- "$OUTPUT_REL")" ]]; then
        echo "[$ENGINE] No changes to vendored schema." >&2
        exit 0
    fi
fi

echo "" >&2
echo "=== git diff --stat $OUTPUT_REL ===" >&2
git diff --stat -- "$OUTPUT_REL" || true
echo "" >&2
echo "=== git diff $OUTPUT_REL (first 200 lines) ===" >&2
git --no-pager diff -- "$OUTPUT_REL" | head -200 || true
echo "" >&2
cat <<EOF >&2
Schema changed.
  - Legitimate refresh? Review the diff, \`git add $OUTPUT_REL\`, and open a PR.
  - Exploring a custom fork or stale image? Revert with:
      git checkout -- $OUTPUT_REL
EOF
