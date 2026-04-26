#!/usr/bin/env bash
# Re-vendor a vendored-rules JSON by running scripts/vendor_rules.py inside the
# appropriate Docker image.
#
# Usage: ./scripts/update_engine_rules.sh {transformers|vllm|tensorrt}
#
# Mirror of scripts/update_engine_schema.sh — same idioms (Dockerfile ARG
# lookup, image build fallback, diff-only-no-commit output) — different
# artifact. Run this locally to re-vendor against the pinned image before
# opening a PR; CI will re-run inside the same image on the PR branch.
#
# Output: src/llenergymeasure/config/vendored_rules/<engine>.json
# The JSON IS the canonical SSOT — authority comes from `git commit`, not
# from who ran vendoring.
#
# Legitimate refresh (e.g. you bumped a Dockerfile FROM tag):
#   review the diff, `git add`, and open a PR.
# Exploring a fork or stale image:
#   `git checkout src/llenergymeasure/config/vendored_rules/<engine>.json`
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/update_engine_rules.sh {transformers|vllm|tensorrt}

Builds or pulls the engine's Docker image, runs scripts/vendor_rules.py inside
it against the tracked corpus, writes the JSON envelope, and prints the git
diff. Does NOT commit.
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
    transformers)
        VER="$(_arg_default "$REPO_ROOT/docker/Dockerfile.transformers" TRANSFORMERS_VERSION)"
        IMAGE="llenergymeasure:transformers-${VER}"
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            echo "[$ENGINE] Image $IMAGE not found; building from docker/Dockerfile.transformers..." >&2
            docker build -f "$REPO_ROOT/docker/Dockerfile.transformers" -t "$IMAGE" "$REPO_ROOT"
        fi
        ;;
    vllm)
        VER="$(_arg_default "$REPO_ROOT/docker/Dockerfile.vllm" VLLM_VERSION)"
        IMAGE="vllm/vllm-openai:${VER}"
        ;;
    tensorrt)
        VER="$(_arg_default "$REPO_ROOT/docker/Dockerfile.tensorrt" TRTLLM_VERSION)"
        IMAGE="nvcr.io/nvidia/tensorrt-llm/release:${VER}"
        ;;
    *)
        echo "Unknown engine: $ENGINE" >&2
        usage >&2
        exit 1
        ;;
esac

CORPUS_REL="configs/validation_rules/${ENGINE}.yaml"
OUTPUT_REL="src/llenergymeasure/config/vendored_rules/${ENGINE}.json"

if [[ ! -f "$REPO_ROOT/$CORPUS_REL" ]]; then
    echo "[$ENGINE] Corpus $CORPUS_REL not found. Run the miner first:" >&2
    echo "    python -m scripts.miners.${ENGINE}_miner --out $CORPUS_REL" >&2
    exit 1
fi

echo "[$ENGINE] Running vendor_rules.py inside $IMAGE..." >&2
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --entrypoint python3 \
    -v "$REPO_ROOT:/repo" \
    -w /repo \
    "$IMAGE" \
    scripts/vendor_rules.py \
    --engine "$ENGINE" \
    --corpus "/repo/$CORPUS_REL" \
    --out "/repo/$OUTPUT_REL" \
    --image-ref "$IMAGE" \
    --base-image-ref "$IMAGE"

cd "$REPO_ROOT"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "[$ENGINE] Not inside a git repo — skipping diff output." >&2
    exit 0
fi

if git diff --quiet -- "$OUTPUT_REL" 2>/dev/null; then
    if [[ -z "$(git status --porcelain -- "$OUTPUT_REL")" ]]; then
        echo "[$ENGINE] No changes to vendored rules JSON." >&2
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
Vendored rules changed.
  - Legitimate refresh? Review the diff, \`git add $OUTPUT_REL\`, and open a PR.
  - Exploring a custom fork or stale image? Revert with:
      git checkout -- $OUTPUT_REL
EOF
