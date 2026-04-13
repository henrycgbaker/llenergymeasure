#!/usr/bin/env bash
# Benchmark Docker build times for documentation.
#
# Measures two scenarios:
#   1. Cold build     — no cache anywhere (--no-cache); simulates offline / first-ever build
#   2. First GHCR pull — fresh builder, layers pulled from GHCR registry cache
#
# Warm local rebuild is intentionally NOT benchmarked: it depends entirely on which
# layers changed and is documented in prose (only changed layers re-execute; stable
# expensive layers like FA3 are placed early in the Dockerfile by design).
#
# Usage:
#   scripts/benchmark_docker_builds.sh                         # GHCR pull only (fast)
#   scripts/benchmark_docker_builds.sh --cold                  # also cold build (slow!)
#   scripts/benchmark_docker_builds.sh --engines transformers  # single engine
#   scripts/benchmark_docker_builds.sh --engines vllm,tensorrt # subset
#
# Output: markdown suitable for pasting into docs/installation.md
#
# Requirements:
#   - llem-builder buildx builder (make docker-builder-setup)
#   - GHCR cache seeded (make docker-seed-transformers + CI published vllm/tensorrt)

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
ENGINES="transformers vllm tensorrt"
RUN_COLD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engines) ENGINES="${2//,/ }"; shift 2 ;;
        --cold)    RUN_COLD=true; shift ;;
        *)         echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BUILDER_NAME="${BUILDX_BUILDER:-llem-builder}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
cpu_model=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
cpu_cores=$(nproc)
ram_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo "unknown")
docker_ver=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
buildx_ver=$(docker buildx version 2>/dev/null | awk '{print $2}' || echo "unknown")
pkg_ver=$(python3 -c "from llenergymeasure._version import __version__; print(__version__)" 2>/dev/null || echo "unknown")
hostname_str=$(hostname -s 2>/dev/null || echo "unknown")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
fmt_time() { printf "%dm %02ds" $(($1 / 60)) $(($1 % 60)); }

builder_reset() {
    echo "  [resetting builder '${BUILDER_NAME}'...]" >&2
    docker buildx rm "$BUILDER_NAME" 2>/dev/null || true
    docker buildx create \
        --name "$BUILDER_NAME" \
        --driver docker-container \
        --driver-opt network=host \
        --buildkitd-flags '--allow-insecure-entitlement network.host' \
        >/dev/null 2>&1 || true
    docker buildx use "$BUILDER_NAME"
}

run_build() {
    # run_build <engine> [extra compose flags]
    local engine=$1; shift
    local log="/tmp/llem-bench-${engine}-$$.log"
    local t0 t1
    t0=$(date +%s)
    BUILDKIT_PROGRESS=plain docker compose build "$@" "$engine" >"$log" 2>&1
    t1=$(date +%s)
    local secs=$(( t1 - t0 ))
    local cached; cached=$(grep -cE "^#[0-9]+ CACHED$" "$log" 2>/dev/null || true)
    local imported; imported=$(grep -c "importing cache manifest from ghcr.io" "$log" 2>/dev/null || true)
    echo "$secs $cached $imported"
}

image_size() {
    docker image inspect "llenergymeasure:${1}" --format '{{.Size}}' 2>/dev/null \
        | awk '{printf "%.1f GB", $1/1073741824}' || echo "not built"
}

# ---------------------------------------------------------------------------
# Storage for results
# ---------------------------------------------------------------------------
declare -A COLD_SECS GHCR_SECS GHCR_LAYERS SIZE

# ---------------------------------------------------------------------------
# Scenario 2: First GHCR pull  (always run)
# ---------------------------------------------------------------------------
echo "" >&2
echo "=== Scenario 2: First GHCR pull ===" >&2
for engine in $ENGINES; do
    echo "  $engine..." >&2
    builder_reset
    read -r secs cached imported <<< "$(run_build "$engine")"
    GHCR_SECS[$engine]=$secs
    GHCR_LAYERS[$engine]=$cached
    echo "  → $(fmt_time $secs), ${cached} layers reused, registry_hit=${imported}" >&2
done

# ---------------------------------------------------------------------------
# Scenario 1: Cold build  (opt-in — slow)
# ---------------------------------------------------------------------------
if $RUN_COLD; then
    echo "" >&2
    echo "=== Scenario 1: Cold build (--no-cache) ===" >&2
    for engine in $ENGINES; do
        echo "  $engine  (this may take 20-60+ min for transformers)..." >&2
        builder_reset
        read -r secs cached imported <<< "$(run_build "$engine" "--no-cache")"
        COLD_SECS[$engine]=$secs
        echo "  → $(fmt_time $secs)" >&2
    done
fi

# ---------------------------------------------------------------------------
# Image sizes  (post-build)
# ---------------------------------------------------------------------------
for engine in $ENGINES; do
    SIZE[$engine]=$(image_size "$engine")
done

# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------
echo ""
echo "<!-- benchmark output — paste into docs/installation.md -->"
echo ""
echo "Measured on **\`$hostname_str\`** — $cpu_model, $cpu_cores cores,"
echo "${ram_gb} GB RAM — Docker $docker_ver / Buildx $buildx_ver / llenergymeasure $pkg_ver"
echo ""

if $RUN_COLD; then
    echo "| Engine | Image size | Cold build | First GHCR pull |"
    echo "|--------|-----------|------------|-----------------|"
    for engine in $ENGINES; do
        echo "| ${engine^} | ${SIZE[$engine]} | $(fmt_time ${COLD_SECS[$engine]:-0}) | $(fmt_time ${GHCR_SECS[$engine]}) (${GHCR_LAYERS[$engine]} layers reused) |"
    done
else
    echo "| Engine | Image size | First GHCR pull |"
    echo "|--------|-----------|-----------------|"
    for engine in $ENGINES; do
        echo "| ${engine^} | ${SIZE[$engine]} | $(fmt_time ${GHCR_SECS[$engine]}) (${GHCR_LAYERS[$engine]} layers reused) |"
    done
    echo ""
    echo "> Cold build times not included in this run (use \`--cold\` to measure;"
    echo "> expect 20-60+ min for Transformers, hardware-dependent)."
fi

echo ""
echo "**Scenarios:**"
echo ""
echo "- **Cold build** — no cache anywhere; simulates a new machine with GHCR offline or"
echo "  the very first build before any cache exists. Time is dominated by flash-attn FA3"
echo "  Hopper compilation (~80 nvcc invocations at MAX_JOBS=$(nproc))."
echo "- **First GHCR pull** — fresh builder, all layers pulled from the GHCR registry cache"
echo "  seeded by CI on each release. What a new developer gets after \`make docker-builder-setup\`."
echo "- **Warm local rebuild** — after the first build, only layers whose inputs changed"
echo "  are re-executed. The Dockerfile places stable expensive layers (FA3 compile, base"
echo "  deps) before application code by design, so a source-only edit typically rebuilds"
echo "  in seconds regardless of engine."
