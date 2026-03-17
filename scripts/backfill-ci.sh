#!/usr/bin/env bash
# backfill-ci.sh — Re-trigger CI for commits rewritten by filter-branch.
#
# Prerequisites:
#   - The workflow_dispatch trigger must be merged to main first
#   - gh CLI authenticated with repo access
#   - backup/main-before-rebase branch exists locally
#
# Usage:
#   ./scripts/backfill-ci.sh tag       # Phase 1: create + push temp tags
#   ./scripts/backfill-ci.sh trigger   # Phase 2: trigger CI for each tag
#   ./scripts/backfill-ci.sh status    # Phase 3: check run status
#   ./scripts/backfill-ci.sh cleanup   # Phase 4: delete temp tags
#   ./scripts/backfill-ci.sh all       # Run phases 1-2 sequentially

set -euo pipefail

TAG_PREFIX="ci-rerun"
WORKFLOW="ci.yml"
BACKUP_REF="backup/main-before-rebase"
THROTTLE_SECONDS=30  # delay between workflow triggers
MAX_PARALLEL=3       # concurrent CI runs to allow

# ── Helpers ──────────────────────────────────────────────────────────────

affected_shas() {
    comm -23 \
        <(git log --format='%H' main | sort) \
        <(git log --format='%H' "$BACKUP_REF" | sort)
}

tag_name() {
    echo "${TAG_PREFIX}/${1:0:7}"
}

count_running() {
    gh run list --workflow="$WORKFLOW" --json status \
        --jq '[.[] | select(.status == "in_progress" or .status == "queued")] | length'
}

# ── Phase 1: Create and push tags ────────────────────────────────────────

cmd_tag() {
    echo "Creating tags for affected commits..."
    local count=0
    while IFS= read -r sha; do
        local tag
        tag=$(tag_name "$sha")
        if git rev-parse "refs/tags/$tag" >/dev/null 2>&1; then
            echo "  skip (exists): $tag"
        else
            git tag "$tag" "$sha"
            count=$((count + 1))
        fi
    done < <(affected_shas)

    echo "Created $count tags locally."
    echo "Pushing tags to origin..."
    git push origin "refs/tags/${TAG_PREFIX}/*" 2>&1 || true
    echo "Done. $(git tag -l "${TAG_PREFIX}/*" | wc -l) tags on remote."
}

# ── Phase 2: Trigger CI ──────────────────────────────────────────────────

cmd_trigger() {
    local tags
    tags=$(git tag -l "${TAG_PREFIX}/*" | sort)
    local total
    total=$(echo "$tags" | wc -l)
    local i=0

    echo "Triggering CI for $total tags (throttle: ${THROTTLE_SECONDS}s, max parallel: $MAX_PARALLEL)..."

    for tag in $tags; do
        i=$((i + 1))

        # Throttle: wait if too many runs are active
        while true; do
            local running
            running=$(count_running)
            if [ "$running" -lt "$MAX_PARALLEL" ]; then
                break
            fi
            echo "  ($running runs active, waiting 30s...)"
            sleep 30
        done

        echo "  [$i/$total] gh workflow run $WORKFLOW --ref $tag"
        gh workflow run "$WORKFLOW" --ref "$tag" || {
            echo "  WARN: failed to trigger for $tag, continuing..."
            continue
        }
        sleep "$THROTTLE_SECONDS"
    done

    echo "All triggers dispatched."
}

# ── Phase 3: Status ─────────────────────────────────────────────────────

cmd_status() {
    echo "CI run status for ${TAG_PREFIX}/* refs:"
    echo ""

    # Summary counts
    gh run list --workflow="$WORKFLOW" --limit=100 --json status,conclusion,headBranch \
        --jq '
            [.[] | select(.headBranch | startswith("refs/tags/'"$TAG_PREFIX"'") or startswith("'"$TAG_PREFIX"'"))] as $runs |
            if ($runs | length) == 0 then
                "No matching runs found. Runs may take a moment to appear."
            else
                "Total: \($runs | length)\n" +
                "  Completed: \([$runs[] | select(.status == "completed")] | length)\n" +
                "    Success: \([$runs[] | select(.conclusion == "success")] | length)\n" +
                "    Failure: \([$runs[] | select(.conclusion == "failure")] | length)\n" +
                "  In progress: \([$runs[] | select(.status == "in_progress")] | length)\n" +
                "  Queued: \([$runs[] | select(.status == "queued")] | length)"
            end
        '

    echo ""
    echo "Failed runs (if any):"
    gh run list --workflow="$WORKFLOW" --limit=100 --json databaseId,headBranch,conclusion \
        --jq '[.[] | select(.conclusion == "failure") | select(.headBranch | startswith("refs/tags/'"$TAG_PREFIX"'") or startswith("'"$TAG_PREFIX"'"))] | .[] | "\(.databaseId) \(.headBranch)"' \
        || echo "  (none or not yet available)"
}

# ── Phase 4: Cleanup ────────────────────────────────────────────────────

cmd_cleanup() {
    local tags
    tags=$(git tag -l "${TAG_PREFIX}/*")
    local count
    count=$(echo "$tags" | grep -c . || true)

    if [ "$count" -eq 0 ]; then
        echo "No ${TAG_PREFIX}/* tags to clean up."
        return
    fi

    echo "Deleting $count remote tags..."
    echo "$tags" | xargs -I{} git push origin --delete "refs/tags/{}" 2>&1 || true

    echo "Deleting $count local tags..."
    echo "$tags" | xargs git tag -d 2>&1 || true

    echo "Cleanup complete."
}

# ── Phase: All ───────────────────────────────────────────────────────────

cmd_all() {
    cmd_tag
    echo ""
    cmd_trigger
    echo ""
    echo "Triggers dispatched. Use './scripts/backfill-ci.sh status' to monitor."
    echo "When all runs complete, use './scripts/backfill-ci.sh cleanup' to remove temp tags."
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-help}" in
    tag)     cmd_tag ;;
    trigger) cmd_trigger ;;
    status)  cmd_status ;;
    cleanup) cmd_cleanup ;;
    all)     cmd_all ;;
    *)
        echo "Usage: $0 {tag|trigger|status|cleanup|all}"
        echo ""
        echo "  tag      Create + push temporary tags for 68 affected commits"
        echo "  trigger  Dispatch CI workflow for each tag"
        echo "  status   Check progress of backfill runs"
        echo "  cleanup  Delete temporary tags (local + remote)"
        echo "  all      Run tag + trigger sequentially"
        exit 1
        ;;
esac
