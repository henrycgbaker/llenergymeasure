#!/usr/bin/env bash
# backfill-ci.sh — Copy CI check results from old SHAs to rewritten SHAs.
#
# Uses a GitHub Actions workflow with GITHUB_TOKEN to create official
# check runs (appearing as "GitHub Actions", not personal account).
#
# Prerequisites:
#   - ci-backfill.yml workflow merged to main
#   - gh CLI authenticated with repo scope
#   - backup/main-before-rebase branch exists locally
#
# Usage:
#   ./scripts/backfill-ci.sh preview   # Show mapping + what will be stamped
#   ./scripts/backfill-ci.sh run       # Trigger the backfill workflow
#   ./scripts/backfill-ci.sh status    # Check workflow run status
#   ./scripts/backfill-ci.sh verify    # Verify check runs on new SHAs
#   ./scripts/backfill-ci.sh cleanup   # Delete ci-rerun/* tags

set -euo pipefail

REPO="henrycgbaker/LLenergyMeasure"
BACKUP_REF="backup/main-before-rebase"
WORKFLOW="ci-backfill.yml"
TAG_PREFIX="ci-rerun"

# ── Helpers ──────────────────────────────────────────────────────────────

# Build JSON array of {old_sha, new_sha} pairs
build_json_mapping() {
    local pairs="["
    local first=true

    while IFS= read -r line; do
        local new_sha msg base_msg old_sha
        new_sha=$(echo "$line" | cut -d' ' -f1)
        msg=$(echo "$line" | cut -d' ' -f2-)
        base_msg=$(echo "$msg" | sed 's/ (#[0-9]*)$//')
        old_sha=$(git log --format='%H %s' "$BACKUP_REF" | grep -F "$base_msg" | head -1 | cut -d' ' -f1)

        if [ -n "$old_sha" ] && [ "$old_sha" != "$new_sha" ]; then
            if [ "$first" = true ]; then
                first=false
            else
                pairs+=","
            fi
            pairs+="{\"old_sha\":\"${old_sha}\",\"new_sha\":\"${new_sha}\"}"
        fi
    done < <(git log --format='%H %s' main)

    pairs+="]"
    echo "$pairs"
}

# ── Commands ─────────────────────────────────────────────────────────────

cmd_preview() {
    echo "Building old -> new SHA mapping..."
    echo ""

    local count=0
    git log --format='%H %s' main | while IFS= read -r line; do
        local new_sha msg base_msg old_sha
        new_sha=$(echo "$line" | cut -d' ' -f1)
        msg=$(echo "$line" | cut -d' ' -f2-)
        base_msg=$(echo "$msg" | sed 's/ (#[0-9]*)$//')
        old_sha=$(git log --format='%H %s' "$BACKUP_REF" | grep -F "$base_msg" | head -1 | cut -d' ' -f1)

        if [ -n "$old_sha" ] && [ "$old_sha" != "$new_sha" ]; then
            count=$((count + 1))
            local check_count
            check_count=$(gh api "repos/${REPO}/commits/${old_sha}/check-runs" \
                --jq '.total_count' 2>/dev/null || echo "0")
            echo "  ${old_sha:0:7} -> ${new_sha:0:7} ($check_count checks)  $msg"
        fi
    done

    echo ""

    local json
    json=$(build_json_mapping)
    local total
    total=$(echo "$json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
    echo "Total: $total commits to stamp"
    echo "JSON payload size: $(echo -n "$json" | wc -c) bytes"
}

cmd_run() {
    echo "Building SHA mapping..."
    local json
    json=$(build_json_mapping)
    local total
    total=$(echo "$json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")

    echo "Triggering ci-backfill.yml with $total SHA pairs..."
    echo "  (workflow runs on main, creates check runs via Checks API)"
    echo ""

    gh workflow run "$WORKFLOW" --ref main -f sha_pairs="$json"

    echo "Workflow dispatched. Use './scripts/backfill-ci.sh status' to monitor."
}

cmd_status() {
    echo "CI Backfill workflow runs:"
    echo ""
    gh run list --workflow="$WORKFLOW" --limit=5 \
        --json databaseId,status,conclusion,createdAt \
        --jq '.[] | "\(.databaseId)  \(.status)  \(.conclusion // "-")  \(.createdAt)"'
}

cmd_verify() {
    echo "Verifying check runs on rewritten commits..."
    echo ""

    local pass=0 fail=0 total=0

    git log --format='%H %s' main | while IFS= read -r line; do
        local new_sha msg base_msg old_sha
        new_sha=$(echo "$line" | cut -d' ' -f1)
        msg=$(echo "$line" | cut -d' ' -f2-)
        base_msg=$(echo "$msg" | sed 's/ (#[0-9]*)$//')
        old_sha=$(git log --format='%H %s' "$BACKUP_REF" | grep -F "$base_msg" | head -1 | cut -d' ' -f1)

        if [ -n "$old_sha" ] && [ "$old_sha" != "$new_sha" ]; then
            total=$((total + 1))
            local check_count
            check_count=$(gh api "repos/${REPO}/commits/${new_sha}/check-runs" \
                --jq '.total_count' 2>/dev/null || echo "0")

            if [ "$check_count" -gt 0 ]; then
                local conclusions
                conclusions=$(gh api "repos/${REPO}/commits/${new_sha}/check-runs" \
                    --jq '[.check_runs[].conclusion] | unique | join(",")' 2>/dev/null)
                echo "  ${new_sha:0:7}: $check_count checks [$conclusions]  $(echo "$msg" | cut -c1-60)"
                pass=$((pass + 1))
            else
                echo "  ${new_sha:0:7}: MISSING  $(echo "$msg" | cut -c1-60)"
                fail=$((fail + 1))
            fi
        fi
    done

    echo ""
    echo "With checks: $pass  Missing: $fail  Total: $((pass + fail))"
}

cmd_cleanup() {
    local tags
    tags=$(git tag -l "${TAG_PREFIX}/*" 2>/dev/null || true)
    local count
    count=$(echo "$tags" | grep -c . 2>/dev/null || echo "0")

    if [ "$count" -eq 0 ]; then
        echo "No ${TAG_PREFIX}/* tags to clean up."
    else
        echo "Deleting $count remote tags..."
        echo "$tags" | xargs -I{} git push origin --delete "refs/tags/{}" 2>&1 || true

        echo "Deleting $count local tags..."
        echo "$tags" | xargs git tag -d 2>&1 || true
    fi

    # Clean up test status from earlier
    echo "Done."
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-help}" in
    preview) cmd_preview ;;
    run)     cmd_run ;;
    status)  cmd_status ;;
    verify)  cmd_verify ;;
    cleanup) cmd_cleanup ;;
    *)
        echo "Usage: $0 {preview|run|verify|cleanup}"
        echo ""
        echo "  preview  Show SHA mapping and original check counts"
        echo "  run      Trigger backfill workflow on GitHub Actions"
        echo "  status   Check backfill workflow run status"
        echo "  verify   Verify check runs exist on rewritten SHAs"
        echo "  cleanup  Delete ci-rerun/* tags from earlier attempt"
        exit 1
        ;;
esac
