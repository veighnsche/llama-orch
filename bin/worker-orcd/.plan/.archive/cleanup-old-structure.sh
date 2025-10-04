#!/usr/bin/env bash
# Cleanup Old Planning Structure
# Date: 2025-10-04
# Purpose: Remove old week-based folders and other human team artifacts

set -e

PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🧹 Cleaning up old planning structure..."
echo "📁 Plan root: $PLAN_ROOT"
echo ""

# ============================================================================
# REMOVE OLD WEEK-BASED FOLDERS
# ============================================================================

echo "🗑️  Removing old week-based sprint folders..."

# Llama Team
if [ -d "$PLAN_ROOT/llama-team/sprints/week-1" ]; then
    echo "  • Llama Team: Removing week-1 through week-7..."
    rm -rf "$PLAN_ROOT/llama-team/sprints/week-"*
fi

# GPT Team
if [ -d "$PLAN_ROOT/gpt-team/sprints/week-1" ]; then
    echo "  • GPT Team: Removing week-1 through week-7..."
    rm -rf "$PLAN_ROOT/gpt-team/sprints/week-"*
fi

echo "  ✅ Old week folders removed"
echo ""

# ============================================================================
# REMOVE OLD STORY WORKFLOW FOLDERS (if any remain)
# ============================================================================

echo "🗑️  Removing old story workflow folders..."

for team in foundation-team llama-team gpt-team; do
    if [ -d "$PLAN_ROOT/$team/stories/backlog" ]; then
        echo "  • $team: Removing backlog/in-progress/review/done..."
        rm -rf "$PLAN_ROOT/$team/stories/backlog"
        rm -rf "$PLAN_ROOT/$team/stories/in-progress"
        rm -rf "$PLAN_ROOT/$team/stories/review"
        rm -rf "$PLAN_ROOT/$team/stories/done"
    fi
done

echo "  ✅ Old story workflow folders removed"
echo ""

# ============================================================================
# REMOVE OLD EXECUTIVE SUMMARIES (human team assumptions)
# ============================================================================

echo "🗑️  Archiving old executive summaries..."

for team in foundation-team llama-team gpt-team; do
    if [ -f "$PLAN_ROOT/$team/EXECUTIVE_SUMMARY.md" ]; then
        echo "  • $team: Moving EXECUTIVE_SUMMARY.md to .archive/"
        mkdir -p "$PLAN_ROOT/$team/.archive"
        mv "$PLAN_ROOT/$team/EXECUTIVE_SUMMARY.md" "$PLAN_ROOT/$team/.archive/EXECUTIVE_SUMMARY.old.md"
    fi
    
    if [ -f "$PLAN_ROOT/$team/INDEX.md" ]; then
        echo "  • $team: Moving INDEX.md to .archive/"
        mv "$PLAN_ROOT/$team/INDEX.md" "$PLAN_ROOT/$team/.archive/INDEX.old.md"
    fi
done

echo "  ✅ Old summaries archived"
echo ""

# ============================================================================
# REMOVE EMPTY INTEGRATION-GATES FOLDERS
# ============================================================================

echo "🗑️  Cleaning up empty integration-gates folders..."

for team in foundation-team llama-team gpt-team; do
    if [ -d "$PLAN_ROOT/$team/integration-gates" ]; then
        # Check if empty
        if [ -z "$(ls -A "$PLAN_ROOT/$team/integration-gates")" ]; then
            echo "  • $team: Removing empty integration-gates/"
            rm -rf "$PLAN_ROOT/$team/integration-gates"
        fi
    fi
done

echo "  ✅ Empty folders cleaned up"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "✅ Cleanup Complete!"
echo ""
echo "📊 Removed:"
echo "  • Old week-based sprint folders (week-1 through week-7)"
echo "  • Old story workflow folders (backlog, in-progress, review, done)"
echo "  • Old executive summaries (archived to .archive/)"
echo "  • Empty integration-gates folders"
echo ""
echo "📁 Current Structure:"
echo "  • Sprints: Named by goal + day range"
echo "  • Stories: Organized by ID range"
echo "  • Execution: Ready for tracking"
echo ""
echo "🚀 Clean slate ready!"
