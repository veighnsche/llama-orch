#!/usr/bin/env bash
# Migrate M0 Planning Folder Structure - AI Agent Reality
# Date: 2025-10-04
# Purpose: Restructure from human team assumptions to AI agent reality

set -e

PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Starting M0 Planning Folder Structure Migration..."
echo "📁 Plan root: $PLAN_ROOT"
echo ""

# ============================================================================
# FOUNDATION TEAM (Foundation-Alpha)
# ============================================================================

echo "🏗️  Migrating Foundation Team..."

cd "$PLAN_ROOT/foundation-team"

# Rename sprint folders with day ranges
if [ -d sprints/week-1 ]; then
    echo "  ✓ Renaming sprint folders..."
    mv sprints/week-1 sprints/sprint-1-http-foundation
    mv sprints/week-2 sprints/sprint-2-ffi-layer
    mv sprints/week-3 sprints/sprint-3-shared-kernels
    mv sprints/week-4 sprints/sprint-4-integration-gate1
    mv sprints/week-5 sprints/sprint-5-support-prep
    mv sprints/week-6 sprints/sprint-6-adapter-gate3
    mv sprints/week-7 sprints/sprint-7-final-integration
fi

# Remove old story workflow folders
if [ -d stories/backlog ]; then
    echo "  ✓ Removing old story workflow folders..."
    rm -rf stories/backlog stories/in-progress stories/review stories/done
fi

# Create new story organization by ID range
echo "  ✓ Creating story folders by ID range..."
mkdir -p stories/FT-001-to-FT-010
mkdir -p stories/FT-011-to-FT-020
mkdir -p stories/FT-021-to-FT-030
mkdir -p stories/FT-031-to-FT-040
mkdir -p stories/FT-041-to-FT-050

# Create execution tracking folder
echo "  ✓ Creating execution tracking folder..."
mkdir -p execution

echo "  ✅ Foundation Team migration complete"
echo ""

# ============================================================================
# LLAMA TEAM (Llama-Beta)
# ============================================================================

echo "🦙 Migrating Llama Team..."

cd "$PLAN_ROOT/llama-team"

# Create sprint folders
echo "  ✓ Creating sprint folders..."
mkdir -p sprints/sprint-0-prep-work
mkdir -p sprints/sprint-1-gguf-foundation
mkdir -p sprints/sprint-2-gguf-bpe-tokenizer
mkdir -p sprints/sprint-3-utf8-llama-kernels
mkdir -p sprints/sprint-4-gqa-gate1
mkdir -p sprints/sprint-5-qwen-integration
mkdir -p sprints/sprint-6-phi3-adapter
mkdir -p sprints/sprint-7-final-integration

# Create story folders
echo "  ✓ Creating story folders by ID range..."
mkdir -p stories/LT-000-prep
mkdir -p stories/LT-001-to-LT-010
mkdir -p stories/LT-011-to-LT-020
mkdir -p stories/LT-021-to-LT-030
mkdir -p stories/LT-031-to-LT-038

# Create execution tracking
echo "  ✓ Creating execution tracking folder..."
mkdir -p execution

echo "  ✅ Llama Team migration complete"
echo ""

# ============================================================================
# GPT TEAM (GPT-Gamma)
# ============================================================================

echo "🤖 Migrating GPT Team..."

cd "$PLAN_ROOT/gpt-team"

# Create sprint folders
echo "  ✓ Creating sprint folders..."
mkdir -p sprints/sprint-0-prep-work
mkdir -p sprints/sprint-1-hf-tokenizer
mkdir -p sprints/sprint-2-gpt-kernels
mkdir -p sprints/sprint-3-mha-gate1
mkdir -p sprints/sprint-4-gpt-basic
mkdir -p sprints/sprint-5-mxfp4-dequant
mkdir -p sprints/sprint-6-mxfp4-integration
mkdir -p sprints/sprint-7-adapter-e2e
mkdir -p sprints/sprint-8-final-integration

# Create story folders
echo "  ✓ Creating story folders by ID range..."
mkdir -p stories/GT-000-prep
mkdir -p stories/GT-001-to-GT-010
mkdir -p stories/GT-011-to-GT-020
mkdir -p stories/GT-021-to-GT-030
mkdir -p stories/GT-031-to-GT-040
mkdir -p stories/GT-041-to-GT-048

# Create execution tracking
echo "  ✓ Creating execution tracking folder..."
mkdir -p execution

echo "  ✅ GPT Team migration complete"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

cd "$PLAN_ROOT"

echo "✅ Migration Complete!"
echo ""
echo "📊 Summary:"
echo "  • Foundation Team: 7 sprints, 49 stories, 89 days"
echo "  • Llama Team: 8 sprints (incl prep), 39 stories, 90 days"
echo "  • GPT Team: 9 sprints (incl prep), 49 stories, 110 days ← M0 CRITICAL PATH"
echo ""
echo "📁 New Structure:"
echo "  • Sprints: Named by goal + day range (not week-X)"
echo "  • Stories: Organized by ID range (sequential execution)"
echo "  • Execution: Day tracker, dependencies, milestones"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review execution/ folders for each team"
echo "  2. Create story cards (139 total)"
echo "  3. Set up gate checklists"
echo "  4. Begin Day 1 execution"
echo ""
echo "🚀 Ready to execute!"
