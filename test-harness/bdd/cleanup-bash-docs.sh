#!/usr/bin/env bash
# Cleanup bash script documentation before Rust port
# TEAM-111 - 2025-10-18

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="$SCRIPT_DIR/.archive/bash-script"

echo "🧹 Cleaning up bash script documentation..."
echo ""

# Create archive directory
echo "📁 Creating archive directory..."
mkdir -p "$ARCHIVE_DIR"
echo "   Created: $ARCHIVE_DIR"
echo ""

# Archive reference docs (useful for Rust port)
echo "📦 Archiving reference documentation..."
if [[ -f "$SCRIPT_DIR/.docs/ARCHITECTURE.md" ]]; then
    mv "$SCRIPT_DIR/.docs/ARCHITECTURE.md" "$ARCHIVE_DIR/"
    echo "   ✅ Archived ARCHITECTURE.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/BDD_RUNNER_IMPROVEMENTS.md" ]]; then
    mv "$SCRIPT_DIR/.docs/BDD_RUNNER_IMPROVEMENTS.md" "$ARCHIVE_DIR/"
    echo "   ✅ Archived BDD_RUNNER_IMPROVEMENTS.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/RERUN_FEATURE.md" ]]; then
    mv "$SCRIPT_DIR/.docs/RERUN_FEATURE.md" "$ARCHIVE_DIR/"
    echo "   ✅ Archived RERUN_FEATURE.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/REFACTOR_COMPLETE.md" ]]; then
    mv "$SCRIPT_DIR/.docs/REFACTOR_COMPLETE.md" "$ARCHIVE_DIR/"
    echo "   ✅ Archived REFACTOR_COMPLETE.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/SUMMARY.md" ]]; then
    mv "$SCRIPT_DIR/.docs/SUMMARY.md" "$ARCHIVE_DIR/"
    echo "   ✅ Archived SUMMARY.md"
fi
echo ""

# Delete bash-specific docs
echo "🗑️  Deleting bash-specific documentation..."
if [[ -f "$SCRIPT_DIR/QUICK_START.md" ]]; then
    rm "$SCRIPT_DIR/QUICK_START.md"
    echo "   ❌ Deleted QUICK_START.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/DEVELOPER_GUIDE.md" ]]; then
    rm "$SCRIPT_DIR/.docs/DEVELOPER_GUIDE.md"
    echo "   ❌ Deleted DEVELOPER_GUIDE.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/REFACTOR_INVENTORY.md" ]]; then
    rm "$SCRIPT_DIR/.docs/REFACTOR_INVENTORY.md"
    echo "   ❌ Deleted REFACTOR_INVENTORY.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/EXAMPLE_OUTPUT.md" ]]; then
    rm "$SCRIPT_DIR/.docs/EXAMPLE_OUTPUT.md"
    echo "   ❌ Deleted EXAMPLE_OUTPUT.md"
fi

if [[ -f "$SCRIPT_DIR/.docs/INDEX.md" ]]; then
    rm "$SCRIPT_DIR/.docs/INDEX.md"
    echo "   ❌ Deleted INDEX.md"
fi
echo ""

# Archive bash scripts
echo "📦 Archiving bash scripts..."
if [[ -f "$SCRIPT_DIR/run-bdd-tests.sh" ]]; then
    mv "$SCRIPT_DIR/run-bdd-tests.sh" "$ARCHIVE_DIR/"
    echo "   ✅ Archived run-bdd-tests.sh"
fi

if [[ -f "$SCRIPT_DIR/run-bdd-tests-old.sh.backup" ]]; then
    mv "$SCRIPT_DIR/run-bdd-tests-old.sh.backup" "$ARCHIVE_DIR/"
    echo "   ✅ Archived run-bdd-tests-old.sh.backup"
fi
echo ""

# Summary
echo "✅ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "   Archived: 7 files (5 docs + 2 scripts)"
echo "   Deleted:  5 files (bash-specific docs)"
echo "   Location: $ARCHIVE_DIR"
echo ""
echo "📝 Next steps:"
echo "   1. Review README.md and update bash script references"
echo "   2. Port features to Rust xtask"
echo "   3. Create new Rust-specific documentation"
echo ""
echo "🚀 Ready for Rust port!"
