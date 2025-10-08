#!/bin/bash
# Cross-Reference Validation Script
# Tests that reference implementations work before adding logging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Cross-Reference Validation: Pre-Flight Check           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Verify tinygrad
echo "=== Step 1: Verify tinygrad ==="
cd "$REPO_ROOT/reference/tinygrad"

if [ ! -d "tinygrad" ]; then
    echo "❌ tinygrad source not found"
    exit 1
fi

echo "Current branch:"
git branch --show-current || echo "(detached HEAD)"

echo ""
echo "Checking if tinygrad runs..."
if PYTHONPATH="$REPO_ROOT/reference/tinygrad" python3 -c "from tinygrad import Tensor; print('✅ tinygrad imports successfully')" 2>/dev/null; then
    echo "✅ tinygrad: Ready"
else
    echo "⚠️  tinygrad: Import failed, may need setup"
    echo "   Try: cd $REPO_ROOT/reference/tinygrad && pip install -e ."
fi

echo ""

# Step 2: Verify candle
echo "=== Step 2: Verify candle ==="
cd "$REPO_ROOT/reference/candle"

echo "Current branch:"
git branch --show-current || echo "(detached HEAD)"

if [ -f "Cargo.toml" ]; then
    echo "✅ candle: Cargo project found"
else
    echo "❌ candle: Not a Cargo project"
    exit 1
fi

echo ""

# Step 3: Verify mistral.rs
echo "=== Step 3: Verify mistral.rs ==="
cd "$REPO_ROOT/reference/mistral.rs"

echo "Current branch:"
git branch --show-current || echo "(detached HEAD)"

if [ -f "Cargo.toml" ]; then
    echo "✅ mistral.rs: Cargo project found"
else
    echo "❌ mistral.rs: Not a Cargo project"
    exit 1
fi

echo ""

# Step 4: Check for orch_log branch
echo "=== Step 4: Check orch_log branches ==="

cd "$REPO_ROOT/reference/tinygrad"
if git branch -a | grep -q "orch_log"; then
    echo "✅ tinygrad: orch_log branch exists"
else
    echo "⚠️  tinygrad: orch_log branch not found"
fi

cd "$REPO_ROOT/reference/candle"
if git branch -a | grep -q "orch_log"; then
    echo "✅ candle: orch_log branch exists"
else
    echo "⚠️  candle: orch_log branch not found"
fi

cd "$REPO_ROOT/reference/mistral.rs"
if git branch -a | grep -q "orch_log"; then
    echo "✅ mistral.rs: orch_log branch exists"
else
    echo "⚠️  mistral.rs: orch_log branch not found"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pre-Flight Check Complete                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Switch to orch_log branches (if available)"
echo "  2. Add non-blocking logging to references"
echo "  3. Extract checkpoint outputs"
echo "  4. Run cross-reference validation tests"
