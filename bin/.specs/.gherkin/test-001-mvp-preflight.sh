#!/usr/bin/env bash
# Test-001 MVP: Preflight Checks
# Verifies all prerequisites before running e2e test
#
# Created by: TEAM-029

set -euo pipefail

echo "=== Test-001 MVP: Preflight Checks ==="
echo ""

FAILED=0

# Check 1: Rust toolchain
echo "[Check 1] Rust toolchain..."
if command -v cargo &> /dev/null; then
    RUST_VERSION=$(cargo --version)
    echo "  ✓ $RUST_VERSION"
else
    echo "  ✗ cargo not found"
    FAILED=1
fi
echo ""

# Check 2: Build rbee-hive
echo "[Check 2] Building rbee-hive..."
if cargo build --bin rbee-hive 2>&1 | grep -q "Finished"; then
    echo "  ✓ rbee-hive builds successfully"
else
    echo "  ✗ rbee-hive build failed"
    FAILED=1
fi
echo ""

# Check 3: Build rbee-keeper
echo "[Check 3] Building rbee-keeper..."
if cargo build --bin rbee 2>&1 | grep -q "Finished"; then
    echo "  ✓ rbee builds successfully"
else
    echo "  ✗ rbee build failed"
    FAILED=1
fi
echo ""

# Check 4: llm-worker-rbee binary
echo "[Check 4] Checking llm-worker-rbee binary..."
if [ -f "./target/debug/llm-worker-rbee" ]; then
    echo "  ✓ llm-worker-rbee binary exists"
    
    # Try to get help to verify it runs
    if ./target/debug/llm-worker-rbee --help &> /dev/null; then
        echo "  ✓ llm-worker-rbee runs"
    else
        echo "  ⚠ llm-worker-rbee exists but may not run properly"
    fi
else
    echo "  ✗ llm-worker-rbee binary not found"
    echo "     Run: cargo build --bin llm-worker-rbee"
    FAILED=1
fi
echo ""

# Check 5: Model file
echo "[Check 5] Checking for model file..."
MODEL_PATHS=(
    "/models/model.gguf"
    "$HOME/.cache/llama-orch/models/tinyllama.gguf"
    "./models/tinyllama.gguf"
)

MODEL_FOUND=0
for path in "${MODEL_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "  ✓ Model found at: $path"
        MODEL_FOUND=1
        break
    fi
done

if [ $MODEL_FOUND -eq 0 ]; then
    echo "  ✗ No model file found"
    echo "     Checked paths:"
    for path in "${MODEL_PATHS[@]}"; do
        echo "       - $path"
    done
    echo "     Run: bin/llm-worker-rbee/download_test_model.sh"
    FAILED=1
fi
echo ""

# Check 6: Port 8080 availability
echo "[Check 6] Checking port 8080..."
if lsof -i :8080 &> /dev/null; then
    echo "  ⚠ Port 8080 is already in use"
    echo "     Process using port:"
    lsof -i :8080 | grep LISTEN || true
    echo "     You may need to kill the process or use a different port"
else
    echo "  ✓ Port 8080 is available"
fi
echo ""

# Check 7: SQLite
echo "[Check 7] Checking SQLite..."
if command -v sqlite3 &> /dev/null; then
    SQLITE_VERSION=$(sqlite3 --version | cut -d' ' -f1)
    echo "  ✓ SQLite $SQLITE_VERSION"
else
    echo "  ⚠ sqlite3 command not found (not required, but useful for debugging)"
fi
echo ""

# Summary
echo "=== Preflight Summary ==="
if [ $FAILED -eq 0 ]; then
    echo "✅ All critical checks passed - ready to run e2e test"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./bin/.specs/.gherkin/test-001-mvp-local.sh"
    echo "  2. Or manually:"
    echo "     Terminal 1: ./target/debug/rbee-hive daemon"
    echo "     Terminal 2: ./target/debug/rbee infer --node localhost --model test --prompt 'hello' --max-tokens 5"
    exit 0
else
    echo "❌ Some checks failed - fix issues before running e2e test"
    exit 1
fi
