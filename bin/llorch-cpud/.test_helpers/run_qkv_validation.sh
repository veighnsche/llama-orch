#!/bin/bash
# Complete QKV Projection Validation Suite
# Runs llorch-cpud test, reference implementations, and comparison

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Checkpoint 2: QKV Projection Validation Suite          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Step 1: Run our implementation test
echo "[1/4] Running llorch-cpud QKV test..."
cargo test --test isolated_checkpoint_02 test_isolated_checkpoint_02_all -- --nocapture 2>&1 | tee /tmp/llorch_qkv_test.log

# Extract outputs from test log (if test writes to files)
# For now, we'll generate outputs manually
echo ""
echo "[1.5/4] Generating our QKV outputs..."
# TODO: Update test to write output files, or extract from log

# Step 2: Run Candle reference
echo ""
echo "[2/4] Running Candle reference implementation..."
if [ -d "$SCRIPT_DIR/candle_qkv_test" ]; then
    cd "$SCRIPT_DIR/candle_qkv_test"
    cargo run --release 2>&1 | tee /tmp/candle_qkv_test.log
    # Copy output files to parent directory for comparison
    cp checkpoint_02_*_candle.txt "$SCRIPT_DIR/"
    cd "$PROJECT_ROOT"
else
    echo "⚠️  Candle test directory not found"
fi

# Step 3: Run Mistral.rs reference
echo ""
echo "[3/4] Running Mistral.rs reference implementation..."
if [ -d "$SCRIPT_DIR/mistralrs_qkv_test" ]; then
    cd "$SCRIPT_DIR/mistralrs_qkv_test"
    cargo run --release 2>&1 | tee /tmp/mistralrs_qkv_test.log
    # Copy output files to parent directory for comparison
    cp checkpoint_02_*_mistralrs.txt "$SCRIPT_DIR/"
    cd "$PROJECT_ROOT"
else
    echo "⚠️  Mistral.rs test directory not found"
fi

# Step 4: Compare outputs
echo ""
echo "[4/4] Comparing outputs..."
cd "$SCRIPT_DIR"
python3 compare_qkv_outputs.py

echo ""
echo "✅ Validation complete!"
echo ""
echo "Output files:"
echo "  - /tmp/llorch_qkv_test.log"
echo "  - /tmp/candle_qkv_test.log"
echo "  - /tmp/mistralrs_qkv_test.log"
