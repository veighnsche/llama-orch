#!/bin/bash
# Run real GPT-2 model validation for llorch-cpud
# This script validates Checkpoints 1 & 2 with actual GPT-2 weights

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Real GPT-2 Model Validation for llorch-cpud            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/.test-models/gpt2/extracted_weights"

# Step 1: Check Python dependencies
echo "[1/4] Checking Python dependencies..."
if ! python3 -c "import torch, transformers, numpy" 2>/dev/null; then
    echo "❌ Missing Python dependencies"
    echo ""
    echo "Please install:"
    echo "  pip install torch transformers numpy"
    echo ""
    exit 1
fi
echo "✅ Python dependencies OK"
echo ""

# Step 2: Extract GPT-2 weights if needed
echo "[2/4] Checking GPT-2 weights..."
if [ ! -d "$WEIGHTS_DIR" ] || [ ! -f "$WEIGHTS_DIR/metadata.json" ]; then
    echo "Extracting GPT-2 weights from HuggingFace..."
    cd "$REPO_ROOT"
    python3 .docs/testing/extract_gpt2_weights.py
    echo ""
else
    echo "✅ GPT-2 weights already extracted"
    echo "   Location: $WEIGHTS_DIR"
    echo ""
fi

# Step 3: Run Checkpoint 1 test
echo "[3/4] Running Checkpoint 1: LayerNorm with real GPT-2 weights..."
cd "$SCRIPT_DIR"
cargo test --test real_gpt2_checkpoint_01 -- --nocapture
echo ""

# Step 4: Run Checkpoint 2 test
echo "[4/4] Running Checkpoint 2: QKV with real GPT-2 weights..."
cargo test --test real_gpt2_checkpoint_02 -- --nocapture
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Real GPT-2 Validation Complete!                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Checkpoint 1: LayerNorm validated with real GPT-2 weights"
echo "✅ Checkpoint 2: QKV validated with real GPT-2 weights"
echo ""
echo "Next steps:"
echo "  1. Update documentation to reflect real validation"
echo "  2. Remove 'synthetic weights only' warnings"
echo "  3. Add 'production-ready' claims (now justified!)"
echo ""
