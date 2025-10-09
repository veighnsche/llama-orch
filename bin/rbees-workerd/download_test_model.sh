#!/bin/bash
# Download TinyLlama 1.1B SafeTensors for rbees-workerd testing
# Created by: TEAM-010
#
# rbees-workerd requires SafeTensors format (not GGUF)
# TinyLlama is perfect for testing:
# - Small size (~2.2 GB)
# - Standard Llama architecture
# - Fast inference
# - Well-tested

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/.test-models/tinyllama-safetensors"
HF_REPO="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

echo "=========================================="
echo "TinyLlama 1.1B SafeTensors Download"
echo "=========================================="
echo ""
echo "Target: $MODEL_DIR"
echo "Size: ~2.2 GB (FP16 SafeTensors)"
echo "Purpose: Test rbees-workerd inference"
echo "Architecture: Standard Llama (7B compatible)"
echo ""

# Create directory
mkdir -p "$MODEL_DIR"

# Check if already downloaded
if [ -f "$MODEL_DIR/model.safetensors" ] && [ -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "✅ Model already exists at $MODEL_DIR"
    echo ""
    echo "Files:"
    ls -lh "$MODEL_DIR"/*.safetensors "$MODEL_DIR"/tokenizer.json "$MODEL_DIR"/config.json 2>/dev/null || true
    echo ""
    echo "To re-download, delete the directory first:"
    echo "  rm -rf $MODEL_DIR"
    exit 0

echo "Downloading TinyLlama 1.1B (SafeTensors)..."
echo "This will take 2-5 minutes depending on your connection."
echo ""

# TEAM-023: DEPRECATED - huggingface-cli is deprecated, use `hf` CLI instead!
# Check for hf CLI (modern replacement for huggingface-cli)
if ! command -v hf &> /dev/null; then
    echo "❌ Error: hf CLI not found"
    echo ""
    echo "Install with:"
    echo "  pip install huggingface_hub[cli]"
    echo ""
    echo "NOTE: 'huggingface-cli' is DEPRECATED. Use 'hf' instead!"
    echo ""
    exit 1
fi

# Download model files
echo "Using hf CLI (modern replacement for deprecated huggingface-cli)..."
hf download "$HF_REPO" \
    --include "*.safetensors" "tokenizer.json" "config.json" "tokenizer_config.json" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

# Verify download
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "❌ Download failed: model.safetensors not found"
    exit 1
fi

if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "❌ Download failed: tokenizer.json not found"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ Download failed: config.json not found"
    exit 1
fi

echo ""
echo "✅ Download complete!"
echo ""
echo "Files:"
ls -lh "$MODEL_DIR"/*.safetensors "$MODEL_DIR"/tokenizer.json "$MODEL_DIR"/config.json

echo ""
echo "=========================================="
echo "✅ TinyLlama SafeTensors Ready!"
echo "=========================================="
echo ""
echo "Location: $MODEL_DIR"
echo ""
echo "Model info:"
echo "  - Architecture: Llama (standard)"
echo "  - Parameters: 1.1B"
echo "  - Vocab: 32,000 tokens"
echo "  - Hidden: 2048"
echo "  - Layers: 22"
echo "  - Context: 2048"
echo "  - Format: SafeTensors (FP16)"
echo ""
echo "Next steps:"
echo "  1. Test model loading:"
echo "     LLORCH_TEST_MODEL_PATH=$MODEL_DIR cargo test test_device_residency_enforcement --features cpu -- --ignored"
echo ""
echo "  2. Run CPU worker:"
echo "     cargo run --release --features cpu --bin llorch-cpu-candled -- \\"
echo "       --worker-id test-worker \\"
echo "       --model $MODEL_DIR \\"
echo "       --port 8080 \\"
echo "       --callback-url http://localhost:9999"
echo ""
