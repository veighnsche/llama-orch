#!/bin/bash
# Download Mistral-7B-Instruct FP16 GGUF model for parity testing
# 
# Usage: ./download_mistral_7b_fp16.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_ROOT/.test-models/mistral"
MODEL_FILE="mistral-7b-instruct-v0.2.f16.gguf"
HF_REPO="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
HF_FILE="mistral-7b-instruct-v0.2.f16.gguf"

echo "=========================================="
echo "Mistral-7B-Instruct FP16 Model Download"
echo "=========================================="
echo ""
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Size: ~14 GB"
echo "Purpose: Test llama.cpp logging garbage tokens with different architecture"
echo ""

# Create directory
mkdir -p "$MODEL_DIR"

# Check if already downloaded
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✅ Model already exists at $MODEL_DIR/$MODEL_FILE"
    echo ""
    echo "File info:"
    ls -lh "$MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "Downloading Mistral-7B-Instruct (FP16)..."
echo "This will take 10-20 minutes depending on your connection."
echo ""

# Use modern hf command
if command -v hf &> /dev/null; then
    echo "Using hf download (modern CLI)..."
    
    hf download "$HF_REPO" "$HF_FILE" \
        --local-dir "$MODEL_DIR"
    
    echo ""
    echo "✅ Download complete!"
    
else
    echo "❌ Error: hf command not found"
    echo ""
    echo "Install with:"
    echo "  pipx install 'huggingface_hub[cli,hf_transfer]'"
    echo ""
    echo "Note: hf_transfer enables faster downloads for large files"
    exit 1
fi

# Show file info
echo ""
echo "File info:"
ls -lh "$MODEL_DIR/$MODEL_FILE"

echo ""
echo "=========================================="
echo "✅ Mistral-7B FP16 model ready!"
echo "=========================================="
