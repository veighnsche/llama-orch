#!/bin/bash
# Download Llama-3.2-3B-Instruct FP16 GGUF model for parity testing
# 
# Usage: ./download_llama32_3b_fp16.sh
# 
# This script downloads Llama-3.2-3B-Instruct FP16 model to test if
# the garbage token issue in llama.cpp logging is model-specific or systemic.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_ROOT/.test-models/llama32"
MODEL_FILE="Komodo-Llama-3.2-3B-v2-fp16.gguf"
HF_REPO="tensorblock/Komodo-Llama-3.2-3B-v2-fp16-GGUF"
HF_FILE="Komodo-Llama-3.2-3B-v2-fp16.gguf"

echo "=========================================="
echo "Llama-3.2-3B-Instruct FP16 Model Download"
echo "=========================================="
echo ""
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Size: ~6.4 GB"
echo "Purpose: Test llama.cpp logging garbage tokens"
echo ""

# Create directory
mkdir -p "$MODEL_DIR"

# Check if already downloaded
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✅ Model already exists at $MODEL_DIR/$MODEL_FILE"
    echo ""
    echo "File info:"
    ls -lh "$MODEL_DIR/$MODEL_FILE"
    echo ""
    echo "To re-download, delete the file first:"
    echo "  rm $MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "Downloading Llama-3.2-3B-Instruct (FP16)..."
echo "This will take 5-10 minutes depending on your connection."
echo ""

# Use modern hf command
if command -v hf &> /dev/null; then
    echo "Using hf download (modern CLI)..."
    
    hf download "$HF_REPO" "$HF_FILE" \
        --local-dir "$MODEL_DIR"
    
    # Rename if needed
    if [ -f "$MODEL_DIR/$HF_FILE" ] && [ "$HF_FILE" != "$MODEL_FILE" ]; then
        mv "$MODEL_DIR/$HF_FILE" "$MODEL_DIR/$MODEL_FILE"
    fi
    
    echo ""
    echo "✅ Download complete!"
    
else
    echo "❌ Error: hf command not found"
    echo ""
    echo "Install with:"
    echo "  pipx install 'huggingface_hub[cli,hf_transfer]'"
    echo ""
    echo "Note: This model is 6.4GB, hf_transfer enables faster downloads"
    exit 1
fi

# Show file info
echo ""
echo "File info:"
ls -lh "$MODEL_DIR/$MODEL_FILE"

echo ""
echo "Computing SHA256 checksum..."
sha256sum "$MODEL_DIR/$MODEL_FILE"

echo ""
echo "=========================================="
echo "✅ Llama-3.2-3B FP16 model ready!"
echo "=========================================="
echo ""
echo "Location: $MODEL_DIR/$MODEL_FILE"
echo ""
echo "Next steps:"
echo "  1. Run parity test with this model"
echo "  2. Compare garbage token behavior with Qwen"
echo "  3. Document findings in investigation-teams/parity/"
