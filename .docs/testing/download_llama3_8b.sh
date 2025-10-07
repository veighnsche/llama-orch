#!/bin/bash
# Download DeepSeek-R1-Distill-Llama-8B FP16 for parity testing
# 
# Usage: ./download_deepseek_llama_8b_fp16.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_ROOT/.test-models/llama3"
MODEL_FILE="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
HF_REPO="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
HF_FILE="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

echo "=========================================="
echo "Llama-3-8B-Instruct Q4_K_M Download"
echo "=========================================="
echo ""
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Size: ~4.9 GB (Q4_K_M)"
echo "Purpose: Test llama.cpp logging with Llama-3 architecture"
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

echo "Downloading Llama-3-8B-Instruct (Q4_K_M)..."
echo "This will take 5-10 minutes depending on your connection."
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
echo "Computing SHA256 checksum..."
sha256sum "$MODEL_DIR/$MODEL_FILE"

echo ""
echo "=========================================="
echo "✅ Llama-3-8B-Instruct ready!"
echo "=========================================="
echo ""
echo "Location: $MODEL_DIR/$MODEL_FILE"
echo ""
echo "Next steps:"
echo "  1. Test with llama.cpp for garbage tokens"
echo "  2. Compare with TinyLlama (also Llama-based, clean)"
echo "  3. Document findings in parity/"
