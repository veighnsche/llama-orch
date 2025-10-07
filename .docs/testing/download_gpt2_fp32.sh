#!/bin/bash
# Download GPT-2 FP32 GGUF model for parity testing
# 
# Usage: ./download_gpt2_fp32.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_ROOT/.test-models/gpt2"
MODEL_FILE="gpt2-fp32.gguf"
HF_REPO="openai-community/gpt2"

echo "=========================================="
echo "GPT-2 FP32 Model Download (Official)"
echo "=========================================="
echo ""
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Size: ~500 MB (FP32 - NO quantization)"
echo "Purpose: Test llama.cpp logging with pure GPT-2 architecture"
echo "Note: Official OpenAI GPT-2, simplest transformer architecture"
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

echo "Downloading GPT-2 (FP32 - Official OpenAI model)..."
echo "This will download the PyTorch model and we'll convert to GGUF."
echo "This will take 2-5 minutes depending on your connection."
echo ""

# Use modern hf command to download the whole repo
if command -v hf &> /dev/null; then
    echo "Using hf download (modern CLI)..."
    
    # Download all model files
    hf download "$HF_REPO" \
        --local-dir "$MODEL_DIR/pytorch"
    
    echo ""
    echo "✅ Download complete!"
    echo ""
    echo "Now converting to GGUF format..."
    
    # Convert to GGUF using llama.cpp converter
    if [ -f "/home/vince/Projects/llama-orch/reference/llama.cpp/convert_hf_to_gguf.py" ]; then
        cd /home/vince/Projects/llama-orch/reference/llama.cpp
        python3 convert_hf_to_gguf.py "$MODEL_DIR/pytorch" \
            --outfile "$MODEL_DIR/$MODEL_FILE" \
            --outtype f32
        echo ""
        echo "✅ Conversion complete!"
    else
        echo "⚠️ Warning: llama.cpp converter not found"
        echo "Please convert manually or the model is already in GGUF format"
    fi
    
else
    echo "❌ Error: hf command not found"
    echo ""
    echo "Install with:"
    echo "  pipx install 'huggingface_hub[cli,hf_transfer]'"
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
echo "✅ GPT-2 FP32 model ready!"
echo "=========================================="
echo ""
echo "Location: $MODEL_DIR/$MODEL_FILE"
echo ""
echo "Model Info:"
echo "  - Architecture: GPT-2 (original transformer)"
echo "  - Parameters: 124M"
echo "  - Vocab Size: 50,257"
echo "  - Precision: FP32 (NO quantization - pure research)"
echo ""
echo "Next steps:"
echo "  1. Test with llama.cpp for garbage tokens"
echo "  2. Compare with other architectures"
echo "  3. Document findings in parity/"
