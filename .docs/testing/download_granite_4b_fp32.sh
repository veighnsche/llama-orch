#!/bin/bash
# Download IBM Granite-4.0-Micro and convert to GGUF FP32
# 
# Usage: ./download_granite_4b_fp32.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_ROOT/.test-models/granite"
MODEL_FILE="granite-4.0-micro-fp32.gguf"
HF_REPO="ibm-granite/granite-4.0-micro"

echo "=========================================="
echo "IBM Granite-4.0-Micro FP32 Download"
echo "=========================================="
echo ""
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Size: ~8 GB (FP32 - NO quantization)"
echo "Purpose: Test llama.cpp logging with Granite architecture"
echo "Note: IBM Granite is a new architecture for parity testing"
echo ""

# Create directory
mkdir -p "$MODEL_DIR"

# Check if already converted
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✅ Model already exists at $MODEL_DIR/$MODEL_FILE"
    echo ""
    echo "File info:"
    ls -lh "$MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "Downloading IBM Granite-4.0-Micro (FP32)..."
echo "This will download the PyTorch model and convert to GGUF."
echo "This will take 5-10 minutes depending on your connection."
echo ""

# Use modern hf command to download the whole repo
if command -v hf &> /dev/null; then
    echo "Using hf download (modern CLI)..."
    
    # Download all model files
    export PATH="$HOME/.local/bin:$PATH"
    hf download "$HF_REPO" \
        --local-dir "$MODEL_DIR/pytorch"
    
    echo ""
    echo "✅ Download complete!"
    echo ""
    echo "Now converting to GGUF format (FP32)..."
    
    # Convert to GGUF using llama.cpp converter
    CONVERTER="$REPO_ROOT/reference/llama.cpp/convert_hf_to_gguf.py"
    
    if [ -f "$CONVERTER" ]; then
        cd "$REPO_ROOT/reference/llama.cpp"
        python3 "$CONVERTER" "$MODEL_DIR/pytorch" \
            --outfile "$MODEL_DIR/$MODEL_FILE" \
            --outtype f32
        
        echo ""
        echo "✅ Conversion complete!"
    else
        echo "⚠️ Warning: llama.cpp converter not found at $CONVERTER"
        echo "Please convert manually"
        exit 1
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
echo "✅ IBM Granite-4.0-Micro FP32 ready!"
echo "=========================================="
echo ""
echo "Location: $MODEL_DIR/$MODEL_FILE"
echo ""
echo "Model Info:"
echo "  - Architecture: IBM Granite (new architecture)"
echo "  - Parameters: 4B"
echo "  - Vocab Size: ~50K (estimated)"
echo "  - Precision: FP32 (NO quantization - pure research)"
echo ""
echo "Next steps:"
echo "  1. Test with llama.cpp: cd reference/llama.cpp && ./test_logging.sh granite"
echo "  2. Analyze garbage tokens"
echo "  3. Compare with other architectures"
echo "  4. Document findings in parity/"
