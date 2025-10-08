#!/bin/bash
# Download Llama-2-7B FP16 GGUF model - The Perfect Foundation Model
# 
# Usage: ./download_llama2_7b_fp16.sh
# 
# WHY LLAMA-2 7B FP16:
# - Modern architecture (RoPE, RMSNorm, SwiGLU) - industry standard
# - Perfect size (7B params = ~14GB FP16, fits RTX 3060)
# - Future-proof: Same architecture as Llama-3, Mistral, Qwen, CodeLlama
# - Well-documented: Battle-tested, tons of references
# - GGUF native: Learn the format properly from day 1
# - Production-ready: Actually used in real deployments
#
# This is THE foundation model for llorch-cpud validation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_ROOT/.test-models/llama2-7b"
MODEL_FILE="llama-2-7b.Q8_0.gguf"
HF_REPO="TheBloke/Llama-2-7B-GGUF"
HF_FILE="llama-2-7b.Q8_0.gguf"

echo "=========================================="
echo "Llama-2-7B FP16 Foundation Model Download"
echo "=========================================="
echo ""
echo "🎯 THE PERFECT STARTING POINT"
echo ""
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Size: ~7.2 GB (Q8_0 - 8-bit quantization, near FP16 quality)"
echo "Architecture: Llama-2 (RoPE + RMSNorm + SwiGLU)"
echo "Purpose: Foundation model for llorch-cpud validation"
echo ""
echo "Why this model:"
echo "  ✅ Modern architecture (not outdated like GPT-2)"
echo "  ✅ Manageable size (fits consumer GPUs)"
echo "  ✅ Future-proof (50+ models use same architecture)"
echo "  ✅ GGUF native (learn format properly)"
echo "  ✅ Production-ready (real deployments)"
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

echo "Downloading Llama-2-7B (FP16)..."
echo "This will take 10-15 minutes depending on your connection."
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
    echo "Note: This model is 13.5GB, hf_transfer enables faster downloads"
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
echo "✅ Llama-2-7B FP16 Foundation Model Ready!"
echo "=========================================="
echo ""
echo "Location: $MODEL_DIR/$MODEL_FILE"
echo ""
echo "Next steps:"
echo "  1. Test with llama.cpp to verify model works"
echo "  2. Extract reference checkpoints using Team 006's tool"
echo "  3. Build llorch-cpud Llama-2 implementation"
echo "  4. Compare checkpoints for validation"
echo ""
echo "Test command:"
echo "  cd reference/llama.cpp"
echo "  ./llama-cli -m $MODEL_DIR/$MODEL_FILE -p \"Hello\" -n 10"
echo ""
