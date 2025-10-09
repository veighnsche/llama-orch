#!/bin/bash
# Download TinyLlama 1.1B Chat model for testing
#
# TinyLlama is the simplest, most basic, and most widely tested small model.
# - Architecture: Standard Llama (no special features)
# - Size: 1.1B parameters
# - Format: Q4_K_M quantized (good balance of size/quality)
# - Well-tested and stable
# - Good instruction following despite small size

set -e

MODEL_DIR=".test-models/tinyllama"
MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# Create directory
mkdir -p "$MODEL_DIR"

# Check if already downloaded
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already exists: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    exit 0
fi

echo "📥 Downloading TinyLlama 1.1B Chat (Q4_K_M)..."
echo "   Size: ~669 MB"
echo "   Format: Q4_K_M quantized (good balance)"
echo "   Architecture: Standard Llama (simplest possible)"
echo ""

# Download from HuggingFace
HF_REPO="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
HF_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "$HF_REPO" "$HF_FILE" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    
    # Rename if needed
    if [ -f "$MODEL_DIR/$HF_FILE" ] && [ ! -f "$MODEL_PATH" ]; then
        mv "$MODEL_DIR/$HF_FILE" "$MODEL_PATH"
    fi
else
    echo "Using wget (huggingface-cli not found)..."
    
    # Construct HuggingFace URL
    HF_URL="https://huggingface.co/$HF_REPO/resolve/main/$HF_FILE"
    
    wget -O "$MODEL_PATH" "$HF_URL" || {
        echo "❌ Download failed. Trying alternative method..."
        
        # Try curl as fallback
        curl -L -o "$MODEL_PATH" "$HF_URL" || {
            echo "❌ All download methods failed."
            echo ""
            echo "Please download manually:"
            echo "  1. Visit: https://huggingface.co/$HF_REPO"
            echo "  2. Download: $HF_FILE"
            echo "  3. Save to: $MODEL_PATH"
            exit 1
        }
    }
fi

# Verify download
if [ -f "$MODEL_PATH" ]; then
    echo ""
    echo "✅ Download complete!"
    echo "   Path: $MODEL_PATH"
    echo "   Size: $(du -h "$MODEL_PATH" | cut -f1)"
    echo ""
    echo "Model info:"
    echo "  - Architecture: Standard Llama (simplest)"
    echo "  - Parameters: 1.1B"
    echo "  - Vocab: 32,000 tokens"
    echo "  - Hidden: 2048"
    echo "  - Layers: 22"
    echo "  - Context: 2048"
    echo "  - Format: Q4_K_M quantized"
    echo ""
    echo "Why TinyLlama?"
    echo "  ✓ Most basic/standard architecture"
    echo "  ✓ Well-tested and stable"
    echo "  ✓ Good instruction following"
    echo "  ✓ Fast inference"
    echo "  ✓ Small size (669 MB)"
    echo ""
    echo "Usage:"
    echo "  cargo test --test simple_generation_test --features cuda -- --ignored"
else
    echo "❌ Download verification failed"
    exit 1
fi
