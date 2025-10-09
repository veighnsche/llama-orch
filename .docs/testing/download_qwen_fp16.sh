#!/bin/bash
# Download Qwen 2.5 0.5B FP16 model for testing
#
# This downloads an FP16 (unquantized) version of Qwen 2.5 0.5B
# which bypasses all quantization issues and should load directly.
#
# Model: Qwen/Qwen2.5-0.5B-Instruct (FP16 GGUF)
# Size: ~1.1 GB (unquantized)
# Format: FP16 (no dequantization needed)

set -e

MODEL_DIR=".test-models/qwen"
MODEL_FILE="qwen2.5-0.5b-instruct-fp16.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# Create directory
mkdir -p "$MODEL_DIR"

# Check if already downloaded
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already exists: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    exit 0
fi

echo "📥 Downloading Qwen 2.5 0.5B Instruct (FP16)..."
echo "   Size: ~1.1 GB"
echo "   Format: FP16 (unquantized)"
echo ""

# Download from HuggingFace
# Using Qwen2.5-0.5B-Instruct FP16 GGUF
HF_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
HF_FILE="qwen2.5-0.5b-instruct-fp16.gguf"

if command -v hf &> /dev/null; then
    echo "Using hf download (modern CLI)..."
    hf download "$HF_REPO" "$HF_FILE" \
        --local-dir "$MODEL_DIR"
    
    # Rename if needed
    if [ -f "$MODEL_DIR/$HF_FILE" ] && [ ! -f "$MODEL_PATH" ]; then
        mv "$MODEL_DIR/$HF_FILE" "$MODEL_PATH"
    fi
else
    echo "Using wget (hf command not found)..."
    
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
    echo "  - Format: FP16 (no quantization)"
    echo "  - Vocab: 151,936 tokens"
    echo "  - Hidden: 896"
    echo "  - Layers: 24"
    echo "  - Context: 32K"
    echo ""
    echo "Usage:"
    echo "  cargo test --test haiku_generation_anti_cheat --features cuda -- --ignored"
else
    echo "❌ Download verification failed"
    exit 1
fi
