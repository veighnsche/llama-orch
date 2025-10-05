#!/bin/bash
# Download Phi-3-Mini-4K-Instruct GGUF model for testing
# 
# Usage: ./download_phi3.sh
# 
# This script downloads the Phi-3-Mini-4K-Instruct Q4_K_M quantized model
# to .test-models/phi3/ as per TEST_MODELS.md guidelines.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_ROOT/.test-models/phi3"
MODEL_FILE="phi-3-mini-4k-instruct-q4.gguf"
MODEL_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

echo "=========================================="
echo "Phi-3-Mini-4K-Instruct Model Download"
echo "=========================================="
echo ""
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Size: ~2.4 GB"
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

echo "Downloading Phi-3-Mini-4K-Instruct (Q4_K_M)..."
echo "This will take a few minutes depending on your connection."
echo ""

# Try huggingface-cli first (recommended)
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli (recommended)..."
    cd "$MODEL_DIR"
    huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
        Phi-3-mini-4k-instruct-q4.gguf \
        --local-dir . \
        --local-dir-use-symlinks False
    
    echo ""
    echo "✅ Download complete!"
    
# Fallback to wget
elif command -v wget &> /dev/null; then
    echo "Using wget..."
    wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    
    echo ""
    echo "✅ Download complete!"
    
# Fallback to curl
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    
    echo ""
    echo "✅ Download complete!"
    
else
    echo "❌ Error: No download tool found (huggingface-cli, wget, or curl)"
    echo ""
    echo "Install one of:"
    echo "  - huggingface-cli: pip install huggingface-hub"
    echo "  - wget: sudo pacman -S wget"
    echo "  - curl: sudo pacman -S curl"
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
echo "✅ Phi-3 model ready for testing!"
echo "=========================================="
echo ""
echo "Location: $MODEL_DIR/$MODEL_FILE"
echo ""
echo "Next steps:"
echo "  1. Run worker-orcd tests: cargo test -p worker-orcd"
echo "  2. Run CUDA tests: cd bin/worker-orcd/cuda && bash run_sprint1_tests.sh"
echo "  3. See .test-models/phi3/README.md for usage examples"
