#!/bin/bash
# ============================================================================
# [TEAM PRINTER] 2025-10-07T01:24Z - Run llama.cpp with Checkpoint Logging
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_DIR="/home/vince/Projects/llama-orch/reference/llama.cpp"
MODEL_PATH="/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf"

echo "[TEAM PRINTER] ============================================"
echo "[TEAM PRINTER] Running llama.cpp"
echo "[TEAM PRINTER] ============================================"
echo ""

# Check if llama-cli exists
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-cli" ]; then
    echo "[TEAM PRINTER] ❌ ERROR: llama-cli not found at $LLAMA_CPP_DIR/build/bin/llama-cli"
    echo "[TEAM PRINTER] Please build llama.cpp first:"
    echo "  cd $LLAMA_CPP_DIR"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DGGML_CUDA=ON"
    echo "  cmake --build . --config Release"
    exit 1
fi

echo "[TEAM PRINTER] Test configuration:"
echo "  Model: Qwen2.5-0.5B FP16"
echo "  Prompt: 'Write a haiku about GPU computing'"
echo "  Temperature: 0.0 (greedy)"
echo "  Seed: 12345"
echo "  Tokens to generate: 2 (BOS context + 1 new token)"
echo ""

# Run llama.cpp with greedy sampling
# Note: llama.cpp doesn't have built-in checkpoint logging, so we'll capture its output
# and manually extract token IDs and decoded strings

echo "[TEAM PRINTER] Running llama.cpp..."
cd "$LLAMA_CPP_DIR"

./build/bin/llama-cli \
    --model "$MODEL_PATH" \
    --prompt "Write a haiku about GPU computing" \
    --temp 0.0 \
    --top-p 1.0 \
    --top-k 0 \
    --seed 12345 \
    --n-predict 2 \
    --log-disable \
    --verbose-prompt \
    2>&1 | tee "$SCRIPT_DIR/llamacpp.run.log"

echo ""
echo "[TEAM PRINTER] ============================================"
echo "[TEAM PRINTER] llama.cpp run complete"
echo "[TEAM PRINTER] Log saved to: $SCRIPT_DIR/llamacpp.run.log"
echo "[TEAM PRINTER] ============================================"
echo ""
echo "[TEAM PRINTER] NOTE: llama.cpp does not have built-in checkpoint logging."
echo "[TEAM PRINTER] To get full parity data, you would need to:"
echo "[TEAM PRINTER]   1. Modify llama.cpp source to add checkpoint dumps"
echo "[TEAM PRINTER]   2. Or use llama.cpp's --verbose flag and parse output"
echo "[TEAM PRINTER]   3. Or use a debugger to extract intermediate values"
echo ""
echo "[TEAM PRINTER] For now, we can at least compare:"
echo "[TEAM PRINTER]   - Token IDs generated"
echo "[TEAM PRINTER]   - Decoded UTF-8 strings"
echo "[TEAM PRINTER]   - Final output quality"
