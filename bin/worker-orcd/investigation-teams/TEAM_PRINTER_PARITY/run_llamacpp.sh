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
# CRITICAL: Redirect to file first, then grep afterward to avoid interactive pipe deadlock
# DO NOT pipe llama-cli directly to grep/head - it launches an interactive REPL
# DO NOT use --log-disable if you want tokenization/BOS/EOS details

echo "[TEAM PRINTER] Running llama.cpp..."
cd "$LLAMA_CPP_DIR"

./build/bin/llama-cli \
    --model "$MODEL_PATH" \
    --prompt "Write a haiku about GPU computing" \
    --temp 0.0 \
    --top-p 1.0 \
    --top-k 0 \
    --seed 12345 \
    --n-predict 50 \
    --verbose-prompt \
    > "$SCRIPT_DIR/llamacpp.run.log" 2>&1

echo ""
echo "[TEAM PRINTER] ============================================"
echo "[TEAM PRINTER] llama.cpp run complete"
echo "[TEAM PRINTER] Log saved to: $SCRIPT_DIR/llamacpp.run.log"
echo "[TEAM PRINTER] ============================================"
echo ""
echo "[TEAM PRINTER] Extracting tokenization details..."
echo ""

# Extract vocab/token info (grep the file, don't pipe the running process)
grep -E "vocab|token|special|bos|eos" "$SCRIPT_DIR/llamacpp.run.log" | head -30 || echo "(No tokenization details found)"

echo ""
echo "[TEAM PRINTER] ============================================"
echo "[TEAM PRINTER] Analysis Commands"
echo "[TEAM PRINTER] ============================================"
echo ""
echo "# Compare token IDs:"
echo "grep -i 'token' $SCRIPT_DIR/llamacpp.run.log | head -20"
echo ""
echo "# Check for special tokens:"
echo "grep -E '(im_start|im_end|bos|eos)' $SCRIPT_DIR/llamacpp.run.log"
echo ""
echo "# View generated text:"
echo "tail -50 $SCRIPT_DIR/llamacpp.run.log"
echo ""
echo "[TEAM PRINTER] NOTE: For full checkpoint parity data:"
echo "[TEAM PRINTER]   1. Patch llama.cpp to add checkpoint dumps (modify source)"
echo "[TEAM PRINTER]   2. Use HTTP API for pure batch mode (no interactive REPL)"
echo "[TEAM PRINTER]   3. Or use GDB to extract intermediate tensor values"
