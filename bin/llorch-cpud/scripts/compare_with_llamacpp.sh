#!/usr/bin/env bash
# TEAM-008: Compare llorch-cpud output with llama.cpp reference
#
# This script runs the same prompt through both implementations and compares results.
# Usage: ./compare_with_llamacpp.sh [prompt] [n_tokens]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLORCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${LLORCH_ROOT}/.test-models/llama2-7b/llama-2-7b.Q8_0.gguf"
LLAMA_CLI="${LLORCH_ROOT}/reference/llama.cpp/build/bin/llama-cli"

# Default test parameters
PROMPT="${1:-Hello}"
N_TOKENS="${2:-10}"
TEMP="0.0"  # Deterministic for comparison

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TEAM-008: llama.cpp vs llorch-cpud Comparison          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Model:    ${MODEL_PATH}"
echo "Prompt:   \"${PROMPT}\""
echo "Tokens:   ${N_TOKENS}"
echo "Temp:     ${TEMP} (deterministic)"
echo ""

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "❌ Model not found: ${MODEL_PATH}"
    echo "   Run: ./.docs/testing/download_llama2_7b_fp16.sh"
    exit 1
fi

# Check if llama-cli exists
if [ ! -f "${LLAMA_CLI}" ]; then
    echo "❌ llama-cli not found: ${LLAMA_CLI}"
    echo "   Run: cd reference/llama.cpp && cmake -B build && cmake --build build"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "Running llama.cpp (reference implementation)..."
echo "═══════════════════════════════════════════════════════════"
echo ""

# Run llama.cpp and capture output
LLAMACPP_OUTPUT=$(mktemp)
"${LLAMA_CLI}" \
    -m "${MODEL_PATH}" \
    -p "${PROMPT}" \
    -n "${N_TOKENS}" \
    --temp "${TEMP}" \
    --seed 42 \
    --no-display-prompt \
    2>/dev/null | tee "${LLAMACPP_OUTPUT}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "llama.cpp output saved to: ${LLAMACPP_OUTPUT}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# TODO: Run llorch-cpud when implemented
echo "⏳ llorch-cpud implementation not yet ready"
echo ""
echo "Next steps:"
echo "  1. Implement RMSNorm, RoPE, SwiGLU (Week 2)"
echo "  2. Implement attention + full model (Week 3)"
echo "  3. Run this script again to compare outputs"
echo ""
echo "Expected workflow:"
echo "  cargo run --release -- \\"
echo "    --model ${MODEL_PATH} \\"
echo "    --prompt \"${PROMPT}\" \\"
echo "    --n-predict ${N_TOKENS} \\"
echo "    --temp ${TEMP} \\"
echo "    --seed 42"
echo ""

# For now, just show what we got from llama.cpp
echo "═══════════════════════════════════════════════════════════"
echo "Reference output (llama.cpp):"
echo "═══════════════════════════════════════════════════════════"
cat "${LLAMACPP_OUTPUT}"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✅ Reference output captured"
echo "📝 Save this output for comparison when llorch-cpud is ready"
echo ""

# Keep the temp file for manual inspection
echo "Reference output file: ${LLAMACPP_OUTPUT}"
echo "(Will be deleted on next reboot)"
