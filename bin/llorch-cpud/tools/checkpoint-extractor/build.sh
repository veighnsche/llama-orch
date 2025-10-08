#!/usr/bin/env bash
# TEAM-007: Build script for checkpoint extractor
# Ensures llama.cpp is built before building the wrapper tool

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# TEAM-007: Navigate to llama.cpp from tools/checkpoint-extractor
# Path: tools/checkpoint-extractor -> tools -> llorch-cpud -> bin -> llama-orch -> reference/llama.cpp
LLAMA_CPP_DIR="$(cd "${SCRIPT_DIR}/../../../../reference/llama.cpp" && pwd)"
LLAMA_BUILD_DIR="${LLAMA_CPP_DIR}/build"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TEAM-007: Checkpoint Extractor Build Script            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Build llama.cpp if needed
if [ ! -f "${LLAMA_BUILD_DIR}/llama-config.cmake" ]; then
    echo "📦 Building llama.cpp (first time)..."
    echo "   Source: ${LLAMA_CPP_DIR}"
    echo "   Build:  ${LLAMA_BUILD_DIR}"
    mkdir -p "${LLAMA_BUILD_DIR}"
    cd "${LLAMA_BUILD_DIR}"
    cmake "${LLAMA_CPP_DIR}" -DBUILD_SHARED_LIBS=ON
    make -j$(nproc)
    echo "✅ llama.cpp built successfully"
    echo ""
else
    echo "✅ llama.cpp already built at ${LLAMA_BUILD_DIR}"
    echo ""
fi

# Step 2: Build wrapper tool
echo "🔧 Building checkpoint extractor..."
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

cmake .. -DCMAKE_PREFIX_PATH="${LLAMA_BUILD_DIR}"
make -j$(nproc)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ Build Complete                                       ║"
echo "║  Binary: ${SCRIPT_DIR}/build/llorch-checkpoint-extractor ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Usage:"
echo "  ./build/llorch-checkpoint-extractor <model.gguf> <prompt> [output_dir]"
echo ""
