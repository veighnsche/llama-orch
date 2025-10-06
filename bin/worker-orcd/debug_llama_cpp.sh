#!/bin/bash
set -e

echo "=== Rebuilding llama.cpp with debug instrumentation ==="

cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Clean previous build
rm -rf build

# Create build directory
mkdir -p build
cd build

# Configure with CUDA support, disable unnecessary features, enable ccache
cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Build (only the main binary, not tests)
cmake --build . --config Release --target llama-cli -j$(nproc)

echo ""
echo "=== Build complete ==="
echo ""
echo "=== Running test with Qwen2.5-0.5B model ==="
echo ""

# Run with the same prompt as our test
./bin/llama-cli \
  -m /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku about the number 46 and CUDA" \
  -n 20 \
  --temp 0.0 \
  --seed 42 \
  2>&1 | tee /home/vince/Projects/llama-orch/bin/worker-orcd/llama_cpp_debug.log

echo ""
echo "=== Test complete ==="
echo "Debug output saved to: bin/worker-orcd/llama_cpp_debug.log"
echo ""
echo "Key sections to check:"
echo "  grep 'LLAMA.CPP Q DEBUG' llama_cpp_debug.log"
echo "  grep 'ATTN SCORE' llama_cpp_debug.log"
echo "  grep 'SOFTMAX' llama_cpp_debug.log"
