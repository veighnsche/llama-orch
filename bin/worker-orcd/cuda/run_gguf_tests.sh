#!/bin/bash
# Run GGUF header parser tests

set -e

cd "$(dirname "$0")/build"

echo "=== Building CUDA tests ==="
cmake --build . --target cuda_tests 2>&1

echo ""
echo "=== Running all tests to verify GGUF tests are included ==="
./cuda_tests --gtest_list_tests 2>&1 | grep -i gguf || echo "WARNING: No GGUF tests found in binary"

echo ""
echo "=== Running all GGUF-related tests ==="
./cuda_tests --gtest_filter="*GGUF*:*gguf*" --gtest_color=yes 2>&1

echo ""
echo "=== Test run complete ==="
