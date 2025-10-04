#!/bin/bash
# Run Llama metadata extraction tests

set -e

cd "$(dirname "$0")/build"

echo "=== Building CUDA tests ==="
cmake --build . --target cuda_tests 2>&1

echo ""
echo "=== Running all tests to verify Llama metadata tests are included ==="
./cuda_tests --gtest_list_tests 2>&1 | grep -i llama || echo "WARNING: No Llama tests found in binary"

echo ""
echo "=== Running all Llama metadata tests ==="
./cuda_tests --gtest_filter="*Llama*:*llama*" --gtest_color=yes 2>&1

echo ""
echo "=== Test run complete ==="
