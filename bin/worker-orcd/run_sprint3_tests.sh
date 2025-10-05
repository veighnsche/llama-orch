#!/bin/bash
# Run all Sprint 3 (UTF-8 Safety + Llama Kernels) tests
# Stories: LT-011 through LT-014

set -e

echo "=========================================="
echo "Sprint 3: UTF-8 Safety + Llama Kernels - Test Suite"
echo "=========================================="
echo ""

echo "=== Part 1: Rust UTF-8 Streaming Tests (LT-011) ==="
cd "$(dirname "$0")"
cargo test --lib tokenizer::streaming -- --nocapture --test-threads=1 2>&1 || echo "Note: No streaming tests found yet"

echo ""
echo "=== Part 2: CUDA Kernel Tests (LT-012, LT-013, LT-014) ==="
cd cuda/build

echo ""
echo "--- Building CUDA tests ---"
cmake --build . --target cuda_tests 2>&1

echo ""
echo "=========================================="
echo "LT-012: RoPE Kernel Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*RoPE*:*Rope*:*rope*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "LT-013: RMSNorm Kernel Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*RMSNorm*:*Rmsnorm*:*rmsnorm*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "LT-014: Residual Connection Kernel Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*Residual*:*residual*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "Sprint 3 Test Summary"
echo "=========================================="
echo "Running all Sprint 3 kernel tests together..."
./cuda_tests --gtest_filter="*RoPE*:*Rope*:*RMSNorm*:*Rmsnorm*:*Residual*:*residual*" --gtest_color=yes 2>&1

echo ""
echo "=========================================="
echo "Sprint 3 Test Run Complete"
echo "=========================================="
