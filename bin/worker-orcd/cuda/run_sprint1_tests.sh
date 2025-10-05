#!/bin/bash
# Run all Sprint 1 (GGUF Foundation) tests
# Stories: LT-001 through LT-006

set -e

cd "$(dirname "$0")/build"

echo "=========================================="
echo "Sprint 1: GGUF Foundation - Test Suite"
echo "=========================================="
echo ""

echo "=== Building CUDA tests ==="
cmake --build . --target cuda_tests 2>&1

echo ""
echo "=== Listing all Sprint 1 test suites ==="
./cuda_tests --gtest_list_tests 2>&1 | grep -E "(GGUF|Llama|Mmap|Chunked|PreLoad|Architecture)" || echo "Checking test suites..."

echo ""
echo "=========================================="
echo "LT-001: GGUF Header Parser Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*GGUF*:*gguf*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "LT-002: Llama Metadata Extraction Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*Llama*:*llama*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "LT-003: Memory-Mapped I/O Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*Mmap*:*mmap*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "LT-004: Chunked Transfer Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*Chunked*:*chunked*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "LT-005: Pre-Load Validation Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*PreLoad*:*Validation*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "LT-006: Architecture Detection Tests"
echo "=========================================="
./cuda_tests --gtest_filter="*Architecture*:*Detect*" --gtest_color=yes 2>&1 || true

echo ""
echo "=========================================="
echo "Sprint 1 Test Summary"
echo "=========================================="
echo "Running full test suite to get final counts..."
./cuda_tests --gtest_filter="*GGUF*:*Llama*:*Mmap*:*Chunked*:*PreLoad*:*Architecture*" --gtest_color=yes 2>&1

echo ""
echo "=========================================="
echo "Sprint 1 Test Run Complete"
echo "=========================================="
