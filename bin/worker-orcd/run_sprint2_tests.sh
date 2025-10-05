#!/bin/bash
# Run all Sprint 2 (GGUF-BPE Tokenizer) tests
# Stories: LT-007 through LT-010

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Sprint 2: GGUF-BPE Tokenizer - Test Suite"
echo "=========================================="
echo ""

echo "=== Running Rust tests for tokenizer module ==="
cargo test --lib tokenizer -- --nocapture --test-threads=1 2>&1

echo ""
echo "=========================================="
echo "Sprint 2 Test Run Complete"
echo "=========================================="
