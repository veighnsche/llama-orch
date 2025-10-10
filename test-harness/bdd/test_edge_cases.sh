#!/bin/bash
# TEAM-060: Quick test script for edge case scenarios
# Tests that real command execution works

set -e

cd "$(dirname "$0")"

echo "🧪 TEAM-060: Testing real edge case command execution"
echo "=================================================="

# Clean up any existing processes
pkill -9 -f "bdd-runner|mock-worker|queen-rbee" 2>/dev/null || true
sleep 1

# Build binaries
echo "📦 Building binaries..."
cargo build --bin bdd-runner --bin mock-worker 2>&1 | grep -E "(Compiling|Finished)" || true

echo ""
echo "🧪 Running edge case tests..."
echo ""

# Run specific edge case scenarios with timeout
timeout 60 cargo run --bin bdd-runner -- \
    --tags "@edge-case" \
    2>&1 | tee /tmp/edge_case_test.log || {
    echo "⚠️  Test timed out or failed"
    pkill -9 -f "bdd-runner|mock-worker|queen-rbee" 2>/dev/null || true
}

echo ""
echo "📊 Test Results:"
grep -E "(scenarios|steps)" /tmp/edge_case_test.log | tail -2 || echo "No results found"

# Cleanup
pkill -9 -f "bdd-runner|mock-worker|queen-rbee" 2>/dev/null || true

echo ""
echo "✅ Test complete"
