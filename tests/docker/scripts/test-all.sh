#!/bin/bash
set -e

TOPOLOGY=${1:-localhost}

echo "🧪 Running Docker tests with topology: $TOPOLOGY"
echo ""

# Start environment
./tests/docker/scripts/start.sh $TOPOLOGY

# Run tests
echo ""
echo "🧪 Running smoke tests..."
cargo test --package xtask --test docker_smoke_test --ignored -- --nocapture

# Cleanup
./tests/docker/scripts/stop.sh $TOPOLOGY

echo ""
echo "✅ All tests passed!"
