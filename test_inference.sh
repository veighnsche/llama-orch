#!/bin/bash
# TEAM-035: End-to-end inference test script
# Tests the full flow: rbee-keeper -> rbee-hive -> llm-worker-rbee -> inference

set -e

echo "=== TEAM-035: End-to-End Inference Test ==="
echo ""

# Kill any existing processes
echo "Cleaning up any existing processes..."
pkill -f rbee-hive || true
pkill -f llm-worker-rbee || true
sleep 2

# Start rbee-hive (pool manager) in background
echo "Starting rbee-hive on localhost:8080..."
cargo run -p rbee-hive -- daemon --addr 127.0.0.1:8080 > /tmp/rbee-hive.log 2>&1 &
HIVE_PID=$!
echo "rbee-hive PID: $HIVE_PID"
sleep 3

# Check if rbee-hive is running
if ! curl -s http://127.0.0.1:8080/v1/health > /dev/null; then
    echo "ERROR: rbee-hive failed to start"
    cat /tmp/rbee-hive.log
    kill $HIVE_PID || true
    exit 1
fi

echo "✓ rbee-hive is running"
echo ""

# Run inference command
echo "Running inference command..."
echo ""
cargo run -p rbee-keeper -- infer \
    --node localhost \
    --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
    --prompt "Once upon a time" \
    --max-tokens 20 \
    --temperature 0.7

# Cleanup
echo ""
echo "Cleaning up..."
kill $HIVE_PID || true
pkill -f llm-worker-rbee || true

echo ""
echo "=== Test Complete ==="
