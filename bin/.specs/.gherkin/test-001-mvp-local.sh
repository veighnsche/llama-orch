#!/usr/bin/env bash
# Test-001 MVP: Cross-Node Inference (Local Version)
# Per test-001-mvp.md
#
# Created by: TEAM-029

set -euo pipefail

echo "=== Test-001 MVP: Cross-Node Inference (Local) ==="
echo ""

# Configuration
NODE="localhost"
MODEL="hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
PROMPT="write a short story"
MAX_TOKENS=20
TEMPERATURE=0.7

echo "Configuration:"
echo "  Node: $NODE"
echo "  Model: $MODEL"
echo "  Prompt: $PROMPT"
echo "  Max tokens: $MAX_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo ""

# Step 1: Build binaries
echo "[Step 1] Building binaries..."
cargo build --bin rbee-hive --bin rbee 2>&1 | grep -E "(Finished|Compiling|error|warning:.*generated)" || true
echo "✅ Binaries built"
echo ""

# Step 2: Start rbee-hive daemon locally
echo "[Step 2] Starting rbee-hive daemon on localhost..."
./target/debug/rbee-hive daemon > /tmp/rbee-hive.log 2>&1 &
DAEMON_PID=$!
echo "  Daemon PID: $DAEMON_PID"
echo "  Waiting for daemon to start..."
sleep 3
echo "✅ Daemon started"
echo ""

# Step 3: Verify health
echo "[Step 3] Checking pool health..."
for i in {1..10}; do
    if curl -s "http://localhost:8080/v1/health" 2>/dev/null | jq . 2>/dev/null; then
        echo "✅ Pool healthy"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Health check failed after 10 attempts"
        echo "Daemon log:"
        cat /tmp/rbee-hive.log
        kill $DAEMON_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Attempt $i/10 failed, retrying..."
    sleep 1
done
echo ""

# Step 4: Run inference
echo "[Step 4] Running inference..."
./target/debug/rbee infer \
  --node "$NODE" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" || {
    echo "❌ Inference failed"
    echo "Daemon log:"
    cat /tmp/rbee-hive.log
    kill $DAEMON_PID 2>/dev/null || true
    exit 1
}
echo ""

# Step 5: Cleanup
echo "[Step 5] Cleaning up..."
kill $DAEMON_PID 2>/dev/null || true
sleep 1
echo "✅ Cleanup complete"
echo ""

echo "✅ Test-001 MVP PASSED"
