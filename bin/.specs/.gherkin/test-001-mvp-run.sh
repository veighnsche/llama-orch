#!/usr/bin/env bash
# Test-001 MVP: Cross-Node Inference
# Per test-001-mvp.md
#
# ARCHITECTURE (TEAM-030):
#   - Worker registry: In-memory (ephemeral, no SQLite)
#   - Model catalog: SQLite (persistent, ~/.rbee/models.db)
#
# Created by: TEAM-027
# Modified by: TEAM-030

set -euo pipefail

echo "=== Test-001 MVP: Cross-Node Inference ==="
echo ""

# Configuration
NODE="mac"
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
cargo build --bin rbee-hive --bin rbee || {
    echo "❌ Build failed"
    exit 1
}
echo "✅ Binaries built"
echo ""

# Step 2: Start rbee-hive daemon on mac (via SSH)
echo "[Step 2] Starting rbee-hive daemon on $NODE..."
ssh "$NODE.home.arpa" "cd ~/Projects/llama-orch && ./target/debug/rbee-hive daemon" &
DAEMON_PID=$!
echo "  Daemon PID: $DAEMON_PID"
sleep 5
echo "✅ Daemon started"
echo ""

# Step 3: Verify health
echo "[Step 3] Checking pool health..."
curl -s "http://$NODE.home.arpa:8080/v1/health" | jq . || {
    echo "❌ Health check failed"
    kill $DAEMON_PID 2>/dev/null || true
    exit 1
}
echo "✅ Pool healthy"
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
    kill $DAEMON_PID 2>/dev/null || true
    exit 1
}
echo ""

# Step 5: Cleanup
echo "[Step 5] Cleaning up..."
kill $DAEMON_PID 2>/dev/null || true
echo "✅ Cleanup complete"
echo ""

echo "✅ Test-001 MVP PASSED"
