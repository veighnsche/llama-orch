#!/bin/bash
# TEAM-085: ONE COMMAND to ask "Why is the sky blue?"
# This script demonstrates the CORRECT local inference flow

set -e

echo "🚀 Starting local inference: 'Why is the sky blue?'"
echo ""

# Step 1: Start rbee-hive (pool manager) in background
echo "📦 Starting rbee-hive pool manager..."
cargo run --release -p rbee-hive -- daemon --addr 127.0.0.1:9200 > /tmp/rbee-hive.log 2>&1 &
HIVE_PID=$!
echo "✓ rbee-hive started (PID: $HIVE_PID)"

# Wait for rbee-hive to be ready
echo "⏳ Waiting for rbee-hive to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:9200/v1/health > /dev/null 2>&1; then
        echo "✓ rbee-hive is ready!"
        break
    fi
    sleep 0.5
done

# Step 2: Run inference (this will auto-start queen-rbee)
echo ""
echo "🤔 Asking: 'Why is the sky blue?'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Use rbee-hive directly for local inference
cargo run --release -p rbee-keeper -- infer \
    --node localhost \
    --model "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 150 \
    --temperature 0.7

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Inference complete!"
echo ""
echo "🧹 Cleaning up..."
kill $HIVE_PID 2>/dev/null || true
echo "✓ Stopped rbee-hive"
echo ""
echo "Done! 🎉"
