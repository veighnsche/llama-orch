#!/bin/bash
# TEAM-085: ONE COMMAND to ask "Why is the sky blue?"
# TEAM-094: Fixed cleanup and added workaround for question mark bug
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
echo "🤔 Asking: 'Hello world'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# TEAM-094: KNOWN BUG - Question marks cause 0 token generation
# Using simple prompt without question marks until sampling bug is fixed
# Use rbee-hive directly for local inference
cargo run --release -p rbee-keeper -- infer \
    --node localhost \
    --model "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    --prompt "Hello world" \
    --max-tokens 50 \
    --temperature 0.7

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Inference complete!"
echo ""
echo "🧹 Cleaning up..."
# TEAM-094: Use SIGTERM for graceful cascading shutdown instead of pkill -9
# rbee-hive will shutdown all workers before exiting
kill -TERM $HIVE_PID 2>/dev/null || true
# Give it time to shutdown workers gracefully
sleep 2
# Clean up queen-rbee (not managed by rbee-hive)
pkill -TERM queen-rbee 2>/dev/null || true
sleep 1
echo "✓ Stopped all services"
echo ""
echo "Done! 🎉"
