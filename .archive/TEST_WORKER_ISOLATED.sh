#!/bin/bash
# Test worker in isolation - verify it can start and callback to rbee-hive

set -e

echo "🧪 Testing worker in isolation"
echo ""

# Step 1: Start rbee-hive on port 9200
echo "📦 Starting rbee-hive on port 9200..."
cargo run --release -p rbee-hive -- daemon --addr 127.0.0.1:9200 > /tmp/rbee-hive-test.log 2>&1 &
HIVE_PID=$!
echo "✓ rbee-hive started (PID: $HIVE_PID)"

# Wait for rbee-hive to be ready
echo "⏳ Waiting for rbee-hive..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:9200/v1/health > /dev/null 2>&1; then
        echo "✓ rbee-hive is ready!"
        break
    fi
    sleep 0.5
done

# Step 2: Manually spawn a worker via rbee-hive API
echo ""
echo "🚀 Spawning worker via rbee-hive API..."
SPAWN_RESPONSE=$(curl -s -X POST http://127.0.0.1:9200/v1/workers/spawn \
    -H "Content-Type: application/json" \
    -d '{
        "model_ref": "hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "backend": "cpu",
        "device": 0,
        "model_path": ""
    }')

echo "Response: $SPAWN_RESPONSE"
WORKER_URL=$(echo $SPAWN_RESPONSE | jq -r '.url')
WORKER_ID=$(echo $SPAWN_RESPONSE | jq -r '.worker_id')

echo "✓ Worker spawned:"
echo "  ID: $WORKER_ID"
echo "  URL: $WORKER_URL"

# Step 3: Wait and check if worker becomes ready
echo ""
echo "⏳ Waiting for worker to become ready (checking logs)..."
sleep 5

# Step 4: Check rbee-hive logs for worker callback
echo ""
echo "📋 Checking rbee-hive logs for worker callback..."
if grep -q "callback_ready" /tmp/rbee-hive-test.log; then
    echo "✅ Worker callback detected in logs!"
    grep "callback_ready" /tmp/rbee-hive-test.log | tail -5
else
    echo "❌ No worker callback found in logs"
    echo ""
    echo "Last 20 lines of rbee-hive log:"
    tail -20 /tmp/rbee-hive-test.log
fi

# Step 5: Try to reach worker directly
echo ""
echo "🔍 Checking if worker is reachable at $WORKER_URL..."
if curl -s "$WORKER_URL/v1/health" > /dev/null 2>&1; then
    echo "✅ Worker is reachable!"
    curl -s "$WORKER_URL/v1/health" | jq .
else
    echo "❌ Worker is not reachable"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."
kill $HIVE_PID 2>/dev/null || true
pkill -9 llm-worker-rbee 2>/dev/null || true
echo "✓ Cleanup complete"
echo ""
echo "Done! 🎉"
