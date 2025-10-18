#!/bin/bash
# Isolation test for llm-worker-rbee
# Tests worker startup, model loading, and callback without full orchestration

set -e

echo "🧪 llm-worker-rbee Isolation Test"
echo "=================================="
echo ""

# Configuration
WORKER_ID="test-worker-$(uuidgen)"
MODEL_PATH="../../.test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_REF="hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
BACKEND="cpu"
DEVICE=0
PORT=18081
CALLBACK_PORT=19200
CALLBACK_URL="http://127.0.0.1:${CALLBACK_PORT}/v1/workers/ready"

# Step 1: Start mock callback server
echo "📡 Starting mock callback server on port ${CALLBACK_PORT}..."
python3 -c "
import http.server
import socketserver
import json
import sys

class CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/v1/workers/ready':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode('utf-8'))
            
            print('✅ Received callback from worker:', file=sys.stderr)
            print(f'   Worker ID: {payload.get(\"worker_id\")}', file=sys.stderr)
            print(f'   URL: {payload.get(\"url\")}', file=sys.stderr)
            print(f'   Model Ref: {payload.get(\"model_ref\")}', file=sys.stderr)
            print(f'   Backend: {payload.get(\"backend\")}', file=sys.stderr)
            print(f'   Device: {payload.get(\"device\")}', file=sys.stderr)
            
            # Validate payload structure
            required_fields = ['worker_id', 'url', 'model_ref', 'backend', 'device']
            missing = [f for f in required_fields if f not in payload]
            if missing:
                print(f'❌ Missing fields: {missing}', file=sys.stderr)
                self.send_response(422)
                self.end_headers()
                self.wfile.write(json.dumps({'error': f'Missing fields: {missing}'}).encode())
                return
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'message': 'Worker registered'}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

with socketserver.TCPServer(('127.0.0.1', ${CALLBACK_PORT}), CallbackHandler) as httpd:
    print('Mock callback server ready', file=sys.stderr)
    httpd.serve_forever()
" > /tmp/worker-callback-server.log 2>&1 &
CALLBACK_PID=$!

# Wait for callback server
sleep 1
if ! kill -0 $CALLBACK_PID 2>/dev/null; then
    echo "❌ Failed to start callback server"
    cat /tmp/worker-callback-server.log
    exit 1
fi
echo "✓ Mock callback server started (PID: $CALLBACK_PID)"

# Step 2: Check if model exists
echo ""
echo "📦 Checking model file..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Run: cargo run -p rbee-hive -- models download hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    kill $CALLBACK_PID 2>/dev/null || true
    exit 1
fi
echo "✓ Model found: $MODEL_PATH"

# Step 3: Start worker
echo ""
echo "🚀 Starting worker..."
echo "   Worker ID: $WORKER_ID"
echo "   Port: $PORT"
echo "   Callback URL: $CALLBACK_URL"
echo ""

cargo run --release --bin llm-worker-rbee -- \
    --worker-id "$WORKER_ID" \
    --model "$MODEL_PATH" \
    --model-ref "$MODEL_REF" \
    --backend "$BACKEND" \
    --device $DEVICE \
    --port $PORT \
    --callback-url "$CALLBACK_URL" \
    > /tmp/worker-test.log 2>&1 &
WORKER_PID=$!

echo "✓ Worker started (PID: $WORKER_PID)"

# Step 4: Wait for worker to load and callback
echo ""
echo "⏳ Waiting for worker to load model and callback..."
sleep 3

# Check callback server logs
echo ""
echo "📋 Callback server logs:"
tail -20 /tmp/worker-callback-server.log

# Step 5: Check if worker is reachable
echo ""
echo "🔍 Testing worker HTTP endpoints..."

# Test health endpoint
if curl -s -f "http://127.0.0.1:${PORT}/v1/health" > /dev/null 2>&1; then
    echo "✅ Health endpoint responding"
    curl -s "http://127.0.0.1:${PORT}/v1/health" | jq .
else
    echo "❌ Health endpoint not responding"
fi

# Test ready endpoint
echo ""
if curl -s -f "http://127.0.0.1:${PORT}/v1/ready" > /dev/null 2>&1; then
    echo "✅ Ready endpoint responding"
    curl -s "http://127.0.0.1:${PORT}/v1/ready" | jq .
else
    echo "❌ Ready endpoint not responding"
fi

# Step 6: Test inference (simple prompt)
echo ""
echo "🤔 Testing inference..."
INFERENCE_RESPONSE=$(curl -s -X POST "http://127.0.0.1:${PORT}/v1/inference" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello, how are you?",
        "max_tokens": 20,
        "temperature": 0.7,
        "stream": false
    }' 2>&1)

if echo "$INFERENCE_RESPONSE" | jq . > /dev/null 2>&1; then
    echo "✅ Inference endpoint responding"
    echo "$INFERENCE_RESPONSE" | jq .
else
    echo "❌ Inference failed or returned invalid JSON"
    echo "$INFERENCE_RESPONSE"
fi

# Step 7: Show worker logs
echo ""
echo "📋 Worker logs (last 30 lines):"
tail -30 /tmp/worker-test.log

# Cleanup
echo ""
echo "🧹 Cleaning up..."
kill $WORKER_PID 2>/dev/null || true
kill $CALLBACK_PID 2>/dev/null || true
echo "✓ Cleanup complete"

echo ""
echo "✅ Isolation test complete!"
