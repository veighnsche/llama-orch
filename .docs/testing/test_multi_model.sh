#!/usr/bin/env bash
# Multi-model architecture test
# Created by: TEAM-020
# Tests all supported architectures on current backend

set -euo pipefail

BACKEND="${1:-cpu}"
MODELS_DIR="${2:-.test-models}"

echo "Testing multi-model support on $BACKEND backend"
echo "================================================"

# Test results
PASSED=0
FAILED=0
SKIPPED=0

test_model() {
    local arch="$1"
    local model_dir="$2"
    local model_file="$3"
    
    echo ""
    echo "Testing $arch..."
    
    if [[ ! -f "$model_dir/$model_file" ]]; then
        echo "⚠️  Model not found: $model_dir/$model_file (skipping)"
        ((SKIPPED++))
        return
    fi
    
    # Start worker
    ./target/release/llorch-${BACKEND}-candled \
        --worker-id "test-$arch" \
        --model "$model_dir" \
        --port 9876 \
        --callback-url http://localhost:9999 &
    WORKER_PID=$!
    
    sleep 5
    
    # Test inference
    RESPONSE=$(curl -s -X POST http://localhost:9876/execute \
        -H "Content-Type: application/json" \
        -d '{"job_id":"test","prompt":"Hello","max_tokens":10,"seed":42}')
    
    kill $WORKER_PID 2>/dev/null || true
    wait $WORKER_PID 2>/dev/null || true
    
    if echo "$RESPONSE" | grep -q '"type":"token"'; then
        echo "✅ $arch works on $BACKEND"
        ((PASSED++))
    else
        echo "❌ $arch failed on $BACKEND"
        echo "Response: $RESPONSE"
        ((FAILED++))
    fi
}

# Test each architecture
# Note: These are GGUF models, not SafeTensors
# llorch-candled currently only supports SafeTensors
# This script is for future use when GGUF support is added

echo ""
echo "⚠️  NOTE: llorch-candled currently only supports SafeTensors format"
echo "    The downloaded models are GGUF format and won't work yet"
echo "    This script is prepared for future GGUF support"
echo ""

# Uncomment when SafeTensors models are available:
# test_model "llama" "$MODELS_DIR/tinyllama" "model.safetensors"
# test_model "mistral" "$MODELS_DIR/mistral" "model.safetensors"
# test_model "phi" "$MODELS_DIR/phi3" "model.safetensors"
# test_model "qwen" "$MODELS_DIR/qwen" "model.safetensors"

# Summary
echo ""
echo "================================================"
echo "Test Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "================================================"

[[ $FAILED -eq 0 ]]
