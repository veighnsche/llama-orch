#!/bin/bash
# ============================================================================
# [TEAM PRINTER] 2025-10-07T01:24Z - Run Our Engine with Checkpoint Logging
# ============================================================================

set -e

cd "$(dirname "$0")/../../.."

echo "[TEAM PRINTER] ============================================"
echo "[TEAM PRINTER] Running our CUDA/Rust engine"
echo "[TEAM PRINTER] ============================================"
echo ""

# Set environment variables for checkpoint logging
export PRINTER_CHECKPOINT_LOGGING=1
export PRINTER_TOKEN_LIMIT=2
export PRINTER_OUTPUT_PATH="investigation-teams/TEAM_PRINTER_PARITY/ours.checkpoints"

# Run the haiku test with greedy sampling
export REQUIRE_REAL_LLAMA=1

echo "[TEAM PRINTER] Test configuration:"
echo "  Model: Qwen2.5-0.5B FP16"
echo "  Prompt: 'Write a haiku about GPU computing'"
echo "  Temperature: 0.0 (greedy)"
echo "  Seed: 12345"
echo "  Tokens to capture: 0, 1"
echo ""

# Build if needed
if [ ! -f "target/release/worker-orcd" ]; then
    echo "[TEAM PRINTER] Building worker-orcd..."
    cargo build --release --features cuda
fi

# Run test with checkpoint logging
echo "[TEAM PRINTER] Running test..."
cargo test --release --features cuda --test haiku_generation_anti_cheat \
    test_haiku_generation_stub_pipeline_only \
    -- --ignored --nocapture --test-threads=1 2>&1 | tee investigation-teams/TEAM_PRINTER_PARITY/ours.run.log

echo ""
echo "[TEAM PRINTER] ============================================"
echo "[TEAM PRINTER] Our engine run complete"
echo "[TEAM PRINTER] Log saved to: investigation-teams/TEAM_PRINTER_PARITY/ours.run.log"
echo "[TEAM PRINTER] ============================================"
