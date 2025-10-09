#!/usr/bin/env bash
# Homelab Test Script: Workstation (CUDA)
# Created by: TEAM-018
# Target: workstation.home.arpa (NVIDIA GPU)
# Backend: CUDA

set -euo pipefail

# TEAM-019: Hook pre-test telemetry here (start timestamp, test plan)
echo "════════════════════════════════════════════════════════════════"
echo "🧪 WORKSTATION TEST (CUDA Backend)"
echo "════════════════════════════════════════════════════════════════"
echo "Target: workstation.home.arpa"
echo "Backend: CUDA (NVIDIA GPU)"
echo "Started: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"

ssh workstation.home.arpa <<'EOF'
  set -euo pipefail
  
  cd ~/Projects/llama-orch
  
  # TEAM-019: Capture environment metadata (GPU info, CUDA version)
  echo ""
  echo "🔧 Environment Information"
  echo "────────────────────────────────────────────────────────────────"
  echo "CUDA Version:"
  nvcc --version | grep "release" || echo "nvcc not in PATH"
  echo ""
  echo "GPU Information:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi not available"
  echo ""
  
  # TEAM-019: Hook test execution telemetry here (test count, duration, pass/fail)
  echo "🧪 Running CUDA tests..."
  echo "────────────────────────────────────────────────────────────────"
  cd bin/llorch-candled
  
  # Run tests with full output (no capture suppression)
  cargo test --features cuda -- --nocapture
  
  # TEAM-019: Capture test results metadata (tests run, passed, failed, ignored)
  echo ""
  echo "✅ CUDA tests complete"
EOF

# TEAM-019: Hook post-test telemetry here (end timestamp, test summary)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Workstation tests completed successfully"
echo "Finished: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"
