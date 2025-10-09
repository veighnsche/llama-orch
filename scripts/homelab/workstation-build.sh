#!/usr/bin/env bash
# Homelab Build Script: Workstation (CUDA)
# Created by: TEAM-018
# Target: workstation.home.arpa (NVIDIA GPU)
# Backend: CUDA

set -euo pipefail

# TEAM-019: Hook pre-build telemetry here (start timestamp, git commit hash)
echo "════════════════════════════════════════════════════════════════"
echo "🖥️  WORKSTATION BUILD (CUDA Backend)"
echo "════════════════════════════════════════════════════════════════"
echo "Target: workstation.home.arpa"
echo "Backend: CUDA (NVIDIA GPU)"
echo "Started: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"

ssh -o BatchMode=yes -o ConnectTimeout=10 workstation.home.arpa 'bash -s' <<'EOF'
set -euo pipefail
  
  cd ~/Projects/llama-orch
  
  # TEAM-019: Capture git metadata for build correlation
  echo ""
  echo "📍 Repository Status"
  echo "────────────────────────────────────────────────────────────────"
  git log -1 --oneline
  echo "Branch: $(git branch --show-current)"
  echo ""
  
  # TEAM-019: Hook repo sync telemetry here (bytes pulled, duration)
  echo "🔄 Updating repository..."
  echo "────────────────────────────────────────────────────────────────"
  git fetch --all
  git reset --hard origin/main
  echo "✅ Repository updated to latest main"
  echo ""
  
  # TEAM-019: Hook build telemetry here (duration, binary size, warnings count)
  echo "🧱 Building CUDA backend..."
  echo "────────────────────────────────────────────────────────────────"
  cd bin/llorch-candled
  cargo build --release --features cuda --bin llorch-cuda-candled
  
  # TEAM-019: Capture build artifacts metadata
  echo ""
  echo "📦 Build Artifacts"
  echo "────────────────────────────────────────────────────────────────"
  ls -lh ../../target/release/llorch-cuda-candled
  echo ""
  
  echo "✅ CUDA build complete"
EOF

# TEAM-019: Hook post-build telemetry here (end timestamp, success/failure)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Workstation build completed successfully"
echo "Finished: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"
