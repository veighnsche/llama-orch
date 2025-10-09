#!/usr/bin/env bash
# Homelab Build Script: Mac (Metal)
# Created by: TEAM-018
# Target: mac.home.arpa (Apple Silicon)
# Backend: Metal (Pre-release)

set -euo pipefail

# TEAM-019: Hook pre-build telemetry here (start timestamp, git commit hash)
echo "════════════════════════════════════════════════════════════════"
echo "🍎 MAC BUILD (Metal Backend - Pre-release)"
echo "════════════════════════════════════════════════════════════════"
echo "Target: mac.home.arpa"
echo "Backend: Metal (Apple Silicon GPU)"
echo "Status: Pre-release (runtime validation in progress)"
echo "Started: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"

ssh -o BatchMode=yes -o ConnectTimeout=10 mac.home.arpa 'bash -s' <<'EOF'
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
  echo "🧱 Building Metal backend..."
  echo "────────────────────────────────────────────────────────────────"
  cd bin/llorch-candled
  cargo build --release --features metal --bin llorch-metal-candled
  
  # TEAM-019: Capture build artifacts metadata
  echo ""
  echo "📦 Build Artifacts"
  echo "────────────────────────────────────────────────────────────────"
  ls -lh ../../target/release/llorch-metal-candled
  echo ""
  
  echo "✅ Metal build complete"
EOF

# TEAM-019: Hook post-build telemetry here (end timestamp, success/failure)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Mac build completed successfully"
echo "Finished: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  NOTE: Metal backend is PRE-RELEASE"
echo "    Runtime validation in progress on Apple Silicon hardware"
echo "    See: bin/llorch-candled/docs/metal.md"
