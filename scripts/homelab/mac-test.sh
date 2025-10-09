#!/usr/bin/env bash
# Homelab Test Script: Mac (Metal)
# Created by: TEAM-018
# Target: mac.home.arpa (Apple Silicon)
# Backend: Metal (Pre-release)

set -euo pipefail

# TEAM-019: Hook pre-test telemetry here (start timestamp, test plan)
echo "════════════════════════════════════════════════════════════════"
echo "🧪 MAC TEST (Metal Backend - Pre-release)"
echo "════════════════════════════════════════════════════════════════"
echo "Target: mac.home.arpa"
echo "Backend: Metal (Apple Silicon GPU)"
echo "Status: Pre-release (runtime validation in progress)"
echo "Started: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"

ssh mac.home.arpa <<'EOF'
  set -euo pipefail
  
  cd ~/Projects/llama-orch
  
  # TEAM-019: Capture environment metadata (macOS version, chip model, Metal support)
  echo ""
  echo "🔧 Environment Information"
  echo "────────────────────────────────────────────────────────────────"
  echo "macOS Version:"
  sw_vers
  echo ""
  echo "Hardware:"
  sysctl -n machdep.cpu.brand_string
  echo ""
  echo "Metal Support:"
  system_profiler SPDisplaysDataType | grep -A 2 "Metal" || echo "Metal info not available via system_profiler"
  echo ""
  
  # TEAM-019: Hook test execution telemetry here (test count, duration, pass/fail)
  echo "🧪 Running Metal tests..."
  echo "────────────────────────────────────────────────────────────────"
  cd bin/llorch-candled
  
  # Run tests with full output (no capture suppression)
  cargo test --features metal -- --nocapture
  
  # TEAM-019: Capture test results metadata (tests run, passed, failed, ignored)
  echo ""
  echo "✅ Metal tests complete"
EOF

# TEAM-019: Hook post-test telemetry here (end timestamp, test summary)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Mac tests completed successfully"
echo "Finished: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  NOTE: Metal backend is PRE-RELEASE"
echo "    These tests validate code structure and compilation"
echo "    Full runtime validation (model loading, inference) pending"
echo "    See: bin/llorch-candled/docs/metal.md"
