#!/usr/bin/env bash
set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  llorch-cpud LayerNorm Validation Suite                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd "$(dirname "$0")/.."

echo -e "${BLUE}[1/3]${NC} Running llorch-cpud LayerNorm test..."
cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_all -- --nocapture 2>&1 | tail -20
echo ""

echo -e "${BLUE}[2/3]${NC} Running Candle reference implementation..."
cd .test_helpers/candle_ln_test
cargo run --release 2>&1 | grep -A 10 "CANDLE LAYERNORM"
cd ../..
echo ""

echo -e "${BLUE}[3/3]${NC} Comparing outputs..."
python3 .test_helpers/compare_outputs.py
echo ""

echo -e "${GREEN}✅ Validation complete!${NC}"
echo ""
echo "See VALIDATION_SUMMARY.md for full details."
