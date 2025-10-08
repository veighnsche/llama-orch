#!/usr/bin/env bash
set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  llorch-cpud LayerNorm Validation Suite                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

cd "$(dirname "$0")/.."

echo -e "${BLUE}[1/4]${NC} Running llorch-cpud LayerNorm test..."
cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_all -- --nocapture 2>&1 | tail -20
echo ""

echo -e "${BLUE}[2/4]${NC} Running Candle reference implementation..."
cd .test_helpers/candle_ln_test
cargo run --release 2>&1 | grep -A 10 "CANDLE LAYERNORM"
cd ../..
echo ""

echo -e "${BLUE}[3/4]${NC} Running Mistral.rs reference implementation..."
cd .test_helpers/mistralrs_ln_test
cargo run --release 2>&1 | grep -A 10 "MISTRAL.RS LAYERNORM"
cd ../..
echo ""

echo -e "${BLUE}[4/4]${NC} Comparing outputs..."
python3 .test_helpers/compare_outputs.py
echo ""

echo -e "${GREEN}✅ Validation complete!${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} Mistral.rs uses Candle's LayerNorm, so both references are identical."
echo "See VALIDATION_SUMMARY.md for full details."
