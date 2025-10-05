#!/usr/bin/env bash
set -euo pipefail

# Worker Crates Migration - Run All
# Executes all migration scripts in correct dependency order
#
# Usage:
#   ./migrate-all.sh           # Execute all migrations
#   ./migrate-all.sh --dry-run # Preview all changes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN_FLAG=""

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Worker Crates Migration - Full Suite${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

if [[ -n "$DRY_RUN_FLAG" ]]; then
    echo -e "${YELLOW}🔍 DRY RUN MODE - No changes will be made${NC}\n"
fi

# Migration order (respects dependencies)
migrations=(
    "migrate-worker-gguf.sh"
    # "migrate-worker-tokenizer.sh"    # TODO: Create this script
    # "migrate-worker-models.sh"       # TODO: Create this script (depends on worker-gguf)
    # "migrate-worker-common.sh"       # TODO: Create this script
    # "migrate-worker-http.sh"         # TODO: Create this script (depends on worker-common)
)

total=${#migrations[@]}
current=0

for script in "${migrations[@]}"; do
    current=$((current + 1))
    
    echo -e "\n${GREEN}[${current}/${total}]${NC} Running: ${BLUE}$script${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    if [[ -f "$SCRIPT_DIR/$script" ]]; then
        "$SCRIPT_DIR/$script" $DRY_RUN_FLAG
    else
        echo -e "${YELLOW}⚠ Script not found: $script (skipping)${NC}"
    fi
    
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
done

echo -e "\n${GREEN}✅ All migrations complete!${NC}\n"

if [[ -n "$DRY_RUN_FLAG" ]]; then
    echo -e "To execute all migrations, run:"
    echo -e "  ${BLUE}$0${NC}\n"
else
    echo -e "Next steps:"
    echo -e "  1. Review all changes: ${BLUE}git log --oneline -10${NC}"
    echo -e "  2. Run full test suite: ${BLUE}cargo test --workspace${NC}"
    echo -e "  3. Update worker-orcd to use ComputeBackend trait"
    echo -e "  4. Begin worker-aarmd implementation\n"
fi
