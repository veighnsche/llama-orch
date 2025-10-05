#!/usr/bin/env bash
set -euo pipefail

# Worker Models Migration Script
# Extracts model adapters from worker-orcd to worker-crates/worker-models
#
# Usage:
#   ./migrate-worker-gguf.sh           # Execute migration
#   ./migrate-worker-gguf.sh --dry-run # Preview changes only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DRY_RUN=false

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_wmodel_adapter() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
{{ ... }}
    log_step "Step 1: Create backup branch"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would create branch: migration-backup"
    else
        if git rev-parse --verify migration-backup >/dev/null 2>&1; then
            log_wmodel_adapter "Backup branch 'migration-backup' already exists"
        else
            git branch migration-backup
            log_success "Created backup branch: migration-backup"
        fi
    fi
{{ ... }}
        if ! grep -q "worker-gguf" "$cargo_toml"; then
            # Add after [dependencies] section
            sed -i '/^\[dependencies\]/a worker-gguf = { path = "../worker-crates/worker-gguf" }' "$cargo_toml"
        else
            log_wmodel_adapter "worker-gguf dependency already exists in Cargo.toml"
        fi
    fi
    
    # Step 8: Add unit tests for worker-gguf model adapter
    log_step "Step 8: Add unit tests for worker-gguf model adapter"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would add unit tests for worker-gguf model adapter"
    else
        local history_count=$(git log --follow --oneline bin/worker-crates/worker-models/src | wc -l)
        if [[ $history_count -gt 0 ]]; then
            log_success "Git history preserved: $history_count commits found"
            log_wmodel_adapter "No git history found (file might be new)"
        fi
    fi
    
    # Step 10: Commit changes
    log_step "Step 10: Commit changes"
{{ ... }}
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would commit with message:"
        echo -e "  ${YELLOW}refactor: extract worker-gguf from worker-orcd${NC}"
        echo -e "  ${YELLOW}${NC}"
        echo -e "  ${YELLOW}- Move src/gguf/mod.rs to worker-crates/worker-gguf${NC}"
        echo -e "  ${YELLOW}- Update imports in worker-orcd${NC}"
        echo -e "  ${YELLOW}- Enables code reuse for worker-aarmd${NC}"
        echo -e "  ${YELLOW}${NC}"
        echo -e "  ${YELLOW}Refs: .docs/WORKER_AARMD_DEVELOPMENT_PLAN.md Phase 1.1${NC}"
    else
        git add -A
        git commit -m "refactor: extract worker-gguf from worker-orcd

- Move src/gguf/mod.rs to worker-crates/worker-gguf
- Update imports in worker-orcd
- Enables code reuse for worker-aarmd

Refs: .docs/WORKER_AARMD_DEVELOPMENT_PLAN.md Phase 1.1"
        log_success "Changes committed"
    fi
    
    # Summary
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${GREEN}✅ Dry run complete${NC}"
        echo -e "\nTo execute migration, run:"
        echo -e "  ${BLUE}$0${NC}"
    else
        echo -e "${GREEN}✅ Migration complete!${NC}"
        echo -e "\nNext steps:"
        echo -e "  1. Review changes: ${BLUE}git show${NC}"
        echo -e "  2. Run full test suite: ${BLUE}cargo test${NC}"
        echo -e "  3. Continue with next migration: ${BLUE}./migrate-worker-tokenizer.sh${NC}"
        echo -e "\nRollback if needed:"
        echo -e "  ${YELLOW}git reset --hard migration-backup${NC}"
    fi
}

# Run migration
migrate_worker_gguf
