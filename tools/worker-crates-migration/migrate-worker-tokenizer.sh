#!/usr/bin/env bash
set -euo pipefail

# Worker Tokenizer Migration Script
# Extracts tokenizer from worker-orcd to worker-crates/worker-tokenizer
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

log_warning() {
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
            log_warning "Backup branch 'migration-backup' already exists"
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
            log_warning "worker-gguf dependency already exists in Cargo.toml"
        fi
    fi
    
    # Step 4: Move directory with git mv
    log_step "Step 4: Move directory with git mv"
    
    # Remove placeholder files first
    if [[ "$DRY_RUN" == false ]]; then
        rm -rf bin/worker-crates/worker-tokenizer/src/*

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
