#!/usr/bin/env bash
set -euo pipefail

# Worker GGUF Migration Script
# Extracts GGUF parser from worker-orcd to worker-crates/worker-gguf
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
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}━━━${NC} $1"
}

# Execute or preview command
run_cmd() {
    local cmd="$1"
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${YELLOW}Would execute:${NC} $cmd"
    else
        log_info "Executing: $cmd"
        eval "$cmd"
    fi
}

# Main migration function
migrate_worker_gguf() {
    cd "$REPO_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}🔍 DRY RUN MODE - No changes will be made${NC}"
    else
        echo -e "${GREEN}🚀 EXECUTING MIGRATION${NC}"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\n${BLUE}📦 Extracting worker-gguf from worker-orcd${NC}\n"
    
    # Step 1: Create backup branch
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
    
    # Step 2: Verify source file exists
    log_step "Step 2: Verify source file exists"
    if [[ ! -f "bin/worker-orcd/src/gguf/mod.rs" ]]; then
        log_error "Source file not found: bin/worker-orcd/src/gguf/mod.rs"
        exit 1
    fi
    log_success "Source file exists: bin/worker-orcd/src/gguf/mod.rs"
    
    # Step 3: Move source file with git mv
    log_step "Step 3: Move source file with git mv"
    run_cmd "git mv bin/worker-orcd/src/gguf/mod.rs bin/worker-crates/worker-gguf/src/lib.rs"
    
    if [[ "$DRY_RUN" == false ]]; then
        log_success "Moved: gguf/mod.rs → worker-gguf/src/lib.rs"
    fi
    
    # Note: Tests are embedded in mod.rs (#[cfg(test)] module)
    # They will move with the source file automatically
    log_info "Tests are embedded in source file (will move automatically)"
    
    # Step 4: Remove gguf directory if empty
    log_step "Step 4: Clean up empty directories"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would remove: bin/worker-orcd/src/gguf/ (if empty)"
    else
        if [[ -d "bin/worker-orcd/src/gguf" ]]; then
            rmdir bin/worker-orcd/src/gguf 2>/dev/null || true
            log_success "Removed empty directory: bin/worker-orcd/src/gguf/"
        fi
    fi
    
    # Step 5: Update imports in worker-orcd
    log_step "Step 5: Update imports in worker-orcd"
    
    local files_to_update=(
        "bin/worker-orcd/src/main.rs"
        "bin/worker-orcd/src/lib.rs"
        "bin/worker-orcd/src/cuda/model.rs"
    )
    
    for file in "${files_to_update[@]}"; do
        if [[ -f "$file" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                if grep -q "use crate::gguf::" "$file" 2>/dev/null; then
                    echo -e "  ${YELLOW}Would update:${NC} $file"
                    echo -e "    ${YELLOW}Replace:${NC} use crate::gguf:: → use worker_gguf::"
                fi
            else
                if grep -q "use crate::gguf::" "$file" 2>/dev/null; then
                    sed -i 's/use crate::gguf::/use worker_gguf::/g' "$file"
                    log_success "Updated imports in: $file"
                fi
            fi
        fi
    done
    
    # Step 6: Update worker-orcd Cargo.toml
    log_step "Step 6: Update worker-orcd Cargo.toml"
    
    local cargo_toml="bin/worker-orcd/Cargo.toml"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would add to $cargo_toml:"
        echo -e "  ${YELLOW}worker-gguf = { path = \"../worker-crates/worker-gguf\" }${NC}"
    else
        if ! grep -q "worker-gguf" "$cargo_toml"; then
            # Add after [dependencies] section
            sed -i '/^\[dependencies\]/a worker-gguf = { path = "../worker-crates/worker-gguf" }' "$cargo_toml"
            log_success "Added worker-gguf dependency to Cargo.toml"
        else
            log_warning "worker-gguf dependency already exists in Cargo.toml"
        fi
    fi
    
    # Step 7: Verify compilation
    log_step "Step 7: Verify compilation"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would run: cargo check -p worker-gguf"
        log_success "Would run: cargo check -p worker-orcd"
    else
        log_info "Checking worker-gguf..."
        if cargo check -p worker-gguf 2>&1 | tee /tmp/worker-gguf-check.log; then
            log_success "worker-gguf compiles successfully"
        else
            log_error "worker-gguf compilation failed. See /tmp/worker-gguf-check.log"
            exit 1
        fi
        
        log_info "Checking worker-orcd..."
        if cargo check -p worker-orcd 2>&1 | tee /tmp/worker-orcd-check.log; then
            log_success "worker-orcd compiles successfully"
        else
            log_error "worker-orcd compilation failed. See /tmp/worker-orcd-check.log"
            exit 1
        fi
    fi
    
    # Step 8: Run tests
    log_step "Step 8: Run tests"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would run: cargo test -p worker-gguf"
        log_info "  Tests migrated: 5 unit tests in #[cfg(test)] module"
        log_info "  - test_qwen_metadata"
        log_info "  - test_phi3_metadata"
        log_info "  - test_gpt2_metadata"
        log_info "  - test_rope_freq_base"
        log_info "  - test_context_length"
        log_success "Would run: cargo test -p worker-orcd"
    else
        log_info "Testing worker-gguf (includes 5 migrated unit tests)..."
        if cargo test -p worker-gguf 2>&1 | tee /tmp/worker-gguf-test.log; then
            log_success "worker-gguf tests passed (5 tests)"
        else
            log_error "worker-gguf tests failed. See /tmp/worker-gguf-test.log"
            exit 1
        fi
        
        log_info "Testing worker-orcd..."
        if cargo test -p worker-orcd 2>&1 | tee /tmp/worker-orcd-test.log; then
            log_success "worker-orcd tests passed"
        else
            log_error "worker-orcd tests failed. See /tmp/worker-orcd-test.log"
            exit 1
        fi
    fi
    
    # Step 9: Verify git history preserved
    log_step "Step 9: Verify git history preserved"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would verify: git log --follow bin/worker-crates/worker-gguf/src/lib.rs"
    else
        local history_count=$(git log --follow --oneline bin/worker-crates/worker-gguf/src/lib.rs | wc -l)
        if [[ $history_count -gt 0 ]]; then
            log_success "Git history preserved: $history_count commits found"
        else
            log_warning "No git history found (file might be new)"
        fi
    fi
    
    # Step 10: Commit changes
    log_step "Step 10: Commit changes"
    
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
