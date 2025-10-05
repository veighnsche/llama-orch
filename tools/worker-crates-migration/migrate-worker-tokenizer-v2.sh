#!/usr/bin/env bash
set -euo pipefail

# Worker Tokenizer Migration Script
# Extracts tokenizer from worker-orcd to worker-crates/worker-tokenizer
#
# Usage:
#   ./migrate-worker-tokenizer-v2.sh           # Execute migration
#   ./migrate-worker-tokenizer-v2.sh --dry-run # Preview changes only

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
migrate_worker_tokenizer() {
    cd "$REPO_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}🔍 DRY RUN MODE - No changes will be made${NC}"
    else
        echo -e "${GREEN}🚀 EXECUTING MIGRATION${NC}"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\n${BLUE}📦 Extracting worker-tokenizer from worker-orcd${NC}\n"
    
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
    
    # Step 2: Verify source directory exists
    log_step "Step 2: Verify source directory exists"
    if [[ ! -d "bin/worker-orcd/src/tokenizer" ]]; then
        log_error "Source directory not found: bin/worker-orcd/src/tokenizer"
        exit 1
    fi
    log_success "Source directory exists: bin/worker-orcd/src/tokenizer"
    
    # Step 3: Remove placeholder files
    log_step "Step 3: Remove placeholder files from worker-tokenizer"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would remove: bin/worker-crates/worker-tokenizer/src/lib.rs"
    else
        rm -f bin/worker-crates/worker-tokenizer/src/lib.rs
        log_success "Removed placeholder file"
    fi
    
    # Step 4: Move tokenizer directory with git mv
    log_step "Step 4: Move tokenizer directory with git mv"
    
    local tokenizer_files=(
        "backend.rs"
        "decoder.rs"
        "discovery.rs"
        "encoder.rs"
        "error.rs"
        "hf_json.rs"
        "merges.rs"
        "metadata.rs"
        "mod.rs"
        "streaming.rs"
        "vocab.rs"
    )
    
    for file in "${tokenizer_files[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "  ${YELLOW}Would move:${NC} bin/worker-orcd/src/tokenizer/$file → bin/worker-crates/worker-tokenizer/src/$file"
        else
            if [[ -f "bin/worker-orcd/src/tokenizer/$file" ]]; then
                git mv "bin/worker-orcd/src/tokenizer/$file" "bin/worker-crates/worker-tokenizer/src/$file"
                log_success "Moved: $file"
            fi
        fi
    done
    
    if [[ "$DRY_RUN" == false ]]; then
        # Rename mod.rs to lib.rs
        if [[ -f "bin/worker-crates/worker-tokenizer/src/mod.rs" ]]; then
            git mv bin/worker-crates/worker-tokenizer/src/mod.rs bin/worker-crates/worker-tokenizer/src/lib.rs
            log_success "Renamed mod.rs → lib.rs"
        fi
    fi
    
    # Step 5: Clean up empty directories
    log_step "Step 5: Clean up empty directories"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would remove: bin/worker-orcd/src/tokenizer/ (if empty)"
    else
        if [[ -d "bin/worker-orcd/src/tokenizer" ]]; then
            rmdir bin/worker-orcd/src/tokenizer 2>/dev/null || true
            log_success "Removed empty directory: bin/worker-orcd/src/tokenizer/"
        fi
    fi
    
    # Step 6: Update imports in worker-orcd
    log_step "Step 6: Update imports in worker-orcd"
    
    local files_to_update=(
        "bin/worker-orcd/src/main.rs"
        "bin/worker-orcd/src/lib.rs"
        "bin/worker-orcd/src/inference_executor.rs"
    )
    
    for file in "${files_to_update[@]}"; do
        if [[ -f "$file" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                if grep -q "use crate::tokenizer::" "$file" 2>/dev/null || grep -q "mod tokenizer" "$file" 2>/dev/null; then
                    echo -e "  ${YELLOW}Would update:${NC} $file"
                    echo -e "    ${YELLOW}Replace:${NC} use crate::tokenizer:: → use worker_tokenizer::"
                    echo -e "    ${YELLOW}Remove:${NC} mod tokenizer;"
                fi
            else
                if grep -q "use crate::tokenizer::" "$file" 2>/dev/null; then
                    sed -i 's/use crate::tokenizer::/use worker_tokenizer::/g' "$file"
                    log_success "Updated imports in: $file"
                fi
                if grep -q "^mod tokenizer;" "$file" 2>/dev/null; then
                    sed -i '/^mod tokenizer;/d' "$file"
                    log_success "Removed mod declaration in: $file"
                fi
            fi
        fi
    done
    
    # Step 7: Update worker-orcd Cargo.toml
    log_step "Step 7: Update worker-orcd Cargo.toml"
    
    local cargo_toml="bin/worker-orcd/Cargo.toml"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would add to $cargo_toml:"
        echo -e "  ${YELLOW}worker-tokenizer = { path = \"../worker-crates/worker-tokenizer\" }${NC}"
    else
        if ! grep -q "worker-tokenizer" "$cargo_toml"; then
            # Add after worker-gguf dependency
            sed -i '/^worker-gguf = /a worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }' "$cargo_toml"
            log_success "Added worker-tokenizer dependency to Cargo.toml"
        else
            log_warning "worker-tokenizer dependency already exists in Cargo.toml"
        fi
    fi
    
    # Step 8: Move pure Rust integration tests
    log_step "Step 8: Move pure Rust integration tests"
    
    local test_files=(
        "tokenizer_conformance_qwen.rs"
        "phi3_tokenizer_conformance.rs"
        "utf8_edge_cases.rs"
    )
    
    for test_file in "${test_files[@]}"; do
        if [[ -f "bin/worker-orcd/tests/$test_file" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo -e "  ${YELLOW}Would move:${NC} bin/worker-orcd/tests/$test_file → bin/worker-crates/worker-tokenizer/tests/$test_file"
            else
                mkdir -p bin/worker-crates/worker-tokenizer/tests
                git mv "bin/worker-orcd/tests/$test_file" "bin/worker-crates/worker-tokenizer/tests/$test_file"
                log_success "Moved test: $test_file"
            fi
        fi
    done
    
    # Step 9: Update test imports
    log_step "Step 9: Update test imports in moved tests"
    if [[ "$DRY_RUN" == false ]]; then
        for test_file in "${test_files[@]}"; do
            local test_path="bin/worker-crates/worker-tokenizer/tests/$test_file"
            if [[ -f "$test_path" ]]; then
                sed -i 's/use worker_orcd::tokenizer::/use worker_tokenizer::/g' "$test_path"
                log_success "Updated imports in: $test_file"
            fi
        done
    fi
    
    # Step 10: Remove test declarations from worker-orcd Cargo.toml
    log_step "Step 10: Remove moved test declarations from worker-orcd Cargo.toml"
    if [[ "$DRY_RUN" == false ]]; then
        for test_file in "${test_files[@]}"; do
            local test_name="${test_file%.rs}"
            sed -i "/\[\[test\]\]/,/path = \"tests\/$test_file\"/d" "$cargo_toml"
        done
        log_success "Removed test declarations from worker-orcd Cargo.toml"
    fi
    
    # Step 11: Verify compilation
    log_step "Step 11: Verify compilation"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would run: cargo check -p worker-tokenizer"
        log_success "Would run: cargo check -p worker-orcd"
    else
        log_info "Checking worker-tokenizer..."
        if cargo check -p worker-tokenizer 2>&1 | tee /tmp/worker-tokenizer-check.log; then
            log_success "worker-tokenizer compiles successfully"
        else
            log_error "worker-tokenizer compilation failed. See /tmp/worker-tokenizer-check.log"
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
    
    # Step 12: Run tests
    log_step "Step 12: Run tests"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would run: cargo test -p worker-tokenizer"
        log_success "Would run: cargo test -p worker-orcd"
    else
        log_info "Testing worker-tokenizer..."
        if cargo test -p worker-tokenizer 2>&1 | tee /tmp/worker-tokenizer-test.log; then
            log_success "worker-tokenizer tests passed"
        else
            log_error "worker-tokenizer tests failed. See /tmp/worker-tokenizer-test.log"
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
    
    # Step 13: Verify git history preserved
    log_step "Step 13: Verify git history preserved"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would verify: git log --follow bin/worker-crates/worker-tokenizer/src/lib.rs"
    else
        local history_count=$(git log --follow --oneline bin/worker-crates/worker-tokenizer/src/lib.rs | wc -l)
        if [[ $history_count -gt 0 ]]; then
            log_success "Git history preserved: $history_count commits found"
        else
            log_warning "No git history found (file might be new)"
        fi
    fi
    
    # Step 14: Commit changes
    log_step "Step 14: Commit changes"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Would commit with message:"
        echo -e "  ${YELLOW}refactor: extract worker-tokenizer from worker-orcd${NC}"
        echo -e "  ${YELLOW}${NC}"
        echo -e "  ${YELLOW}- Move src/tokenizer/ to worker-crates/worker-tokenizer${NC}"
        echo -e "  ${YELLOW}- Move 3 pure Rust integration tests${NC}"
        echo -e "  ${YELLOW}- Update imports in worker-orcd${NC}"
        echo -e "  ${YELLOW}- Enables code reuse for worker-aarmd${NC}"
        echo -e "  ${YELLOW}${NC}"
        echo -e "  ${YELLOW}Refs: .docs/WORKER_AARMD_DEVELOPMENT_PLAN.md Phase 1.2${NC}"
    else
        git add -A
        git commit -m "refactor: extract worker-tokenizer from worker-orcd

- Move src/tokenizer/ to worker-crates/worker-tokenizer
- Move 3 pure Rust integration tests
- Update imports in worker-orcd
- Enables code reuse for worker-aarmd

Refs: .docs/WORKER_AARMD_DEVELOPMENT_PLAN.md Phase 1.2"
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
        echo -e "  3. Continue with next migration: ${BLUE}./migrate-worker-models-v2.sh${NC}"
        echo -e "\nRollback if needed:"
        echo -e "  ${YELLOW}git reset --hard migration-backup${NC}"
    fi
}

# Run migration
migrate_worker_tokenizer
