#!/usr/bin/env bash
set -euo pipefail

# Worker Models Migration Script
# Extracts model adapters from worker-orcd to worker-crates/worker-models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_step() { echo -e "\n${BLUE}━━━${NC} $1"; }

migrate_worker_models() {
    cd "$REPO_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}🔍 DRY RUN MODE${NC}"
    else
        echo -e "${GREEN}🚀 EXECUTING MIGRATION${NC}"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\n${BLUE}📦 Extracting worker-models from worker-orcd${NC}\n"
    
    log_step "Step 1: Verify source directory"
    if [[ ! -d "bin/worker-orcd/src/models" ]]; then
        log_error "Source not found: bin/worker-orcd/src/models"
        exit 1
    fi
    log_success "Source exists: bin/worker-orcd/src/models"
    
    log_step "Step 2: Remove placeholder"
    if [[ "$DRY_RUN" == false ]]; then
        rm -f bin/worker-crates/worker-models/src/lib.rs
        log_success "Removed placeholder"
    fi
    
    log_step "Step 3: Move models directory"
    local model_files=("adapter.rs" "factory.rs" "gpt.rs" "mod.rs" "phi3.rs" "qwen.rs")
    
    for file in "${model_files[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "  ${YELLOW}Would move:${NC} $file"
        else
            if [[ -f "bin/worker-orcd/src/models/$file" ]]; then
                git mv "bin/worker-orcd/src/models/$file" "bin/worker-crates/worker-models/src/$file"
                log_success "Moved: $file"
            fi
        fi
    done
    
    if [[ "$DRY_RUN" == false ]]; then
        if [[ -f "bin/worker-crates/worker-models/src/mod.rs" ]]; then
            git mv bin/worker-crates/worker-models/src/mod.rs bin/worker-crates/worker-models/src/lib.rs
            log_success "Renamed mod.rs → lib.rs"
        fi
        if [[ -d "bin/worker-orcd/src/models" ]]; then
            rmdir bin/worker-orcd/src/models 2>/dev/null || true
        fi
    fi
    
    log_step "Step 4: Fix imports in worker-models"
    if [[ "$DRY_RUN" == false ]]; then
        find bin/worker-crates/worker-models/src -name '*.rs' -exec sed -i 's/use crate::models::/use crate::/g' {} \;
        find bin/worker-crates/worker-models/src -name '*.rs' -exec sed -i 's/use crate::gguf::/use worker_gguf::/g' {} \;
        log_success "Fixed imports"
    fi
    
    log_step "Step 5: Update worker-models Cargo.toml"
    if [[ "$DRY_RUN" == false ]]; then
        cat >> bin/worker-crates/worker-models/Cargo.toml << 'EOF'

[dependencies]
worker-gguf = { path = "../worker-gguf" }
thiserror = "1.0"
tracing = "0.1"

[dev-dependencies]
EOF
        log_success "Updated Cargo.toml"
    fi
    
    log_step "Step 6: Update worker-orcd"
    if [[ "$DRY_RUN" == false ]]; then
        sed -i '/^pub mod models;/d' bin/worker-orcd/src/lib.rs
        sed -i '/^worker-tokenizer = /a worker-models = { path = "../worker-crates/worker-models" }' bin/worker-orcd/Cargo.toml
        log_success "Updated worker-orcd"
    fi
    
    log_step "Step 7: Move integration tests"
    local tests=("adapter_integration.rs" "adapter_factory_integration.rs")
    for test in "${tests[@]}"; do
        if [[ -f "bin/worker-orcd/tests/$test" ]] && [[ "$DRY_RUN" == false ]]; then
            mkdir -p bin/worker-crates/worker-models/tests
            git mv "bin/worker-orcd/tests/$test" "bin/worker-crates/worker-models/tests/$test"
            sed -i 's/use worker_orcd::models::/use worker_models::/g' "bin/worker-crates/worker-models/tests/$test"
            sed -i 's/use worker_orcd::gguf::/use worker_gguf::/g' "bin/worker-crates/worker-models/tests/$test"
            log_success "Moved test: $test"
        fi
    done
    
    log_step "Step 8: Remove test declarations"
    if [[ "$DRY_RUN" == false ]]; then
        sed -i '/\[\[test\]\]/,/path = "tests\/adapter_integration.rs"/d' bin/worker-orcd/Cargo.toml
        sed -i '/\[\[test\]\]/,/path = "tests\/adapter_factory_integration.rs"/d' bin/worker-orcd/Cargo.toml
    fi
    
    log_step "Step 9: Verify compilation"
    if [[ "$DRY_RUN" == false ]]; then
        cargo check -p worker-models && log_success "worker-models compiles"
        cargo check -p worker-orcd && log_success "worker-orcd compiles"
    fi
    
    log_step "Step 10: Run tests"
    if [[ "$DRY_RUN" == false ]]; then
        cargo test -p worker-models --lib && log_success "worker-models tests pass"
    fi
    
    log_step "Step 11: Commit"
    if [[ "$DRY_RUN" == false ]]; then
        git add -A
        git commit -m "refactor: extract worker-models from worker-orcd

- Move src/models/ to worker-crates/worker-models
- Move 2 pure Rust integration tests
- Update imports to use worker-gguf
- Add dependencies: worker-gguf, thiserror, tracing

Refs: .docs/WORKER_AARMD_DEVELOPMENT_PLAN.md Phase 1.3"
        log_success "Committed"
    fi
    
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}✅ Migration complete!${NC}"
}

migrate_worker_models
