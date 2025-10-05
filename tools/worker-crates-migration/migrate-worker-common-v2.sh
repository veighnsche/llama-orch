#!/usr/bin/env bash
set -euo pipefail

# Worker Common Migration Script
# Extracts common types from worker-orcd to worker-crates/worker-common

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

migrate_worker_common() {
    cd "$REPO_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}🔍 DRY RUN MODE${NC}"
    else
        echo -e "${GREEN}🚀 EXECUTING MIGRATION${NC}"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\n${BLUE}📦 Extracting worker-common from worker-orcd${NC}\n"
    
    log_step "Step 1: Verify source files"
    local common_files=("error.rs" "sampling_config.rs" "inference_result.rs" "startup.rs")
    for file in "${common_files[@]}"; do
        if [[ ! -f "bin/worker-orcd/src/$file" ]]; then
            log_error "Source not found: bin/worker-orcd/src/$file"
            exit 1
        fi
    done
    log_success "All source files exist"
    
    log_step "Step 2: Remove placeholder"
    if [[ "$DRY_RUN" == false ]]; then
        rm -f bin/worker-crates/worker-common/src/lib.rs
        log_success "Removed placeholder"
    fi
    
    log_step "Step 3: Move common files"
    for file in "${common_files[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "  ${YELLOW}Would move:${NC} $file"
        else
            git mv "bin/worker-orcd/src/$file" "bin/worker-crates/worker-common/src/$file"
            log_success "Moved: $file"
        fi
    done
    
    log_step "Step 4: Create lib.rs"
    if [[ "$DRY_RUN" == false ]]; then
        cat > bin/worker-crates/worker-common/src/lib.rs << 'EOF'
//! Worker Common Types
//!
//! Shared types and utilities for llama-orch workers.

pub mod error;
pub mod inference_result;
pub mod sampling_config;
pub mod startup;

pub use error::WorkerError;
pub use inference_result::InferenceResult;
pub use sampling_config::SamplingConfig;
pub use startup::send_ready_callback;
EOF
        log_success "Created lib.rs"
    fi
    
    log_step "Step 5: Update worker-common Cargo.toml"
    if [[ "$DRY_RUN" == false ]]; then
        cat > bin/worker-crates/worker-common/Cargo.toml << 'EOF'
[package]
name = "worker-common"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"
description = "Common types and utilities for llama-orch workers"

[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1.0", features = ["full"] }
tracing = "0.1"

[dev-dependencies]
EOF
        log_success "Updated Cargo.toml"
    fi
    
    log_step "Step 6: Fix imports in worker-common"
    if [[ "$DRY_RUN" == false ]]; then
        # No crate:: references to fix since these are top-level modules
        log_success "Imports OK (no changes needed)"
    fi
    
    log_step "Step 7: Update worker-orcd"
    if [[ "$DRY_RUN" == false ]]; then
        # Remove module declarations
        sed -i '/^pub mod error;/d' bin/worker-orcd/src/lib.rs
        sed -i '/^pub mod inference_result;/d' bin/worker-orcd/src/lib.rs
        sed -i '/^pub mod sampling_config;/d' bin/worker-orcd/src/lib.rs
        sed -i '/^pub mod startup;/d' bin/worker-orcd/src/lib.rs
        
        # Remove re-exports
        sed -i '/^pub use error::WorkerError;/d' bin/worker-orcd/src/lib.rs
        sed -i '/^pub use inference_result::InferenceResult;/d' bin/worker-orcd/src/lib.rs
        sed -i '/^pub use sampling_config::SamplingConfig;/d' bin/worker-orcd/src/lib.rs
        
        # Add worker-common dependency
        sed -i '/^worker-models = /a worker-common = { path = "../worker-crates/worker-common" }' bin/worker-orcd/Cargo.toml
        
        log_success "Updated worker-orcd"
    fi
    
    log_step "Step 8: Update imports in worker-orcd"
    if [[ "$DRY_RUN" == false ]]; then
        # Update imports throughout worker-orcd
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::error::/use worker_common::error::/g' {} \;
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::inference_result::/use worker_common::inference_result::/g' {} \;
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::sampling_config::/use worker_common::sampling_config::/g' {} \;
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::startup::/use worker_common::startup::/g' {} \;
        
        # Update short-form imports
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::{error::WorkerError/use worker_common::{error::WorkerError/g' {} \;
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::error::WorkerError/use worker_common::WorkerError/g' {} \;
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::InferenceResult/use worker_common::InferenceResult/g' {} \;
        find bin/worker-orcd/src -name '*.rs' -exec sed -i 's/use crate::SamplingConfig/use worker_common::SamplingConfig/g' {} \;
        
        log_success "Updated imports"
    fi
    
    log_step "Step 9: Verify compilation"
    if [[ "$DRY_RUN" == false ]]; then
        cargo check -p worker-common && log_success "worker-common compiles"
        cargo check -p worker-orcd && log_success "worker-orcd compiles"
    fi
    
    log_step "Step 10: Run tests"
    if [[ "$DRY_RUN" == false ]]; then
        cargo test -p worker-common --lib && log_success "worker-common tests pass"
    fi
    
    log_step "Step 11: Commit"
    if [[ "$DRY_RUN" == false ]]; then
        git add -A
        git commit -m "refactor: extract worker-common from worker-orcd

- Move error.rs, sampling_config.rs, inference_result.rs, startup.rs
- Update imports throughout worker-orcd
- Add dependencies: thiserror, serde, reqwest, tokio, tracing

Refs: .docs/WORKER_AARMD_DEVELOPMENT_PLAN.md Phase 1.4"
        log_success "Committed"
    fi
    
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}✅ Migration complete!${NC}"
}

migrate_worker_common
