# TEAM-132: queen-rbee Migration Plan

**Binary:** `bin/queen-rbee`  
**Date:** 2025-10-19  
**Estimated Duration:** 20 hours (2.5 days)

---

## Migration Strategy

### Overview

**Approach:** Phased extraction, low-risk first  
**Order:** Registry → Remote → HTTP Server → Orchestrator → Binary Cleanup  
**Verification:** Test after each phase, integration tests at end

### Phase Sequence Rationale

```
Phase 1: Registry (PILOT)
  ↓ (No dependencies, pure data layer)
Phase 2: Remote
  ↓ (Independent utilities)
Phase 3: HTTP Server
  ↓ (Depends on Registry only)
Phase 4: Orchestrator
  ↓ (Depends on Registry + Remote)
Phase 5: Binary Cleanup
  ↓ (Final integration)
```

---

## Phase 1: Extract Registry (PILOT)

**Crate:** `queen-rbee-registry`  
**Risk:** 🟢 LOW  
**Duration:** 2 hours  
**LOC:** 353

### Why Pilot?
- ✅ No external dependencies on other queen-rbee modules
- ✅ Pure data management (no business logic)
- ✅ Well-tested (2 comprehensive test suites)
- ✅ Small size (353 LOC)
- ✅ Clear, stable public API
- ✅ If this fails, low sunk cost

### Steps

#### 1.1 Create Crate Structure (10 min)
```bash
cd bin/
mkdir -p queen-rbee-crates/queen-rbee-registry/src
cd queen-rbee-crates/queen-rbee-registry
```

#### 1.2 Create Cargo.toml (10 min)
```toml
[package]
name = "queen-rbee-registry"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[dependencies]
rusqlite = { version = "0.32", features = ["bundled"] }
tokio = { version = "1", features = ["sync", "fs", "macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
tracing = "0.1"
reqwest = { version = "0.12", features = ["json"] }
dirs = "5.0"

[dev-dependencies]
tempfile = "3.0"
tokio = { version = "1", features = ["rt", "macros"] }
```

#### 1.3 Copy Registry Files (10 min)
```bash
# From bin/queen-rbee-crates/queen-rbee-registry/
cp ../../queen-rbee/src/beehive_registry.rs src/
cp ../../queen-rbee/src/worker_registry.rs src/
```

#### 1.4 Create lib.rs (15 min)
```rust
//! queen-rbee Registry Crate
//!
//! Dual registry system for managing beehive nodes and workers.

pub mod beehive_registry;
pub mod worker_registry;

// Re-export commonly used types
pub use beehive_registry::{BeehiveNode, BeehiveRegistry};
pub use worker_registry::{WorkerInfo, WorkerInfoExtended, WorkerRegistry, WorkerState};
```

#### 1.5 Run Tests (10 min)
```bash
cargo test -p queen-rbee-registry
```

**Expected Output:**
```
running 2 tests
test beehive_registry::tests::test_registry_crud ... ok
test worker_registry::tests::test_worker_registry_crud ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

#### 1.6 Update queen-rbee Dependencies (15 min)

**File:** `bin/queen-rbee/Cargo.toml`

Add dependency:
```toml
[dependencies]
queen-rbee-registry = { path = "../queen-rbee-crates/queen-rbee-registry" }
# ... keep all other dependencies
```

#### 1.7 Update Imports in queen-rbee (30 min)

**Files to update:**
- `src/main.rs`
- `src/lib.rs`
- `src/http/routes.rs`
- `src/http/workers.rs`
- `src/http/beehives.rs`
- `src/http/inference.rs`

**Change:**
```rust
// OLD
use crate::beehive_registry::{BeehiveNode, BeehiveRegistry};
use crate::worker_registry::{WorkerInfo, WorkerRegistry, WorkerState};

// NEW
use queen_rbee_registry::{BeehiveNode, BeehiveRegistry};
use queen_rbee_registry::{WorkerInfo, WorkerRegistry, WorkerState};
```

#### 1.8 Delete Original Files (5 min)
```bash
cd bin/queen-rbee/src
rm beehive_registry.rs worker_registry.rs
```

#### 1.9 Verify All Tests Pass (15 min)
```bash
cd bin/queen-rbee
cargo test
```

**Expected:** All tests pass ✅

#### 1.10 Verify Binary Builds (5 min)
```bash
cargo build --bin queen-rbee
```

**Expected:** Clean build ✅

### Rollback Plan
```bash
git checkout bin/queen-rbee/src/beehive_registry.rs
git checkout bin/queen-rbee/src/worker_registry.rs
git checkout bin/queen-rbee/Cargo.toml
# Revert imports
```

### Success Criteria
- [ ] Registry crate compiles independently
- [ ] Both registry tests pass in isolation
- [ ] queen-rbee binary builds successfully
- [ ] All queen-rbee tests pass
- [ ] No warnings about unused imports

---

## Phase 2: Extract Remote Utilities

**Crate:** `queen-rbee-remote`  
**Risk:** 🟡 MEDIUM (security fix required)  
**Duration:** 3 hours  
**LOC:** 182

### Steps

#### 2.1 Create Crate Structure (10 min)
```bash
cd bin/queen-rbee-crates
mkdir -p queen-rbee-remote/src/preflight
```

#### 2.2 Create Cargo.toml (10 min)
```toml
[package]
name = "queen-rbee-remote"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[dependencies]
tokio = { version = "1", features = ["process", "time", "macros"] }
reqwest = { version = "0.12", features = ["json"] }
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1", features = ["rt", "macros"] }
```

#### 2.3 Copy Files (15 min)
```bash
cp ../../queen-rbee/src/ssh.rs src/
cp ../../queen-rbee/src/preflight/*.rs src/preflight/
```

#### 2.4 🔴 FIX SECURITY VULNERABILITY (45 min)

**File:** `src/ssh.rs`

**Issue:** Command injection at line 79-81

**Current Code:**
```rust
.arg(format!("{}@{}", user, host))
.arg(command)  // ⚠️ UNSAFE: command is user-provided string
```

**Fixed Code:**
```rust
use shellwords; // Add to dependencies

// Sanitize command before execution
let sanitized_command = shellwords::split(command)
    .map_err(|e| anyhow::anyhow!("Invalid command syntax: {}", e))?;

// Validate no dangerous patterns
for part in &sanitized_command {
    if part.contains("&&") || part.contains("||") || part.contains(";") {
        anyhow::bail!("Command injection attempt detected");
    }
}

.arg(format!("{}@{}", user, host))
.arg("--")  // Force argument boundary
.args(&sanitized_command)  // Pass as separate arguments
```

**Add to Cargo.toml:**
```toml
shellwords = "1.1"
```

#### 2.5 Create lib.rs (15 min)
```rust
//! queen-rbee Remote Utilities
//!
//! SSH connection management and preflight validation for remote nodes.

pub mod preflight;
pub mod ssh;

// Re-export commonly used functions
pub use ssh::{execute_remote_command, test_ssh_connection};
pub use preflight::rbee_hive::RbeeHivePreflight;
pub use preflight::ssh::SshPreflight;
```

#### 2.6 Run Tests (10 min)
```bash
cargo test -p queen-rbee-remote
```

#### 2.7 Update queen-rbee Dependencies (10 min)

**File:** `bin/queen-rbee/Cargo.toml`

```toml
[dependencies]
queen-rbee-registry = { path = "../queen-rbee-crates/queen-rbee-registry" }
queen-rbee-remote = { path = "../queen-rbee-crates/queen-rbee-remote" }
```

#### 2.8 Update Imports (20 min)

**Files:**
- `src/main.rs`
- `src/http/inference.rs`
- `src/http/beehives.rs`

**Change:**
```rust
// OLD
use crate::ssh;
use crate::preflight;

// NEW
use queen_rbee_remote::{execute_remote_command, test_ssh_connection};
use queen_rbee_remote::{RbeeHivePreflight, SshPreflight};
```

#### 2.9 Delete Original Files (5 min)
```bash
rm src/ssh.rs
rm -rf src/preflight
```

#### 2.10 Verify (15 min)
```bash
cargo test --bin queen-rbee
cargo build --bin queen-rbee
```

### Security Verification

**Test Command Injection Fix:**
```rust
#[tokio::test]
async fn test_command_injection_blocked() {
    let result = execute_remote_command(
        "localhost",
        22,
        "user",
        None,
        "echo safe && rm -rf /"  // Malicious command
    ).await;
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("injection"));
}
```

### Success Criteria
- [ ] Security vulnerability fixed and tested
- [ ] Remote crate compiles independently
- [ ] All tests pass
- [ ] queen-rbee builds and runs
- [ ] No command injection possible

---

## Phase 3: Extract HTTP Server

**Crate:** `queen-rbee-http-server`  
**Risk:** 🟡 MEDIUM (large crate)  
**Duration:** 4 hours  
**LOC:** 897

### Steps

#### 3.1 Create Crate Structure (10 min)
```bash
mkdir -p queen-rbee-crates/queen-rbee-http-server/src/{middleware,handlers}
```

#### 3.2 Create Cargo.toml (15 min)
```toml
[package]
name = "queen-rbee-http-server"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[dependencies]
# HTTP framework
axum = "0.8"
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors"] }

# Registry
queen-rbee-registry = { path = "../queen-rbee-registry" }

# Shared crates
auth-min = { path = "../../../shared-crates/auth-min" }
input-validation = { path = "../../../shared-crates/input-validation" }
audit-logging = { path = "../../../shared-crates/audit-logging" }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# HTTP client (for worker shutdown)
reqwest = { version = "0.12", features = ["json"] }

# Other
tracing = "0.1"
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"

[dev-dependencies]
tokio = { version = "1", features = ["rt", "macros"] }
tower = { version = "0.5", features = ["util"] }
tempfile = "3.0"
```

#### 3.3 Copy HTTP Files (20 min)
```bash
# Copy all HTTP modules
cp -r ../../queen-rbee/src/http/* src/
```

#### 3.4 Restructure (30 min)

**New structure:**
```
src/
├── lib.rs
├── routes.rs (from http/routes.rs)
├── types.rs (from http/types.rs)
├── handlers/
│   ├── mod.rs
│   ├── health.rs (from http/health.rs)
│   ├── workers.rs (from http/workers.rs)
│   ├── beehives.rs (from http/beehives.rs)
└── middleware/
    ├── mod.rs
    └── auth.rs (from http/middleware/auth.rs)
```

#### 3.5 Create lib.rs (20 min)
```rust
//! queen-rbee HTTP Server
//!
//! HTTP routes, types, and middleware for the queen-rbee orchestrator.

pub mod handlers;
pub mod middleware;
pub mod routes;
pub mod types;

// Re-export router creation
pub use routes::{create_router, AppState};

// Re-export types
pub use types::*;
```

#### 3.6 Update Internal Imports (45 min)

Update all `use crate::http::*` to module paths:
```rust
// OLD
use crate::http::types::*;
use crate::http::routes::AppState;

// NEW
use crate::types::*;
use crate::routes::AppState;
```

#### 3.7 Remove Inference Endpoint (15 min)

**File:** `src/routes.rs`

**Remove these lines:**
```rust
.route("/v2/tasks", post(inference::handle_create_inference_task))
.route("/v1/inference", post(inference::handle_inference_request))
```

**Reason:** Inference belongs to orchestrator crate, will be added back later

#### 3.8 Run Tests (10 min)
```bash
cargo test -p queen-rbee-http-server
```

#### 3.9 Update queen-rbee Dependencies (10 min)
```toml
[dependencies]
queen-rbee-registry = { path = "../queen-rbee-crates/queen-rbee-registry" }
queen-rbee-remote = { path = "../queen-rbee-crates/queen-rbee-remote" }
queen-rbee-http-server = { path = "../queen-rbee-crates/queen-rbee-http-server" }
```

#### 3.10 Update queen-rbee Imports (30 min)

**File:** `src/main.rs`

```rust
// OLD
use crate::http::create_router;

// NEW
use queen_rbee_http_server::create_router;
```

#### 3.11 Delete Original HTTP Directory (5 min)
```bash
rm -rf src/http  # Except inference.rs - keep for Phase 4
mv src/http/inference.rs src/  # Move to root temporarily
rm -rf src/http
```

#### 3.12 Verify (20 min)
```bash
cargo build --bin queen-rbee
cargo test --bin queen-rbee
```

### Success Criteria
- [ ] HTTP server crate compiles independently
- [ ] All HTTP tests pass in isolation
- [ ] queen-rbee binary builds
- [ ] No broken imports
- [ ] Router creation works

---

## Phase 4: Extract Orchestrator

**Crate:** `queen-rbee-orchestrator`  
**Risk:** 🟡 MEDIUM (complex logic)  
**Duration:** 5 hours  
**LOC:** 610

### Steps

#### 4.1 Create Crate Structure (10 min)
```bash
mkdir -p queen-rbee-crates/queen-rbee-orchestrator/src
```

#### 4.2 Create Cargo.toml (15 min)
```toml
[package]
name = "queen-rbee-orchestrator"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[dependencies]
# HTTP framework
axum = "0.8"
tower = "0.5"

# Dependencies on other queen-rbee crates
queen-rbee-registry = { path = "../queen-rbee-registry" }
queen-rbee-remote = { path = "../queen-rbee-remote" }

# HTTP client
reqwest = { version = "0.12", features = ["json", "stream"] }
tokio = { version = "1", features = ["process", "time", "macros"] }
futures = "0.3"

# Shared crates
input-validation = { path = "../../../shared-crates/input-validation" }
deadline-propagation = { path = "../../../shared-crates/deadline-propagation" }

# Other
uuid = { version = "1.0", features = ["v4", "serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"

[dev-dependencies]
wiremock = "0.6"
tokio = { version = "1", features = ["rt", "macros"] }
```

#### 4.3 Copy inference.rs (10 min)
```bash
cp ../../queen-rbee/src/inference.rs src/orchestrator.rs
```

#### 4.4 Refactor into Modules (60 min)

**New structure:**
```
src/
├── lib.rs
├── orchestrator.rs (main logic - 466 LOC)
├── lifecycle.rs (worker lifecycle helpers - 100 LOC)
└── types.rs (request/response types - 44 LOC)
```

**Split large file:**
```rust
// lifecycle.rs - Extract helper functions
pub async fn ensure_local_rbee_hive_running() -> Result<String> { /* ... */ }
pub async fn establish_rbee_hive_connection(node: &BeehiveNode) -> Result<String> { /* ... */ }
pub async fn wait_for_rbee_hive_ready(url: &str) -> Result<()> { /* ... */ }
pub async fn wait_for_worker_ready(worker_url: &str) -> Result<()> { /* ... */ }

// types.rs - Move from http-server or redefine
pub(crate) struct WorkerSpawnResponse { /* ... */ }
pub(crate) struct ReadyResponse { /* ... */ }
```

#### 4.5 Create lib.rs (20 min)
```rust
//! queen-rbee Orchestrator
//!
//! Inference task orchestration and worker lifecycle management.

mod lifecycle;
mod orchestrator;
mod types;

// Re-export main handlers
pub use orchestrator::{
    handle_create_inference_task,
    handle_inference_request,
};

// Re-export lifecycle functions for testing
pub use lifecycle::*;
```

#### 4.6 Update Internal Imports (40 min)

Replace all `crate::*` references:
```rust
// OLD
use crate::beehive_registry::BeehiveNode;
use crate::worker_registry::WorkerRegistry;
use crate::http::types::*;
use crate::http::routes::AppState;
use crate::ssh;

// NEW
use queen_rbee_registry::{BeehiveNode, WorkerRegistry};
use queen_rbee_remote::{execute_remote_command, RbeeHivePreflight};
// Note: AppState will need to be passed or recreated
```

#### 4.7 Handle AppState Dependency (30 min)

**Problem:** Orchestrator needs `AppState` from http-server

**Solution:** Pass components individually or create orchestrator-specific state

```rust
// Option A: Pass registries directly
pub async fn handle_create_inference_task(
    beehive_registry: Arc<BeehiveRegistry>,
    worker_registry: Arc<WorkerRegistry>,
    req: InferenceTaskRequest,
) -> impl IntoResponse { /* ... */ }

// Option B: Create OrchestorState
pub struct OrchestratorState {
    pub beehive_registry: Arc<BeehiveRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
}

// HTTP server will convert AppState → OrchestratorState
```

#### 4.8 Integration with HTTP Server (45 min)

**File:** `queen-rbee-http-server/src/routes.rs`

```rust
use queen_rbee_orchestrator::{handle_create_inference_task, handle_inference_request};

// Add back inference routes
.route("/v2/tasks", post(handle_create_inference_task))
.route("/v1/inference", post(handle_inference_request))
```

**Update http-server Cargo.toml:**
```toml
[dependencies]
queen-rbee-orchestrator = { path = "../queen-rbee-orchestrator" }
```

#### 4.9 Run Tests (10 min)
```bash
cargo test -p queen-rbee-orchestrator
```

#### 4.10 Integration Tests (30 min)

Create integration test in orchestrator:

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_orchestration_flow() {
        // Mock registries
        // Mock HTTP server (wiremock)
        // Test full flow
    }
}
```

#### 4.11 Update queen-rbee Binary (15 min)

```toml
[dependencies]
queen-rbee-registry = { path = "../queen-rbee-crates/queen-rbee-registry" }
queen-rbee-remote = { path = "../queen-rbee-crates/queen-rbee-remote" }
queen-rbee-http-server = { path = "../queen-rbee-crates/queen-rbee-http-server" }
queen-rbee-orchestrator = { path = "../queen-rbee-crates/queen-rbee-orchestrator" }
```

#### 4.12 Delete inference.rs (5 min)
```bash
rm src/inference.rs
```

#### 4.13 Verify (20 min)
```bash
cargo build --bin queen-rbee
cargo test --workspace
```

### Success Criteria
- [ ] Orchestrator crate compiles independently
- [ ] Integration tests pass
- [ ] queen-rbee binary builds
- [ ] All inference functionality works
- [ ] No performance regression

---

## Phase 5: Binary Cleanup

**Target:** `bin/queen-rbee/src/main.rs`  
**Risk:** 🟢 LOW  
**Duration:** 1 hour  
**Final LOC:** ~283

### Steps

#### 5.1 Audit main.rs (15 min)

**Current contents:**
- CLI argument parsing (clap)
- Logging initialization
- Registry initialization
- Router creation
- HTTP server startup
- Shutdown handler

**All should remain** ✅

#### 5.2 Clean Up Imports (15 min)

**File:** `src/main.rs`

```rust
// Simplified imports
use queen_rbee_registry::{BeehiveRegistry, WorkerRegistry};
use queen_rbee_http_server::create_router;

// External dependencies
use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing::{error, info};
```

#### 5.3 Update lib.rs (10 min)

**File:** `src/lib.rs`

```rust
//! queen-rbee library
//!
//! Minimal re-exports for testing

pub use queen_rbee_registry as registry;
pub use queen_rbee_http_server as http;
pub use queen_rbee_orchestrator as orchestrator;
pub use queen_rbee_remote as remote;
```

#### 5.4 Final Cargo.toml (10 min)

**File:** `Cargo.toml`

```toml
[package]
name = "queen-rbee"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[lib]
name = "queen_rbee"
path = "src/lib.rs"

[[bin]]
name = "queen-rbee"
path = "src/main.rs"

[dependencies]
# Queen-rbee crates
queen-rbee-registry = { path = "../queen-rbee-crates/queen-rbee-registry" }
queen-rbee-http-server = { path = "../queen-rbee-crates/queen-rbee-http-server" }
queen-rbee-orchestrator = { path = "../queen-rbee-crates/queen-rbee-orchestrator" }
queen-rbee-remote = { path = "../queen-rbee-crates/queen-rbee-remote" }

# Shared crates
audit-logging = { path = "../shared-crates/audit-logging" }

# Runtime
tokio = { version = "1", features = ["rt-multi-thread", "macros", "signal"] }

# CLI
clap = { version = "4.5", features = ["derive"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }

# Other
anyhow = "1.0"
chrono = "0.4"

[dev-dependencies]
tempfile = "3.0"
```

#### 5.5 Verify Final Build (10 min)
```bash
cargo clean
cargo build --release --bin queen-rbee
```

#### 5.6 Run Full Test Suite (15 min)
```bash
cargo test --workspace
```

#### 5.7 Integration Test (20 min)

Start queen-rbee and verify:
```bash
./target/release/queen-rbee --port 8080 &
curl http://localhost:8080/health
# Should return: {"status":"ok","version":"0.1.0"}
```

### Success Criteria
- [ ] Binary is <300 LOC
- [ ] All dependencies are crates (no inline code)
- [ ] Clean build with no warnings
- [ ] All tests pass
- [ ] Health endpoint works
- [ ] Binary size reasonable

---

## Verification Checklist

### After Each Phase

- [ ] Crate compiles independently: `cargo build -p <crate>`
- [ ] Crate tests pass: `cargo test -p <crate>`
- [ ] queen-rbee binary builds: `cargo build --bin queen-rbee`
- [ ] queen-rbee tests pass: `cargo test --bin queen-rbee`
- [ ] No warnings in compilation output
- [ ] Git commit with clear message

### Final Verification

- [ ] All 4 crates compile independently
- [ ] All tests pass in workspace: `cargo test --workspace`
- [ ] Binary runs: `./target/release/queen-rbee --help`
- [ ] Health endpoint responds: `curl http://localhost:8080/health`
- [ ] Performance acceptable (no regression)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

## Timeline Summary

| Phase | Crate | Duration | Cumulative |
|-------|-------|----------|------------|
| 1 | queen-rbee-registry | 2h | 2h |
| 2 | queen-rbee-remote | 3h | 5h |
| 3 | queen-rbee-http-server | 4h | 9h |
| 4 | queen-rbee-orchestrator | 5h | 14h |
| 5 | Binary cleanup | 1h | 15h |
| **Subtotal** | | **15h** | |
| Buffer (33%) | | 5h | |
| **Total** | | **20h** | **2.5 days** |

---

## Rollback Procedures

### Immediate Rollback (During Phase)
```bash
git reset --hard HEAD
git clean -fd
```

### Partial Rollback (After Phase)
```bash
# Rollback Phase 4 (orchestrator)
git revert <phase4-commit>
rm -rf queen-rbee-crates/queen-rbee-orchestrator
# Restore inference.rs to main binary
```

### Full Rollback (Nuclear Option)
```bash
git checkout <pre-migration-tag>
cargo clean
cargo build --bin queen-rbee
```

---

## Post-Migration Tasks

### Documentation
- [ ] Update README.md with new structure
- [ ] Document each crate's purpose
- [ ] Update contribution guide
- [ ] Add architecture diagram

### CI/CD
- [ ] Update CI to test each crate separately
- [ ] Add crate-level test jobs
- [ ] Update release workflow
- [ ] Add performance benchmarks

### Monitoring
- [ ] Track build times before/after
- [ ] Monitor binary size
- [ ] Track test execution time
- [ ] Verify no performance regression

---

**Migration Plan Complete**  
**Ready to Execute**
