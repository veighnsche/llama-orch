# TEAM-262: Post-TEAM-261 Cleanup & Architecture Consolidation

**Date:** Oct 23, 2025  
**Status:** 🚧 READY TO START  
**Priority:** HIGH  
**Estimated Effort:** 2-3 days

---

## Mission

Clean up deprecated code exposed by TEAM-261's heartbeat simplification and implement missing queen lifecycle management features.

**Context:** TEAM-261 removed hive heartbeats (workers now send directly to queen), which made several crates and components obsolete. This cleanup will remove ~910 LOC of dead code and add ~200 LOC for queen lifecycle management.

---

## Prerequisites

**Read First:**
1. `.arch/CLEANUP_PLAN_TEAM261.md` - Comprehensive cleanup plan
2. `.arch/SSE_ROUTING_ANALYSIS.md` - SSE routing verification
3. `bin/.plan/TEAM_261_IMPLEMENTATION_COMPLETE.md` - What TEAM-261 changed

**Understanding Required:**
- TEAM-261 heartbeat simplification (workers → queen direct)
- Why hive heartbeat aggregation was removed
- SSE job isolation security model
- Queen build configurations (distributed vs integrated)

---

## Phase 1: Delete Dead Code (HIGH Priority)

### 1.1 Delete Obsolete Hive Worker Registry

**Location:** `bin/25_rbee_hive_crates/worker-registry`

**Rationale:** After TEAM-261, workers send heartbeats to QUEEN (not hive). Queen tracks all workers. Hive doesn't need worker tracking.

**Steps:**
```bash
# 1. Verify no usage
grep -r "rbee-hive-worker-registry" --include="*.toml"
grep -r "use.*worker_registry" bin/20_rbee_hive/

# 2. Delete crate
rm -rf bin/25_rbee_hive_crates/worker-registry

# 3. Remove from workspace
# Edit Cargo.toml, remove from members list

# 4. Verify compilation
cargo check --all
```

**Expected Impact:** -300 LOC

---

### 1.2 Delete Empty daemon-ensure Crate

**Location:** `bin/99_shared_crates/daemon-ensure`

**Rationale:** File is EMPTY (1 blank line). Replaced by `daemon-lifecycle`.

**Steps:**
```bash
# 1. Verify empty
cat bin/99_shared_crates/daemon-ensure/src/lib.rs
# Should be blank

# 2. Verify no usage
grep -r "daemon-ensure" --include="*.toml"

# 3. Delete crate
rm -rf bin/99_shared_crates/daemon-ensure

# 4. Remove from workspace
# Edit Cargo.toml, remove from members list

# 5. Verify compilation
cargo check --all
```

**Expected Impact:** -10 LOC

---

### 1.3 Delete Unused hive-core Crate

**Location:** `bin/99_shared_crates/hive-core`

**Rationale:** Not used anywhere. No binary or crate imports it.

**Steps:**
```bash
# 1. Verify no usage
grep -r "hive-core" --include="*.toml" | grep -v "hive-core/Cargo.toml"
# Should return nothing

# 2. Delete crate
rm -rf bin/99_shared_crates/hive-core

# 3. Remove from workspace
# Edit Cargo.toml, remove from members list

# 4. Verify compilation
cargo check --all
```

**Expected Impact:** -200 LOC

---

### 1.4 Clean Heartbeat Crate (Remove Hive Logic)

**Location:** `bin/99_shared_crates/heartbeat/`

**Rationale:** Hive heartbeat aggregation removed in TEAM-261. Only worker → queen heartbeats remain.

**Files to Delete:**
- `src/hive.rs` - Hive → Queen heartbeat (obsolete)
- `src/hive_receiver.rs` - Hive receives worker heartbeats (obsolete)
- `src/queen_receiver.rs` - Queen receives HIVE heartbeats (obsolete)

**Files to Update:**
- `src/lib.rs` - Remove hive module exports
- `src/types.rs` - Remove `HiveHeartbeatPayload`

**Steps:**

1. **Delete obsolete files:**
```bash
cd bin/99_shared_crates/heartbeat/src/
rm hive.rs hive_receiver.rs queen_receiver.rs
```

2. **Update lib.rs:**
```rust
// REMOVE these lines:
pub mod hive;
pub mod hive_receiver;
pub mod queen_receiver;
pub use hive::{start_hive_heartbeat_task, HiveHeartbeatConfig, WorkerStateProvider};

// KEEP these lines:
pub mod worker;
pub mod types;
pub use worker::{start_worker_heartbeat_task, WorkerHeartbeatConfig};
pub use types::{HealthStatus, WorkerHeartbeatPayload};
```

3. **Update types.rs:**
```rust
// REMOVE HiveHeartbeatPayload struct
// REMOVE WorkerState struct (if only used by hive heartbeat)
// KEEP WorkerHeartbeatPayload
```

4. **Update documentation:**
```rust
// Update lib.rs module doc
//! Heartbeat mechanism for health monitoring
//!
//! **Architecture (TEAM-261):**
//! Worker → Queen: POST /v1/worker-heartbeat (30s interval)
//!   Payload: { worker_id, timestamp, health_status }
```

5. **Verify compilation:**
```bash
cargo check -p rbee-heartbeat
cargo check --all
```

**Expected Impact:** -400 LOC

---

## Phase 2: Rename for Clarity (MEDIUM Priority)

### 2.1 Rename hive-registry → worker-registry

**Rationale:** After TEAM-261, this registry tracks WORKERS (not hives). 90% of API is worker-focused.

**Steps:**

1. **Rename directory:**
```bash
cd bin/15_queen_rbee_crates/
mv hive-registry worker-registry
```

2. **Update Cargo.toml:**
```toml
# bin/15_queen_rbee_crates/worker-registry/Cargo.toml
[package]
name = "queen-rbee-worker-registry"  # Changed from hive-registry
```

3. **Rename struct in lib.rs:**
```rust
// OLD
pub struct HiveRegistry { ... }

// NEW
pub struct WorkerRegistry { ... }
```

4. **Update all imports in queen-rbee:**
```rust
// bin/10_queen_rbee/src/main.rs
// OLD
use queen_rbee_hive_registry::HiveRegistry;

// NEW
use queen_rbee_worker_registry::WorkerRegistry;
```

5. **Update workspace Cargo.toml:**
```toml
# Cargo.toml
[workspace]
members = [
    # ...
    "bin/15_queen_rbee_crates/worker-registry",  # Changed
]
```

6. **Update README:**
```markdown
# queen-rbee-worker-registry

In-memory registry for tracking worker runtime state.

After TEAM-261, workers send heartbeats directly to queen.
This registry tracks all workers across all hives.
```

7. **Verify compilation:**
```bash
cargo check --all
```

**Expected Impact:** 0 LOC (refactor only)

---

### 2.2 Rename SseBroadcaster → SseChannelRegistry

**Rationale:** Name is misleading. It's not a broadcast channel, it's a HashMap of isolated MPSC channels.

**Location:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

**Steps:**

1. **Rename struct:**
```rust
// OLD
pub struct SseBroadcaster { ... }

// NEW
pub struct SseChannelRegistry { ... }
```

2. **Rename static:**
```rust
// OLD
static SSE_BROADCASTER: once_cell::sync::Lazy<SseBroadcaster> = ...;

// NEW
static SSE_CHANNEL_REGISTRY: once_cell::sync::Lazy<SseChannelRegistry> = ...;
```

3. **Update all internal usages:**
```rust
// Update all references in sse_sink.rs
SSE_CHANNEL_REGISTRY.create_job_channel(...)
SSE_CHANNEL_REGISTRY.send_to_job(...)
SSE_CHANNEL_REGISTRY.take_job_receiver(...)
```

4. **Update documentation:**
```rust
/// Job-scoped SSE channel registry.
///
/// TEAM-204: SECURITY FIX - Job-scoped channels ONLY. No global channel.
/// TEAM-205: SIMPLIFIED - Use MPSC instead of broadcast.
/// TEAM-262: RENAMED - "Broadcaster" was misleading (it's a registry of isolated channels)
pub struct SseChannelRegistry { ... }
```

5. **Verify compilation:**
```bash
cargo check -p observability-narration-core
cargo check --all
```

**Expected Impact:** 0 LOC (refactor only)

---

## Phase 3: Implement Queen Lifecycle (HIGH Priority)

### 3.1 Add Queen Commands to rbee-keeper

**Location:** `bin/00_rbee_keeper/src/commands/queen.rs` (NEW FILE)

**Implementation:**

```rust
//! Queen lifecycle management commands
//!
//! TEAM-262: Added queen lifecycle (install, uninstall, rebuild, info)

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum QueenCommands {
    /// Start queen-rbee daemon
    Start,
    
    /// Stop queen-rbee daemon
    Stop,
    
    /// Query queen status
    Status,
    
    /// Rebuild queen with different configuration
    Rebuild {
        /// Include local hive for localhost operations (50-100x faster)
        #[arg(long)]
        with_local_hive: bool,
    },
    
    /// Show queen build configuration
    Info,
    
    /// Install queen binary
    Install {
        /// Binary path (optional, auto-detect from target/)
        #[arg(short, long)]
        binary: Option<String>,
    },
    
    /// Uninstall queen binary
    Uninstall,
}

pub async fn handle_queen_command(cmd: QueenCommands) -> Result<()> {
    match cmd {
        QueenCommands::Start => handle_queen_start().await,
        QueenCommands::Stop => handle_queen_stop().await,
        QueenCommands::Status => handle_queen_status().await,
        QueenCommands::Rebuild { with_local_hive } => handle_queen_rebuild(with_local_hive).await,
        QueenCommands::Info => handle_queen_info().await,
        QueenCommands::Install { binary } => handle_queen_install(binary).await,
        QueenCommands::Uninstall => handle_queen_uninstall().await,
    }
}

async fn handle_queen_start() -> Result<()> {
    // TODO: Implement
    println!("🚀 Starting queen-rbee...");
    Ok(())
}

async fn handle_queen_stop() -> Result<()> {
    // TODO: Implement
    println!("🛑 Stopping queen-rbee...");
    Ok(())
}

async fn handle_queen_status() -> Result<()> {
    // TODO: Implement
    println!("📊 Queen status...");
    Ok(())
}

async fn handle_queen_rebuild(with_local_hive: bool) -> Result<()> {
    println!("🔨 Rebuilding queen-rbee...");
    
    if with_local_hive {
        println!("✨ Building with integrated local hive...");
        // cargo build --release --bin queen-rbee --features local-hive
    } else {
        println!("📡 Building distributed queen (remote hives only)...");
        // cargo build --release --bin queen-rbee
    }
    
    // TODO: Implement build logic
    Ok(())
}

async fn handle_queen_info() -> Result<()> {
    // Query queen's /v1/build-info endpoint
    println!("📋 Queen build information:");
    // TODO: Implement
    Ok(())
}

async fn handle_queen_install(binary: Option<String>) -> Result<()> {
    println!("📦 Installing queen-rbee...");
    // TODO: Implement (similar to hive install)
    Ok(())
}

async fn handle_queen_uninstall() -> Result<()> {
    println!("🗑️  Uninstalling queen-rbee...");
    // TODO: Implement (similar to hive uninstall)
    Ok(())
}
```

**Wire into main.rs:**
```rust
// bin/00_rbee_keeper/src/main.rs

mod commands;
use commands::queen::{QueenCommands, handle_queen_command};

#[derive(Parser)]
enum Commands {
    /// Queen lifecycle management (NEW)
    Queen {
        #[command(subcommand)]
        command: QueenCommands,
    },
    
    // ... existing commands
}

match cli.command {
    Commands::Queen { command } => handle_queen_command(command).await?,
    // ... existing handlers
}
```

---

### 3.2 Add Smart Prompts for Localhost Optimization

**Location:** `bin/00_rbee_keeper/src/commands/hive.rs`

**Implementation:**

```rust
pub async fn handle_hive_install(alias: String) -> Result<()> {
    // Check if this is localhost
    if alias == "localhost" || is_localhost_alias(&alias)? {
        // Check queen's build configuration
        let queen_has_local_hive = check_queen_has_local_hive().await?;
        
        if !queen_has_local_hive {
            // PROMPT USER!
            eprintln!("⚠️  Performance Notice:");
            eprintln!();
            eprintln!("   You're installing a hive on localhost, but your queen-rbee");
            eprintln!("   was built without the 'local-hive' feature.");
            eprintln!();
            eprintln!("   📊 Performance comparison:");
            eprintln!("      • Current setup:  ~5-10ms overhead (HTTP)");
            eprintln!("      • Integrated:     ~0.1ms overhead (direct calls)");
            eprintln!();
            eprintln!("   💡 Recommendation:");
            eprintln!("      Rebuild queen-rbee with integrated hive for 50-100x faster");
            eprintln!("      localhost operations:");
            eprintln!();
            eprintln!("      $ rbee-keeper queen rebuild --with-local-hive");
            eprintln!();
            eprintln!("   ℹ️  Or continue with distributed setup if you have specific needs.");
            eprintln!();
            
            // Ask user
            print!("   Continue with distributed setup? [y/N]: ");
            std::io::stdout().flush()?;
            
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            
            if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
                eprintln!("\n✋ Installation cancelled.");
                eprintln!("   Run: rbee-keeper queen rebuild --with-local-hive");
                return Ok(());
            }
        }
    }
    
    // Proceed with installation
    // ... existing hive install logic
}

async fn check_queen_has_local_hive() -> Result<bool> {
    // Query queen for its build configuration
    let response = reqwest::get("http://localhost:8500/v1/build-info")
        .await?
        .json::<BuildInfo>()
        .await?;
    
    Ok(response.features.contains(&"local-hive".to_string()))
}

#[derive(serde::Deserialize)]
struct BuildInfo {
    version: String,
    features: Vec<String>,
    build_timestamp: String,
}
```

---

### 3.3 Add Build Info Endpoint to Queen

**Location:** `bin/10_queen_rbee/src/http/build_info.rs` (NEW FILE)

**Implementation:**

```rust
//! Build information endpoint
//!
//! TEAM-262: Added /v1/build-info endpoint for rbee-keeper to query

use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct BuildInfo {
    pub version: String,
    pub features: Vec<String>,
    pub build_timestamp: String,
}

pub async fn handle_build_info() -> Json<BuildInfo> {
    let mut features = vec![];
    
    #[cfg(feature = "local-hive")]
    features.push("local-hive".to_string());
    
    Json(BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        features,
        build_timestamp: env!("BUILD_TIMESTAMP").unwrap_or("unknown").to_string(),
    })
}
```

**Wire into router:**
```rust
// bin/10_queen_rbee/src/main.rs

mod http;
use http::build_info::handle_build_info;

let app = Router::new()
    .route("/v1/build-info", get(handle_build_info))  // NEW
    // ... existing routes
```

---

## Testing Strategy

### Unit Tests

1. **Worker Registry:**
```bash
cargo test -p queen-rbee-worker-registry
```

2. **SSE Channel Registry:**
```bash
cargo test -p observability-narration-core
```

3. **Heartbeat (after cleanup):**
```bash
cargo test -p rbee-heartbeat
```

### Integration Tests

1. **Queen lifecycle:**
```bash
# Test queen commands
rbee-keeper queen info
rbee-keeper queen status
```

2. **Smart prompts:**
```bash
# Test localhost hive install prompt
rbee-keeper hive install localhost
# Should prompt about local-hive feature
```

### Compilation Verification

```bash
# After each phase
cargo check --all
cargo build --all
cargo test --all
```

---

## Documentation Updates

### 1. Update Architecture Docs

**Files to update:**
- `.arch/01_COMPONENTS_PART_2.md` - Add queen lifecycle section
- `.arch/03_DATA_FLOW_PART_4.md` - Update heartbeat flow
- `.arch/CHANGELOG.md` - Add TEAM-262 entry

### 2. Update README Files

**Files to update:**
- `bin/15_queen_rbee_crates/worker-registry/README.md` - Update from hive-registry
- `bin/99_shared_crates/heartbeat/README.md` - Remove hive heartbeat references

---

## Completion Checklist

### Phase 1: Delete Dead Code
- [ ] Deleted `bin/25_rbee_hive_crates/worker-registry`
- [ ] Deleted `bin/99_shared_crates/daemon-ensure`
- [ ] Deleted `bin/99_shared_crates/hive-core`
- [ ] Cleaned heartbeat crate (removed hive logic)
- [ ] Removed from workspace Cargo.toml
- [ ] Compilation successful

### Phase 2: Rename for Clarity
- [ ] Renamed `hive-registry` → `worker-registry`
- [ ] Renamed `HiveRegistry` → `WorkerRegistry`
- [ ] Updated all imports in queen-rbee
- [ ] Renamed `SseBroadcaster` → `SseChannelRegistry`
- [ ] Updated documentation
- [ ] Compilation successful

### Phase 3: Queen Lifecycle
- [ ] Created `bin/00_rbee_keeper/src/commands/queen.rs`
- [ ] Implemented queen commands (start, stop, status, rebuild, info, install, uninstall)
- [ ] Added smart prompts to hive install
- [ ] Created `/v1/build-info` endpoint in queen
- [ ] Wired into rbee-keeper main.rs
- [ ] Tested queen commands
- [ ] Compilation successful

### Documentation
- [ ] Updated `.arch/01_COMPONENTS_PART_2.md`
- [ ] Updated `.arch/03_DATA_FLOW_PART_4.md`
- [ ] Updated `.arch/CHANGELOG.md`
- [ ] Updated worker-registry README
- [ ] Updated heartbeat README

### Final Verification
- [ ] All tests pass: `cargo test --all`
- [ ] All binaries compile: `cargo build --all`
- [ ] No references to deleted crates
- [ ] Architecture docs accurate
- [ ] TEAM_262_COMPLETE.md created

---

## Expected Outcomes

### Code Metrics
- **Deleted:** ~910 LOC (dead code)
- **Added:** ~200 LOC (queen lifecycle)
- **Net:** -710 LOC (cleaner codebase)

### Crates Removed
1. `bin/25_rbee_hive_crates/worker-registry`
2. `bin/99_shared_crates/daemon-ensure`
3. `bin/99_shared_crates/hive-core`

### Crates Cleaned
1. `bin/99_shared_crates/heartbeat` (removed hive logic)

### Crates Renamed
1. `hive-registry` → `worker-registry`
2. `SseBroadcaster` → `SseChannelRegistry`

### Features Added
1. Queen lifecycle management (install, uninstall, rebuild, info)
2. Smart prompts for localhost optimization
3. `/v1/build-info` endpoint

---

## Handoff Document

When complete, create `TEAM_262_COMPLETE.md` with:
1. Summary of changes
2. Files modified/deleted/created
3. Compilation status
4. Test results
5. Documentation updates
6. Known issues (if any)
7. Next steps for TEAM-263

---

## Questions?

**Refer to:**
- `.arch/CLEANUP_PLAN_TEAM261.md` - Detailed rationale for each cleanup
- `.arch/SSE_ROUTING_ANALYSIS.md` - SSE routing verification
- `bin/.plan/TEAM_261_IMPLEMENTATION_COMPLETE.md` - Context on TEAM-261 changes

**Need help?** Check existing patterns:
- Hive lifecycle: `bin/15_queen_rbee_crates/hive-lifecycle/`
- Daemon management: `bin/99_shared_crates/daemon-lifecycle/`
- CLI commands: `bin/00_rbee_keeper/src/commands/hive.rs`

---

**Status:** Ready for implementation  
**Priority:** HIGH  
**Estimated Time:** 2-3 days  
**Next Team:** TEAM-263 (TBD based on priorities)
