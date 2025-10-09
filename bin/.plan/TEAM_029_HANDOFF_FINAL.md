# TEAM-029 Final Handoff: Architecture Redesign Required

**Date:** 2025-10-10T00:28:00+02:00  
**From:** TEAM-029  
**To:** TEAM-030  
**Status:** 🔴 **CRITICAL ARCHITECTURE CHANGE REQUIRED**

---

## ⚠️ READ FIRST - CRITICAL CONTEXT

**STOP! Before implementing anything, read:**
1. `.windsurf/rules/destructive-actions.md` - You are ALLOWED to be destructive
2. `.windsurf/rules/dev-bee-rules.md` - Development standards
3. `.windsurf/rules/rust-rules.md` - Rust coding standards
4. This entire handoff document

**We are at v0.1.0 - destructive changes are ENCOURAGED for cleanup.**

---

## 🎯 Mission

**Redesign the architecture to support two distinct usage modes:**

1. **Ephemeral Mode (bee-keeper)** - Single inference, everything dies after
2. **Persistent Mode (queen-rbee daemon)** - Long-running, worker reuse

**Current implementation uses SQLite - WRONG APPROACH for ephemeral mode.**

---

## 💡 The Revelation (from User)

### Current Wrong Assumption
We built persistent storage (SQLite) for worker registry and model catalog, assuming workers stay alive between requests.

### Correct Architecture (User's Vision)

```
bee-keeper (CLI - ephemeral, single inference)
    ↓ spawns
queen-rbee (orchestrator daemon - ephemeral if spawned by keeper)
    ↓ SSH spawns
rbee-hive (pool manager - ephemeral if spawned by queen)
    ↓ spawns
llm-worker-rbee (worker - ephemeral)

LIFECYCLE:
- bee-keeper infer → spawns everything → gets result → KILLS EVERYTHING → exits
- Cascading shutdown: queen dies → hives die → workers die
- NO PERSISTENCE NEEDED - everything is in-memory!
```

### Two Usage Modes

#### Mode 1: bee-keeper (Ephemeral - Testing/Single Inference)
```bash
rbee infer --node mac --model tinyllama --prompt "hello"
# 1. Spawns queen-rbee (if not running)
# 2. Queen spawns rbee-hive on mac via SSH
# 3. Hive spawns llm-worker-rbee
# 4. Worker processes inference
# 5. bee-keeper gets result via SSE
# 6. bee-keeper kills queen
# 7. Queen kills all hives (cascading)
# 8. Hives kill all workers
# 9. VRAM cleaned, everything dead
```

**Key Points:**
- ✅ Short-lived - everything dies after task
- ✅ Clean VRAM - no lingering workers
- ✅ No persistence - in-memory only
- ✅ Simple testing - one command, clean slate
- ❌ NOT for production - spawning overhead

#### Mode 2: queen-rbee daemon (Persistent - Production)
```bash
# On control node
queen-rbee daemon &

# Workers stay alive, reused across requests
# User manages lifecycle manually
# Worker registry persists (but still in-memory)
```

**Key Points:**
- ✅ Production use - worker reuse
- ✅ Performance - no spawn overhead
- ✅ Multi-tenant - multiple clients
- ⚠️ Manual lifecycle - user must manage
- ⚠️ VRAM persistence - workers stay loaded

---

## 🔥 What TEAM-029 Built (Needs Redesign)

### ✅ What We Implemented (Good Parts)

1. **Model Catalog** (`bin/shared-crates/model-catalog/`)
   - SQLite-backed model tracking
   - Tracks downloaded models with provider, reference, local path
   - **STATUS:** ⚠️ Needs simplification (remove SQLite)

2. **Model Provisioner** (`bin/rbee-hive/src/provisioner.rs`)
   - Downloads models from HuggingFace via `llorch-models` script
   - Maps HF references to model names
   - **STATUS:** ✅ Keep this, just remove catalog dependency

3. **Integrated Worker Spawn Flow**
   - Checks catalog before spawning
   - Downloads model if missing
   - Passes resolved path to worker
   - **STATUS:** ✅ Keep logic, simplify storage

4. **Bug Fixes**
   - Fixed SQLite connection string format
   - Fixed localhost DNS resolution
   - Improved fail-fast logic (20s instead of 5min timeout)
   - **STATUS:** ✅ Keep all fixes

### ❌ What Needs to Change (SQLite Removal)

1. **Worker Registry** (`bin/shared-crates/worker-registry/`)
   - Currently: SQLite database
   - **SHOULD BE:** `Arc<RwLock<HashMap<String, WorkerInfo>>>`
   - **REASON:** Ephemeral mode doesn't need persistence

2. **Model Catalog** (`bin/shared-crates/model-catalog/`)
   - Currently: SQLite database
   - **SHOULD BE:** Filesystem scan + simple cache
   - **REASON:** Models are files, just check if they exist

3. **rbee-keeper** (`bin/rbee-keeper/`)
   - Currently: Assumes persistent queen-rbee
   - **SHOULD:** Spawn queen, manage lifecycle, kill on exit
   - **REASON:** Ephemeral mode requires lifecycle management

---

## 📋 Implementation Plan for TEAM-030

### Phase 1: Remove SQLite Dependencies ⚠️ DESTRUCTIVE

**Files to DELETE:**
```
bin/shared-crates/worker-registry/  # Delete entire crate
bin/shared-crates/model-catalog/    # Delete entire crate
```

**Files to MODIFY:**
```
Cargo.toml                          # Remove from workspace members
bin/rbee-hive/Cargo.toml           # Remove dependencies
bin/rbee-keeper/Cargo.toml         # Remove dependencies
bin/queen-rbee/Cargo.toml          # Remove dependencies (if added)
```

**Justification:** Per `destructive-actions.md`, we're v0.1.0 - cleanup is encouraged.

### Phase 2: Implement In-Memory Worker Registry

**Create:** `bin/rbee-hive/src/registry.rs` (replace shared crate)

```rust
//! In-memory worker registry
//! TEAM-030: Simplified from SQLite-based shared crate
//! 
//! Ephemeral storage - lost on restart (by design)

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState,
    pub last_activity: SystemTime,
    pub slots_total: u32,
    pub slots_available: u32,
}

#[derive(Clone, Debug)]
pub enum WorkerState {
    Loading,
    Idle,
    Busy,
    Failed,
}

pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(&self, worker: WorkerInfo) {
        let mut workers = self.workers.write().await;
        workers.insert(worker.id.clone(), worker);
    }

    pub async fn find(&self, id: &str) -> Option<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.get(id).cloned()
    }

    pub async fn list(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.values().cloned().collect()
    }

    pub async fn update_state(&self, id: &str, state: WorkerState) {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(id) {
            worker.state = state;
            worker.last_activity = SystemTime::now();
        }
    }

    pub async fn remove(&self, id: &str) {
        let mut workers = self.workers.write().await;
        workers.remove(id);
    }

    // TEAM-030: Add this for ephemeral mode
    pub async fn find_by_node_and_model(
        &self,
        node: &str,
        model_ref: &str,
    ) -> Option<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.values()
            .find(|w| {
                w.url.contains(node) 
                && w.model_ref == model_ref
                && matches!(w.state, WorkerState::Idle)
            })
            .cloned()
    }
}
```

**Benefits:**
- ✅ No database files
- ✅ No async SQLite overhead
- ✅ Simple, fast, testable
- ✅ Ephemeral by design
- ✅ No cleanup needed

### Phase 3: Simplify Model Catalog

**Create:** `bin/rbee-hive/src/model_cache.rs` (replace catalog)

```rust
//! Model cache - filesystem-based model tracking
//! TEAM-030: Simplified from SQLite-based catalog
//!
//! Just checks if models exist on disk

use std::path::{Path, PathBuf};
use anyhow::Result;

pub struct ModelCache {
    base_dir: PathBuf,
}

impl ModelCache {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// Find model by reference
    /// Returns local path if model exists
    pub fn find_model(&self, reference: &str) -> Option<PathBuf> {
        // Extract model name from reference
        let model_name = reference.split('/').last()?.to_lowercase();
        let model_dir = self.base_dir.join(&model_name);

        if !model_dir.exists() {
            return None;
        }

        // Find first .gguf file
        std::fs::read_dir(&model_dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .find(|entry| {
                entry.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<PathBuf> {
        std::fs::read_dir(&self.base_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .flat_map(|dir| {
                        std::fs::read_dir(dir.path())
                            .ok()
                            .into_iter()
                            .flatten()
                            .filter_map(|e| e.ok())
                            .filter(|e| {
                                e.path().extension()
                                    .and_then(|ext| ext.to_str())
                                    .map(|ext| ext == "gguf")
                                    .unwrap_or(false)
                            })
                            .map(|e| e.path())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}
```

**Benefits:**
- ✅ No database
- ✅ Source of truth is filesystem
- ✅ Works with existing `llorch-models` script
- ✅ Simple, no async needed

### Phase 4: Add Cascading Shutdown

**Update:** `bin/queen-rbee/src/main.rs`

```rust
//! Queen-rbee - Orchestrator daemon
//! TEAM-030: Add cascading shutdown support

use std::sync::Arc;
use tokio::sync::RwLock;

struct QueenRbee {
    hives: Arc<RwLock<Vec<HiveConnection>>>,
    ephemeral_mode: bool, // Set to true if spawned by bee-keeper
}

impl QueenRbee {
    async fn shutdown(&self) {
        tracing::info!("Queen shutting down - cascading to hives");
        
        let hives = self.hives.read().await;
        for hive in hives.iter() {
            // Send shutdown signal to each hive
            if let Err(e) = hive.shutdown().await {
                tracing::error!("Failed to shutdown hive {}: {}", hive.node, e);
            }
        }
        
        tracing::info!("All hives notified of shutdown");
    }
}

// Register signal handlers
#[tokio::main]
async fn main() -> Result<()> {
    let queen = Arc::new(QueenRbee::new());
    
    // Handle SIGTERM/SIGINT
    let queen_clone = queen.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        queen_clone.shutdown().await;
        std::process::exit(0);
    });
    
    queen.run().await
}
```

**Update:** `bin/rbee-hive/src/main.rs`

```rust
//! rbee-hive - Pool manager
//! TEAM-030: Add cascading shutdown support

struct RbeeHive {
    workers: Arc<WorkerRegistry>,
}

impl RbeeHive {
    async fn shutdown(&self) {
        tracing::info!("Hive shutting down - killing all workers");
        
        let workers = self.workers.list().await;
        for worker in workers {
            // Send SIGTERM to worker process
            if let Err(e) = kill_worker(&worker.id).await {
                tracing::error!("Failed to kill worker {}: {}", worker.id, e);
            }
        }
        
        tracing::info!("All workers killed");
    }
}
```

### Phase 5: Update bee-keeper Lifecycle

**Update:** `bin/rbee-keeper/src/commands/infer.rs`

```rust
//! Inference command
//! TEAM-030: Add lifecycle management for ephemeral mode

pub async fn handle(args: InferArgs) -> Result<()> {
    // 1. Check if queen-rbee is running
    let queen_running = check_queen_running().await?;
    let spawned_queen = if !queen_running {
        println!("Spawning queen-rbee...");
        Some(spawn_queen().await?)
    } else {
        None
    };

    // 2. Run inference (existing logic)
    let result = run_inference(&args).await;

    // 3. Cleanup if we spawned queen
    if let Some(queen_pid) = spawned_queen {
        println!("Shutting down queen-rbee...");
        kill_process(queen_pid).await?;
        println!("Cleanup complete - all processes terminated");
    }

    result
}

async fn spawn_queen() -> Result<u32> {
    let child = tokio::process::Command::new("queen-rbee")
        .arg("daemon")
        .arg("--ephemeral") // TEAM-030: Add this flag
        .spawn()?;
    
    // Wait for queen to be ready
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    Ok(child.id().unwrap())
}

async fn kill_process(pid: u32) -> Result<()> {
    // Send SIGTERM
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    
    // Wait for graceful shutdown
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    Ok(())
}
```

### Phase 6: Update Specs

**Create:** `bin/.specs/ARCHITECTURE_MODES.md`

```markdown
# Architecture: Two Usage Modes

## Mode 1: Ephemeral (bee-keeper)

**Purpose:** Testing, single inference, clean environment

**Lifecycle:**
1. User runs: `rbee infer --node mac --model tinyllama --prompt "hello"`
2. bee-keeper checks if queen-rbee is running
3. If not, spawns queen-rbee with `--ephemeral` flag
4. Queen spawns rbee-hive on target node via SSH
5. Hive spawns llm-worker-rbee
6. Worker processes inference
7. bee-keeper receives result via SSE
8. bee-keeper sends SIGTERM to queen
9. Queen cascades shutdown to all hives
10. Hives kill all workers
11. All processes exit, VRAM cleaned

**Storage:**
- Worker registry: In-memory HashMap (lost on exit)
- Model catalog: Filesystem scan (persistent)
- No database files

**Use Cases:**
- Testing new models
- One-off inferences
- CI/CD pipelines
- Development

## Mode 2: Persistent (queen-rbee daemon)

**Purpose:** Production, worker reuse, performance

**Lifecycle:**
1. User runs: `queen-rbee daemon &`
2. Queen stays alive, manages hives
3. Workers stay alive between requests
4. User manually manages lifecycle

**Storage:**
- Worker registry: In-memory HashMap (lost on restart)
- Model catalog: Filesystem scan (persistent)
- No database files

**Use Cases:**
- Production workloads
- Multi-tenant systems
- High-throughput inference
- Long-running services

## Design Decisions

### Why No SQLite?

**Worker Registry:**
- Ephemeral mode: Everything dies after task → no persistence needed
- Persistent mode: In-memory is faster, simpler
- Health monitoring rebuilds state anyway

**Model Catalog:**
- Models are files on disk → filesystem is source of truth
- No need to track metadata in DB
- `llorch-models` script handles downloads

### Cascading Shutdown

**Signal Flow:**
```
SIGTERM → queen-rbee
    ↓
queen sends shutdown to all hives
    ↓
hives send SIGTERM to all workers
    ↓
all processes exit
```

**Implementation:**
- Queen tracks hive connections
- Hive tracks worker PIDs
- Graceful shutdown with timeout
- Force kill if timeout exceeded

### Ephemeral Flag

**queen-rbee --ephemeral:**
- Enables aggressive cleanup
- Shorter timeouts
- No worker reuse
- Dies after all tasks complete
```

---

## 🗂️ Files Modified by TEAM-029

### Created (Will Need Modification)
- `bin/shared-crates/model-catalog/` - **DELETE** (replace with simple cache)
- `bin/rbee-hive/src/provisioner.rs` - **KEEP** (remove catalog dependency)
- `bin/.plan/TEAM_029_COMPLETION_SUMMARY.md` - **ARCHIVE**
- `bin/.plan/TEAM_029_HANDOFF.md` - **ARCHIVE**

### Modified (Update for New Architecture)
- `Cargo.toml` - Remove model-catalog from workspace
- `bin/rbee-hive/Cargo.toml` - Remove model-catalog dependency
- `bin/rbee-hive/src/main.rs` - Remove provisioner module
- `bin/rbee-hive/src/commands/daemon.rs` - Remove catalog init
- `bin/rbee-hive/src/http/routes.rs` - Simplify AppState
- `bin/rbee-hive/src/http/workers.rs` - Use simple cache instead of catalog
- `bin/rbee-keeper/src/commands/infer.rs` - Add lifecycle management
- `bin/shared-crates/worker-registry/` - **DELETE** (move to rbee-hive)

### Keep (Good Work!)
- `bin/.specs/.gherkin/test-001-mvp-local.sh` - ✅ Keep
- `bin/.specs/.gherkin/test-001-mvp-preflight.sh` - ✅ Keep
- Bug fixes in `worker-registry` and `infer.rs` - ✅ Port to new code

---

## 🎯 Acceptance Criteria for TEAM-030

### Must Have ✅

1. **SQLite Removed**
   - [ ] No SQLite dependencies in Cargo.toml files
   - [ ] No .db files created at runtime
   - [ ] worker-registry crate deleted
   - [ ] model-catalog crate deleted

2. **In-Memory Registry**
   - [ ] Worker registry is HashMap-based
   - [ ] No persistence to disk
   - [ ] Fast lookups (< 1ms)
   - [ ] Thread-safe (Arc<RwLock>)

3. **Filesystem Model Cache**
   - [ ] Scans .test-models directory
   - [ ] Returns paths to .gguf files
   - [ ] No database tracking
   - [ ] Works with llorch-models script

4. **Cascading Shutdown**
   - [ ] Queen kills all hives on exit
   - [ ] Hives kill all workers on exit
   - [ ] Graceful shutdown (SIGTERM)
   - [ ] Force kill after timeout (5s)

5. **bee-keeper Lifecycle**
   - [ ] Spawns queen if not running
   - [ ] Tracks spawned queen PID
   - [ ] Kills queen on exit
   - [ ] Cleans up all processes

6. **Tests Pass**
   - [ ] `cargo test` passes
   - [ ] `./bin/.specs/.gherkin/test-001-mvp-local.sh` passes
   - [ ] No database files left after test
   - [ ] All processes killed after test

### Nice to Have 🎁

1. **Ephemeral Flag**
   - [ ] `queen-rbee daemon --ephemeral` flag
   - [ ] Different behavior in ephemeral mode
   - [ ] Auto-shutdown when idle

2. **Metrics**
   - [ ] Track spawned processes
   - [ ] Track cleanup success rate
   - [ ] Log cascade shutdown timing

3. **Documentation**
   - [ ] Update README with two modes
   - [ ] Add architecture diagram
   - [ ] Document lifecycle

---

## 📚 Reference Documents

**Must Read:**
1. `bin/.specs/.gherkin/test-001-mvp.md` - Source of truth for MVP flow
2. `bin/.specs/.gherkin/test-001.md` - Original user vision
3. `.windsurf/rules/destructive-actions.md` - You can delete things!
4. `.windsurf/rules/dev-bee-rules.md` - Development standards

**Helpful:**
1. `scripts/llorch-models` - Model download script (reuse this)
2. `bin/.plan/TEAM_028_HANDOFF.md` - Previous team's work
3. `bin/.plan/TEAM_029_COMPLETION_SUMMARY.md` - What we built

---

## 🚨 Common Pitfalls to Avoid

### ❌ Don't Do This:
1. **Keep SQLite "just in case"** - No! Delete it. In-memory is the way.
2. **Add new persistence layer** - No! Ephemeral means ephemeral.
3. **Complex lifecycle management** - Keep it simple: spawn → use → kill.
4. **Ignore cascading shutdown** - Critical! Must kill all children.
5. **Forget to test cleanup** - Verify no lingering processes.

### ✅ Do This:
1. **Delete aggressively** - We're v0.1.0, cleanup is good
2. **Keep it simple** - HashMap > Database for this use case
3. **Test lifecycle** - Spawn, run, kill, verify clean
4. **Document modes** - Make it clear when to use each mode
5. **Follow the rules** - Read `.windsurf/rules/` first

---

## 💬 Questions for User (If Needed)

If you get stuck, ask the user:

1. **Queen spawning:** Should bee-keeper always spawn queen, or check if running?
2. **Timeout values:** How long to wait for graceful shutdown before force kill?
3. **Error handling:** What if cascade shutdown fails? Leave orphans or retry?
4. **Model downloads:** Should ephemeral mode download models, or require pre-download?

---

## 🎉 What Success Looks Like

```bash
# User runs this
rbee infer --node localhost --model tinyllama --prompt "hello world"

# System does this
[bee-keeper] Spawning queen-rbee...
[queen-rbee] Starting in ephemeral mode
[queen-rbee] Spawning hive on localhost...
[rbee-hive] Checking for model tinyllama...
[rbee-hive] Model found: .test-models/tinyllama/tinyllama.gguf
[rbee-hive] Spawning worker...
[worker] Loading model...
[worker] Ready for inference
[bee-keeper] Streaming tokens: hello world from a tiny llama...
[bee-keeper] Inference complete
[bee-keeper] Shutting down queen...
[queen-rbee] Cascading shutdown to hives...
[rbee-hive] Killing workers...
[worker] Exiting
[rbee-hive] Exiting
[queen-rbee] Exiting
[bee-keeper] Cleanup complete

# User sees this
$ ps aux | grep rbee
# Nothing! All clean.

$ ls ~/.rbee/
# No .db files! Just logs.

$ nvidia-smi
# VRAM freed! No lingering models.
```

---

## 📝 Final Notes from TEAM-029

**What we learned:**
- SQLite was over-engineering for ephemeral workloads
- User's vision is brilliant: ephemeral mode = clean slate every time
- Cascading shutdown is the key to proper cleanup
- Two modes (ephemeral vs persistent) serve different needs

**What we're proud of:**
- Model provisioner logic (keep this!)
- Bug fixes (port these!)
- Preflight checks (keep these!)
- Fail-fast improvements (keep these!)

**What we regret:**
- Building SQLite integration (but good learning!)
- Not asking about lifecycle earlier
- Over-engineering the persistence layer

**Advice for TEAM-030:**
- Read the rules first
- Delete aggressively
- Keep it simple
- Test the lifecycle thoroughly
- The user's vision is correct - trust it

---

**Good luck, TEAM-030! You've got this! 🚀**

**Signed:** TEAM-029  
**Date:** 2025-10-10T00:28:00+02:00  
**Status:** Architecture redesign required, but foundation is solid
