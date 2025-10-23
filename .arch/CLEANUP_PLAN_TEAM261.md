# CLEANUP PLAN: Post-TEAM-261 Architectural Changes

**Date:** Oct 23, 2025  
**Status:** 🚧 PENDING IMPLEMENTATION  
**Trigger:** TEAM-261 heartbeat simplification

---

## Executive Summary

TEAM-261 removed hive heartbeats (workers now send directly to queen), which makes several crates and components obsolete. This document tracks all deprecated code that needs cleanup.

---

## 1. Queen Lifecycle Management (NEW)

### Status: ❌ NOT IMPLEMENTED

**Requirement:** Add queen install/uninstall operations to rbee-keeper.

### Rationale

With integrated mode (local-hive feature), users need to manage queen builds:
- Build queen with/without local-hive feature
- Install queen binary
- Uninstall queen binary
- Query queen build info

### Implementation Needed

**Location:** `bin/00_rbee_keeper/src/commands/queen.rs`

```rust
pub enum QueenCommands {
    /// Start queen-rbee daemon
    Start,
    
    /// Stop queen-rbee daemon
    Stop,
    
    /// Query queen status
    Status,
    
    /// Rebuild queen with different configuration
    Rebuild {
        #[arg(long)]
        with_local_hive: bool,
    },
    
    /// Show queen build configuration
    Info,
    
    /// Install queen binary (NEW)
    Install {
        /// Binary path (optional, auto-detect from target/)
        #[arg(short, long)]
        binary: Option<String>,
    },
    
    /// Uninstall queen binary (NEW)
    Uninstall,
}
```

**Smart Prompts:** When installing localhost hive, detect queen build and prompt for rebuild with local-hive if needed.

**Priority:** HIGH (needed for integrated mode UX)

---

## 2. Registry Renaming: hive-registry → worker-registry

### Status: ⚠️ NEEDS RENAMING

**Current:** `bin/15_queen_rbee_crates/hive-registry`  
**Should be:** `bin/15_queen_rbee_crates/worker-registry`

### Rationale

After TEAM-261:
- Hive heartbeats REMOVED
- Workers send heartbeats directly to queen
- Registry tracks WORKER state, not hive state
- Name should reflect reality: "worker-registry"

### Evidence from Code

```rust
// bin/15_queen_rbee_crates/hive-registry/src/lib.rs
// Lines 170-284: WORKER REGISTRY FUNCTIONS

/// Get worker by ID (searches across all hives)
pub fn get_worker(&self, worker_id: &str) -> Option<(String, WorkerInfo)>

/// Get worker URL for direct inference
pub fn get_worker_url(&self, worker_id: &str) -> Option<String>

/// List all workers across all hives
pub fn list_all_workers(&self) -> Vec<(String, WorkerInfo)>

/// Find idle workers (state == "Idle")
pub fn find_idle_workers(&self) -> Vec<(String, WorkerInfo)>

/// Find best worker for model
pub fn find_best_worker_for_model(&self, model_id: &str) -> Option<(String, WorkerInfo)>
```

**Key Insight:** 90% of the API is worker-focused. Hive is just an organizational grouping.

### Migration Plan

1. **Rename crate:**
   ```bash
   mv bin/15_queen_rbee_crates/hive-registry \
      bin/15_queen_rbee_crates/worker-registry
   ```

2. **Update Cargo.toml:**
   ```toml
   # OLD
   name = "queen-rbee-hive-registry"
   
   # NEW
   name = "queen-rbee-worker-registry"
   ```

3. **Update imports in queen-rbee:**
   ```rust
   // OLD
   use queen_rbee_hive_registry::HiveRegistry;
   
   // NEW
   use queen_rbee_worker_registry::WorkerRegistry;
   ```

4. **Rename struct:**
   ```rust
   // OLD
   pub struct HiveRegistry { ... }
   
   // NEW
   pub struct WorkerRegistry { ... }
   ```

5. **Update documentation:**
   - README.md
   - Architecture docs
   - All references to "hive registry"

**Priority:** MEDIUM (naming clarity, not breaking functionality)

---

## 3. Deprecated: worker-registry in 25_rbee_hive_crates

### Status: ❌ SHOULD BE DELETED

**Location:** `bin/25_rbee_hive_crates/worker-registry`

### Rationale

This was created for HIVE to track workers. But after TEAM-261:
- Workers send heartbeats to QUEEN (not hive)
- Queen tracks all workers (in queen's worker-registry)
- Hive doesn't need worker tracking anymore

### Evidence

Hive crates don't need worker tracking. Hive only:
- Spawns workers (lifecycle)
- Downloads models (provisioning)
- Detects devices (capabilities)

Worker state tracking happens in QUEEN.

### Action

**DELETE:** `bin/25_rbee_hive_crates/worker-registry`

**Priority:** HIGH (cleanup dead code)

---

## 4. Deprecated: daemon-ensure

### Status: ❌ SHOULD BE DELETED

**Location:** `bin/99_shared_crates/daemon-ensure`

### Rationale

**File is EMPTY:**

```rust
// bin/99_shared_crates/daemon-ensure/src/lib.rs
// (literally empty - 1 blank line)
```

This was probably a stub that was never implemented or was replaced by `daemon-lifecycle`.

### Evidence

`daemon-lifecycle` crate exists and is actively used:
- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (220 LOC)
- Used by hive-lifecycle for daemon management

### Action

**DELETE:** `bin/99_shared_crates/daemon-ensure`

**Priority:** HIGH (dead code, empty file)

---

## 5. Deprecated: Hive Heartbeat Logic in heartbeat Crate

### Status: ⚠️ NEEDS CLEANUP

**Location:** `bin/99_shared_crates/heartbeat/src/hive.rs`

### Rationale

TEAM-261 removed hive heartbeats:
- Hives NO LONGER send heartbeats to queen
- Workers send heartbeats DIRECTLY to queen
- Hive aggregation logic is OBSOLETE

### Evidence from Code

```rust
// bin/99_shared_crates/heartbeat/src/lib.rs
// Lines 17-19: DEPRECATED ARCHITECTURE

//! - **Hive:** Collects worker heartbeats + sends aggregated heartbeats to queen (Hive → Queen)
//!   ^^^^ THIS IS REMOVED IN TEAM-261! ^^^^
```

### Modules to Clean

1. **DELETE:** `src/hive.rs` - Hive → Queen heartbeat (entire file obsolete)
2. **DELETE:** `src/hive_receiver.rs` - Hive receives worker heartbeats (obsolete)
3. **DELETE:** `src/queen_receiver.rs` - Queen receives HIVE heartbeats (obsolete)
4. **KEEP:** `src/worker.rs` - Worker → Queen heartbeat (still needed, but update target)
5. **KEEP:** `src/types.rs` - Keep WorkerHeartbeatPayload, remove HiveHeartbeatPayload

### Updated Architecture

```rust
// NEW heartbeat flow (TEAM-261)
Worker → Queen: POST /v1/worker-heartbeat (30s interval)
  Payload: { worker_id, timestamp, health_status }

// REMOVED (TEAM-261):
// Hive → Queen: POST /v1/heartbeat
```

### Migration Steps

1. **Delete obsolete files:**
   - `src/hive.rs`
   - `src/hive_receiver.rs`
   - `src/queen_receiver.rs`

2. **Update types.rs:**
   ```rust
   // REMOVE
   pub struct HiveHeartbeatPayload { ... }
   
   // KEEP
   pub struct WorkerHeartbeatPayload { ... }
   ```

3. **Update worker.rs:**
   ```rust
   // OLD
   pub async fn send_heartbeat_to_hive(...) { ... }
   
   // NEW
   pub async fn send_heartbeat_to_queen(...) { ... }
   ```

4. **Update lib.rs:**
   ```rust
   // REMOVE
   pub mod hive;
   pub mod hive_receiver;
   pub mod queen_receiver;
   pub use hive::{start_hive_heartbeat_task, ...};
   
   // KEEP (but update)
   pub mod worker;
   pub use worker::{start_worker_heartbeat_task, ...};
   ```

**Priority:** HIGH (confusing to have deprecated code)

---

## 6. Deprecated: hive-core Crate

### Status: ❌ SHOULD BE DELETED

**Location:** `bin/99_shared_crates/hive-core`

### Rationale

**Not used anywhere!**

```bash
$ grep -r "hive-core" --include="Cargo.toml"
# Only result: hive-core's own Cargo.toml
```

No binary or crate imports `hive-core`.

### Contents

```rust
// src/lib.rs
pub mod catalog;   // ModelCatalog
pub mod error;     // PoolError
pub mod worker;    // WorkerInfo
```

These types are either:
1. Duplicated elsewhere (WorkerInfo in worker-registry)
2. Not needed (ModelCatalog - we use different model management)
3. Not used (PoolError - we use anyhow::Result)

### Action

**DELETE:** `bin/99_shared_crates/hive-core`

**Priority:** HIGH (dead code, unused)

---

## 7. Naming Issue: SseBroadcaster

### Status: ⚠️ SHOULD BE RENAMED

**Location:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

### Rationale

**Name is misleading!**

```rust
pub struct SseBroadcaster {
    // NOT a broadcast channel!
    // It's a HashMap of per-job ISOLATED MPSC channels
    senders: Arc<Mutex<HashMap<String, mpsc::Sender<NarrationEvent>>>>,
    receivers: Arc<Mutex<HashMap<String, mpsc::Receiver<NarrationEvent>>>>,
}
```

**History:**
- TEAM-204: REMOVED global broadcast (privacy hazard)
- TEAM-205: Changed to MPSC (single receiver)
- Result: "Broadcaster" name no longer accurate

### Better Names

**Option 1:** `SseJobChannelManager` (accurate, verbose)  
**Option 2:** `SseChannelRegistry` (shorter, clear)  
**Option 3:** `SseJobRouter` (describes function)

**Recommendation:** `SseChannelRegistry`

### Migration

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

3. **Update all usages** (internal to sse_sink.rs)

**Priority:** LOW (naming clarity, no functional change)

---

## Summary Table

| Item | Status | Priority | Action | LOC Impact |
|------|--------|----------|--------|------------|
| 1. Queen lifecycle | ❌ Not Implemented | HIGH | Implement | +200 LOC |
| 2. hive-registry → worker-registry | ⚠️ Needs Rename | MEDIUM | Rename | 0 (refactor) |
| 3. 25_rbee_hive_crates/worker-registry | ❌ Delete | HIGH | Delete | -300 LOC |
| 4. daemon-ensure | ❌ Delete | HIGH | Delete | -10 LOC |
| 5. heartbeat hive logic | ⚠️ Cleanup | HIGH | Delete | -400 LOC |
| 6. hive-core | ❌ Delete | HIGH | Delete | -200 LOC |
| 7. SseBroadcaster naming | ⚠️ Rename | LOW | Rename | 0 (refactor) |

**Total LOC Reduction:** ~910 lines of dead code removed  
**Total LOC Addition:** ~200 lines for queen lifecycle

---

## Implementation Order

### Phase 1: Delete Dead Code (HIGH Priority)
1. Delete `bin/25_rbee_hive_crates/worker-registry`
2. Delete `bin/99_shared_crates/daemon-ensure`
3. Delete `bin/99_shared_crates/hive-core`

### Phase 2: Clean Heartbeat Crate (HIGH Priority)
1. Delete `src/hive.rs`, `src/hive_receiver.rs`, `src/queen_receiver.rs`
2. Remove `HiveHeartbeatPayload` from types
3. Update worker.rs to target queen
4. Update documentation

### Phase 3: Rename Registry (MEDIUM Priority)
1. Rename `hive-registry` → `worker-registry`
2. Rename `HiveRegistry` → `WorkerRegistry`
3. Update all imports
4. Update architecture docs

### Phase 4: Queen Lifecycle (HIGH Priority)
1. Implement queen install/uninstall
2. Add smart prompts for localhost optimization
3. Add queen rebuild command
4. Update rbee-keeper documentation

### Phase 5: Rename SseBroadcaster (LOW Priority)
1. Rename `SseBroadcaster` → `SseChannelRegistry`
2. Update internal usages
3. Update documentation

---

## Documentation Updates Needed

### Architecture Docs

1. **.arch/01_COMPONENTS_PART_2.md**
   - Add queen lifecycle section to rbee-keeper
   - Update registry naming (hive-registry → worker-registry)

2. **.arch/03_DATA_FLOW_PART_4.md**
   - Update heartbeat flow (remove hive aggregation)
   - Worker → Queen direct

3. **.arch/CHANGELOG.md**
   - Add cleanup phase entry

### README Files

1. **bin/15_queen_rbee_crates/worker-registry/README.md**
   - Update from hive-registry
   - Clarify worker-centric purpose

2. **bin/99_shared_crates/heartbeat/README.md**
   - Remove hive heartbeat references
   - Update architecture diagram

---

## Testing Strategy

### After Each Cleanup Phase

1. **Compilation:** Ensure all binaries compile
2. **Tests:** Run all unit tests
3. **Integration:** Verify BDD tests pass
4. **Documentation:** Update affected docs

### Specific Tests

- Worker heartbeat to queen (verify endpoint works)
- Queen worker registry (verify worker tracking)
- Hive operations without heartbeat (verify still works)

---

## Breaking Changes

### None Expected!

All cleanups are:
1. **Deleting unused code** (no external dependencies)
2. **Renaming internal structs** (private to crates)
3. **Adding new features** (queen lifecycle - net new)

### Migration for External Users

None needed - all changes are internal to llama-orch.

---

## Review Checklist

Before merging cleanup:

- [ ] All deprecated crates deleted
- [ ] Compilation successful
- [ ] All tests pass
- [ ] Architecture docs updated
- [ ] CHANGELOG.md updated
- [ ] No references to removed crates
- [ ] New queen lifecycle implemented
- [ ] Smart prompts working

---

**Status:** Ready for implementation  
**Next:** Assign to TEAM for cleanup execution
