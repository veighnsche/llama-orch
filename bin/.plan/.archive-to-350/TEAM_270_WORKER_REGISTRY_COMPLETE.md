# TEAM-270 COMPLETE: Worker Registry with Direct Heartbeats

**Date:** Oct 23, 2025  
**Status:** ✅ COMPLETE  
**Mission:** Update worker-registry in queen to use new worker-contract types and direct heartbeats

---

## ✅ Summary

Successfully implemented DirectWorkerRegistry in queen-rbee that aligns with the CORRECTION_269_TO_272 architecture:
- ✅ Workers send heartbeats DIRECTLY to queen (not via hive)
- ✅ Uses new worker-contract types (WorkerInfo, WorkerHeartbeat)
- ✅ Thread-safe with RwLock
- ✅ Comprehensive API for worker tracking and scheduling

---

## 📁 Files Created/Modified

### Created
1. `bin/99_shared_crates/worker-contract/` - Worker contract types (TEAM-270)
2. `bin/15_queen_rbee_crates/worker-registry/src/direct_worker_registry.rs` - New registry (320 LOC)

### Modified
1. `bin/15_queen_rbee_crates/worker-registry/Cargo.toml` - Added worker-contract dependency
2. `bin/15_queen_rbee_crates/worker-registry/src/lib.rs` - Exported DirectWorkerRegistry, deprecated old WorkerRegistry

---

## 🏗️ Architecture

### OLD (Deprecated)
```
Worker → Hive → Queen (hive aggregates worker state)
```

### NEW (TEAM-270)
```
Worker → Queen (direct heartbeat via POST /v1/worker-heartbeat)
```

---

## 📊 DirectWorkerRegistry API

```rust
pub struct DirectWorkerRegistry {
    workers: RwLock<HashMap<String, WorkerHeartbeat>>,
}

// Core Methods
pub fn new() -> Self
pub fn update_worker(&self, heartbeat: WorkerHeartbeat)
pub fn get_worker(&self, worker_id: &str) -> Option<WorkerInfo>
pub fn remove_worker(&self, worker_id: &str) -> bool

// Query Methods
pub fn list_all_workers(&self) -> Vec<WorkerInfo>
pub fn list_online_workers(&self) -> Vec<WorkerInfo>
pub fn list_available_workers(&self) -> Vec<WorkerInfo>
pub fn find_workers_by_model(&self, model_id: &str) -> Vec<WorkerInfo>
pub fn find_best_worker_for_model(&self, model_id: &str) -> Option<WorkerInfo>

// Status Methods
pub fn total_worker_count(&self) -> usize
pub fn online_worker_count(&self) -> usize
pub fn is_worker_online(&self, worker_id: &str) -> bool
pub fn cleanup_stale_workers(&self) -> usize
```

---

## 🔑 Key Features

### 1. Direct Worker Tracking
- Workers send heartbeats directly to queen
- No hive aggregation layer
- Simpler architecture, clearer ownership

### 2. Automatic Staleness Detection
- Uses `WorkerHeartbeat::is_recent()` from worker-contract
- 90-second timeout (3 missed heartbeats)
- `cleanup_stale_workers()` removes expired workers

### 3. Smart Worker Selection
- `list_available_workers()` - Online + Ready status
- `find_workers_by_model()` - Filter by model
- `find_best_worker_for_model()` - Best match for scheduling

### 4. Thread-Safe
- RwLock for concurrent access
- Read-heavy optimization (scheduling queries)
- Write-light (heartbeat updates)

---

## 📝 Usage Example

```rust
use queen_rbee_worker_registry::DirectWorkerRegistry;
use worker_contract::{WorkerInfo, WorkerStatus, WorkerHeartbeat};

// Create registry
let registry = DirectWorkerRegistry::new();

// Worker sends heartbeat
let worker = WorkerInfo {
    id: "worker-123".to_string(),
    model_id: "meta-llama/Llama-2-7b".to_string(),
    device: "GPU-0".to_string(),
    port: 9301,
    status: WorkerStatus::Ready,
    implementation: "llm-worker-rbee".to_string(),
    version: "0.1.0".to_string(),
};

registry.update_worker(WorkerHeartbeat::new(worker));

// Query for scheduling
let available = registry.list_available_workers();
let best = registry.find_best_worker_for_model("meta-llama/Llama-2-7b");
```

---

## ✅ Verification

- ✅ **Compilation:** `cargo check -p queen-rbee-worker-registry` - PASS
- ✅ **Architecture:** Aligns with CORRECTION_269_TO_272
- ✅ **Types:** Uses worker-contract types
- ✅ **Documentation:** Comprehensive inline docs

---

## 🚀 Next Steps

### TEAM-271: Worker Spawn Operation
**Mission:** Implement WorkerSpawn in hive (stateless - just spawn and return)

**Key Points:**
- Hive spawns worker process
- Worker configured with `--queen-url` arg
- Worker sends heartbeat to queen (not hive)
- Hive returns spawn info (PID, port) and forgets about worker

### Integration Required
1. Add `POST /v1/worker-heartbeat` endpoint to queen-rbee
2. Wire up DirectWorkerRegistry in queen-rbee main.rs
3. Implement worker spawn in rbee-hive (TEAM-271)
4. Update llm-worker-rbee to send heartbeats to queen

---

## 📚 Related Documents

- `CORRECTION_269_TO_272_ARCHITECTURE_FIX.md` - Architecture corrections
- `TEAM_270_WORKER_CONTRACT_HANDOFF.md` - Worker contract implementation
- `TEAM_270_CRATE_REORGANIZATION.md` - Crate reorganization

---

**TEAM-270 COMPLETE**  
**Status:** ✅ DirectWorkerRegistry implemented and ready for integration
