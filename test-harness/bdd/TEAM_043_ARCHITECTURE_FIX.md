# TEAM-043 Architecture Fix: Dual Registry System

**Date:** 2025-10-10  
**Issue:** Single registry.rs → Should be dual registry system  
**Status:** ✅ FIXED

---

## Problem Identified

User correctly identified that queen-rbee needs **TWO separate registries**, not one:

1. **Beehive Registry** (SQLite) - Persistent registry of remote rbee-hive nodes
2. **Worker Registry** (In-Memory) - Ephemeral registry of active workers

Initial implementation only had a single `registry.rs` which was incorrect.

---

## Changes Made

### 1. Renamed File
```bash
mv src/registry.rs → src/beehive_registry.rs
```

### 2. Created Worker Registry
**New file:** `src/worker_registry.rs`
- In-memory `HashMap<String, WorkerInfo>` with `RwLock`
- CRUD operations: register, update_state, get, list, remove
- Unit tests included

### 3. Updated main.rs
```rust
// Before:
let registry = registry::BeehiveRegistry::new(db_path).await?;

// After:
let beehive_registry = beehive_registry::BeehiveRegistry::new(db_path).await?;
let worker_registry = worker_registry::WorkerRegistry::new();
```

### 4. Updated http.rs AppState
```rust
// Before:
pub struct AppState {
    pub registry: Arc<BeehiveRegistry>,
}

// After:
pub struct AppState {
    pub beehive_registry: Arc<BeehiveRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
}
```

### 5. Updated All HTTP Handlers
- `add_node()` → uses `state.beehive_registry`
- `list_nodes()` → uses `state.beehive_registry`
- `remove_node()` → uses `state.beehive_registry`

---

## Architecture Clarification

### Beehive Registry (SQLite)
**Purpose:** Track infrastructure nodes (rbee-hive instances)

**Persistence:** SQLite at `~/.rbee/beehives.db`

**Data:**
- node_name
- ssh_host, ssh_port, ssh_user, ssh_key_path
- git_repo_url, git_branch, install_path
- last_connected_unix, status

**Lifecycle:** Persistent across restarts

### Worker Registry (In-Memory)
**Purpose:** Track active workers spawned by rbee-hive

**Persistence:** In-memory only (ephemeral)

**Data:**
- worker_id, url
- model_ref, backend, device
- state (Loading/Idle/Busy)
- slots_total, slots_available
- vram_bytes

**Lifecycle:** Cleared on restart

---

## Why This Matters

### Separation of Concerns
- **Beehive nodes** = Infrastructure (persistent)
- **Workers** = Runtime processes (ephemeral)

### Different Lifecycles
- **Beehive nodes** survive restarts (need SSH creds, config)
- **Workers** are transient (re-discovered on startup)

### Different Storage Needs
- **Beehive nodes** need persistence (SQLite)
- **Workers** need speed (in-memory HashMap)

---

## Build Status

✅ Compiles successfully:
```bash
cargo build --bin queen-rbee
# Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.09s
```

Warnings are expected (unused methods will be used when worker endpoints are added).

---

## Files Modified

1. `bin/queen-rbee/src/registry.rs` → **RENAMED** → `beehive_registry.rs`
2. `bin/queen-rbee/src/worker_registry.rs` → **CREATED**
3. `bin/queen-rbee/src/main.rs` → **MODIFIED** (dual registry init)
4. `bin/queen-rbee/src/http.rs` → **MODIFIED** (AppState + handlers)
5. `bin/queen-rbee/ARCHITECTURE.md` → **CREATED** (documentation)

---

## Next Steps for TEAM-044

### Worker Registry Endpoints (TODO)
Add HTTP endpoints for worker management:

```rust
// In http.rs, add these routes:
.route("/v1/workers/register", post(register_worker))
.route("/v1/workers/list", get(list_workers))
.route("/v1/workers/update", post(update_worker))
.route("/v1/workers/{id}", delete(remove_worker))
```

### Worker Ready Callback
When llm-worker-rbee starts, it should call:
```
POST /v1/workers/register
{
  "worker_id": "worker-123",
  "url": "http://localhost:8081",
  "model_ref": "tinyllama",
  "vram_bytes": 8000000000
}
```

---

## Testing Impact

### BDD Tests
No changes needed to BDD tests - they only test beehive registry endpoints which still work the same way.

### Future Tests
When worker endpoints are added, create new BDD scenarios:
```gherkin
Scenario: Worker registers with queen-rbee
  Given queen-rbee is running
  When a worker sends ready callback
  Then queen-rbee adds worker to in-memory registry
  And the worker appears in /v1/workers/list
```

---

## Lessons Learned

✅ **Architecture review is critical** - User caught the single-registry mistake  
✅ **Dual registry pattern** - Persistent + ephemeral is common in orchestrators  
✅ **Separation of concerns** - Infrastructure vs runtime processes  
✅ **Documentation helps** - ARCHITECTURE.md clarifies the design

---

**Status:** ✅ Architecture fixed, compiles successfully, ready for worker endpoint implementation
