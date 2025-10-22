# TEAM-130F: rbee-hive BINARY + CRATES PLAN

**Phase:** Phase 3 Implementation Planning  
**Date:** 2025-10-19  
**Team:** TEAM-130F  
**Status:** 📋 PLAN (Future Architecture)

---

## 🎯 MISSION

Define **PLANNED** architecture for rbee-hive after Phase 3 consolidation.

**Key Changes:**
- ✅ Remove CLI commands (daemon-only)
- ✅ Use shared crates (daemon-lifecycle, rbee-types, rbee-http-client)
- ✅ Move model-catalog from shared-crates to binary (NOT shared)
- ✅ Remove unused dependencies (hive-core, gpu-info, secrets-management)

---

## 📊 METRICS (PLANNED)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total LOC** | 4,184 | ~3,887 | **-297 LOC** |
| **Files** | 33 | 28 | -5 files |
| **CLI commands** | 5 | 1 (daemon only) | -4 commands |
| **Shared crate deps** | 9 | 6 | -3 deps |

**LOC Breakdown:**
- Remove CLI violations: -297 LOC (models.rs, status.rs, worker.rs)
- Refactor monitor.rs: -186 LOC (use daemon-lifecycle)
- Add worker_lifecycle.rs: +150 LOC (new)
- Add shared crate usage: +36 LOC
- **Net savings: -297 LOC**

---

## 📦 INTERNAL CRATES (Within Binary)

### 1. worker-lifecycle (~150 LOC) **NEW**
**Location:** `src/worker_lifecycle.rs`  
**Purpose:** Worker spawning via daemon-lifecycle  
**Why NOT shared:** Hive-specific (preflight, registry, health monitoring)

```rust
pub struct WorkerLifecycleManager {
    lifecycle: DaemonLifecycle,  // Uses shared crate
    registry: Arc<WorkerRegistry>,
    provisioner: Arc<ModelProvisioner>,
}
```

**Dependencies:** `daemon-lifecycle`, `rbee-types`, `rbee-http-client`

---

### 2. worker-registry (~250 LOC)
**Location:** `src/registry.rs`  
**Purpose:** In-memory worker registry (lifecycle context)  
**Why NOT shared:** Hive-specific (PID, restart counts, health checks)

```rust
pub struct WorkerInfo {
    // Core identity
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: rbee_types::WorkerState,  // Shared enum
    
    // Lifecycle-specific (NOT in queen-rbee's WorkerInfo)
    pub pid: Option<u32>,
    pub last_activity: SystemTime,
    pub failed_health_checks: u32,
    pub restart_count: u32,
    pub last_restart: Option<SystemTime>,
    pub last_heartbeat: Option<SystemTime>,
    pub slots_total: u32,
    pub slots_available: u32,
}
```

**CRITICAL:** Different from queen-rbee's WorkerInfo (see TEAM-130E corrections)

**Dependencies:** `rbee-types` (WorkerState enum only)

---

### 3. model-catalog (~300 LOC) **MOVED**
**Location:** `src/model_catalog/` (MOVED from shared-crates)  
**Purpose:** Track downloaded models on THIS hive (SQLite)  
**Why NOT shared:** Single-hive scope, local filesystem

**Action:** `mv bin/shared-crates/model-catalog/ bin/rbee-hive/src/model_catalog/`

**Dependencies:** `rusqlite`, `rbee-types`

---

### 4. model-provisioner (~565 LOC)
**Location:** `src/provisioner/`  
**Purpose:** Model downloading and provisioning  
**Why NOT shared:** Hive-specific (local catalog, download tracking)

---

### 5. health-monitor (~200 LOC)
**Location:** `src/monitor.rs` (REFACTORED from 386 LOC)  
**Purpose:** Worker health monitoring  
**Why NOT shared:** Hive-specific (uses hive's registry, restart policies)

**Dependencies:** `worker-registry`, `worker-lifecycle`, `rbee-http-client`

---

### 6. http-server (~900 LOC)
**Location:** `src/http/`  
**Purpose:** Axum HTTP server  
**Why NOT shared:** Hive-specific API

**Changes:**
- Use `rbee-http-client` for worker communication
- Use `rbee-types` for request/response types

---

### 7. download-tracker (~180 LOC)
**Location:** `src/download_tracker.rs`  
**Purpose:** Track model download progress  
**Why NOT shared:** Hive-specific

---

### 8. metrics (~176 LOC)
**Location:** `src/metrics.rs`  
**Purpose:** Prometheus metrics  
**Why NOT shared:** Hive-specific metrics

---

## 🔗 DEPENDENCIES (PLANNED)

```toml
[dependencies]
# Phase 3 NEW: Shared crates
daemon-lifecycle = { path = "../shared-crates/daemon-lifecycle" }
rbee-http-client = { path = "../shared-crates/rbee-http-client" }
rbee-types = { path = "../shared-crates/rbee-types" }

# Existing: Shared crates
auth-min = { path = "../shared-crates/auth-min" }
input-validation = { path = "../shared-crates/input-validation" }
audit-logging = { path = "../shared-crates/audit-logging" }
deadline-propagation = { path = "../shared-crates/deadline-propagation" }

# REMOVED: hive-core (unused, wrong types)
# REMOVED: model-catalog (moved to src/model_catalog/)
# REMOVED: gpu-info (use hardware-capabilities when created)
# REMOVED: secrets-management (unused)

# HTTP server
axum = { workspace = true }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors"] }

# Async runtime
tokio = { workspace = true, features = ["full"] }
futures = { workspace = true }

# Database (for model catalog)
rusqlite = { version = "0.32", features = ["bundled"] }
dirs = "5.0"

# Serialization
serde = { workspace = true, features = ["derive"] }
serde_json = "1.0"

# CLI (daemon mode only)
clap = { version = "4.5", features = ["derive"] }

# Error handling
anyhow = "1.0"
thiserror = { workspace = true }

# Logging
tracing = { workspace = true }
tracing-subscriber = { workspace = true }

# UUID generation
uuid = { workspace = true, features = ["v4"] }

# Time handling
chrono = { version = "0.4", features = ["serde"] }

# Process management
nix = { version = "0.27", features = ["signal", "process"] }
sysinfo = { workspace = true }

# Metrics
prometheus = "0.13"
lazy_static = "1.4"

# Utilities
rand = "0.8"
async-stream = "0.3"
hostname = "0.4"
```

---

## 📋 BINARY STRUCTURE (PLANNED)

```
bin/rbee-hive/
├─ src/
│  ├─ main.rs                    (~50 LOC) - Entry point (daemon only)
│  ├─ lib.rs                     (~30 LOC) - Library exports
│  ├─ cli.rs                     (~50 LOC) - SIMPLIFIED: daemon args only
│  ├─ worker_lifecycle.rs        (~150 LOC) - NEW: Worker lifecycle manager
│  ├─ registry.rs                (~250 LOC) - Worker registry (lifecycle context)
│  ├─ model_catalog/             (~300 LOC) - MOVED from shared-crates
│  │  ├─ mod.rs
│  │  ├─ catalog.rs
│  │  └─ types.rs
│  ├─ provisioner/               (~565 LOC) - Model provisioning
│  │  ├─ mod.rs
│  │  ├─ download.rs
│  │  └─ operations.rs
│  ├─ monitor.rs                 (~200 LOC) - REFACTORED: Health monitoring
│  ├─ download_tracker.rs        (~180 LOC) - Download progress
│  ├─ http/
│  │  ├─ mod.rs                  (20 LOC)
│  │  ├─ server.rs               (~150 LOC) - Axum server
│  │  ├─ routes.rs               (~80 LOC) - Route definitions
│  │  ├─ health.rs               (~40 LOC) - Health endpoint
│  │  ├─ heartbeat.rs            (~60 LOC) - Worker heartbeat
│  │  ├─ workers.rs              (~200 LOC) - Worker endpoints
│  │  ├─ models.rs               (~150 LOC) - Model endpoints
│  │  ├─ metrics.rs              (~100 LOC) - Prometheus metrics
│  │  └─ middleware/
│  │     ├─ mod.rs               (5 LOC)
│  │     └─ auth.rs              (~50 LOC) - Auth middleware
│  ├─ metrics.rs                 (~176 LOC) - Prometheus metrics definitions
│  ├─ restart.rs                 (~120 LOC) - Worker restart logic
│  ├─ shutdown.rs                (~90 LOC) - Graceful shutdown
│  ├─ timeout.rs                 (~60 LOC) - Timeout handling
│  └─ resources.rs               (~80 LOC) - Resource tracking
├─ Cargo.toml
└─ README.md
```

**Removed Files:**
- ❌ `commands/models.rs` (118 LOC) - CLI violation
- ❌ `commands/status.rs` (74 LOC) - CLI violation
- ❌ `commands/worker.rs` (105 LOC) - CLI violation
- ❌ `commands/detect.rs` (80 LOC) - Moved to HTTP endpoint
- ❌ `worker_provisioner.rs` (67 LOC) - Merged into provisioner/

---

## 🔧 IMPLEMENTATION PLAN

### Day 1: Remove Violations
1. Delete CLI command files (models.rs, status.rs, worker.rs, detect.rs)
2. Simplify cli.rs (daemon args only)
3. Move model-catalog from shared-crates to src/model_catalog/
4. Update Cargo.toml (remove hive-core, gpu-info, secrets-management)

### Day 2: Integrate Shared Crates
1. Add daemon-lifecycle, rbee-http-client, rbee-types dependencies
2. Create worker_lifecycle.rs (uses daemon-lifecycle)
3. Refactor monitor.rs (use worker_lifecycle + rbee-http-client)
4. Update HTTP endpoints (use rbee-types)

### Day 3: Testing
1. Unit tests
2. Integration tests (daemon mode, worker spawning)
3. Verify no CLI commands work

---

## ✅ ACCEPTANCE CRITERIA

1. ✅ No CLI commands (daemon mode only)
2. ✅ model-catalog moved to `src/model_catalog/` (not shared)
3. ✅ Uses `daemon-lifecycle` for worker spawning
4. ✅ Uses `rbee-http-client` for worker communication
5. ✅ Uses `rbee-types` for shared types (WorkerState only)
6. ✅ WorkerInfo stays local (lifecycle context, NOT shared with queen-rbee)
7. ✅ hive-core, gpu-info, secrets-management removed
8. ✅ All tests pass
9. ✅ Binary compiles without warnings

---

## 📝 CRITICAL NOTES

### WorkerInfo is NOT Shared!
- rbee-hive: Lifecycle context (pid, restart_count, heartbeat)
- queen-rbee: Routing context (node_name, slots_available)
- **Only share WorkerState enum**

### model-catalog is NOT Shared!
- Only rbee-hive uses it
- Tracks models on THIS hive
- Should be in `src/model_catalog/`

---

**Status:** 📋 PLAN COMPLETE  
**LOC Impact:** -297 LOC (4,184 → 3,887)  
**Critical Removals:** CLI commands, unused shared crates
