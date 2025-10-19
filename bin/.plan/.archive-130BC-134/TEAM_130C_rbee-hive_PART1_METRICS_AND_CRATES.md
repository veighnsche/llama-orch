# TEAM-130B: rbee-hive - PART 1: METRICS & CRATES

**Binary:** `bin/rbee-hive`  
**Phase:** Phase 2, Day 5-6  
**Date:** 2025-10-19

---

## 🎯 EXECUTIVE SUMMARY

**Current:** Pool manager daemon (4,184 LOC code-only, 33 files)  
**Proposed:** 10 focused crates under `rbee-hive-crates/`  
**Risk:** LOW  
**Timeline:** 3 weeks

**Phase 1 Cross-Binary Corrections Applied:**
- ✅ audit-logging usage corrected (IS USED, 15× in 3 files)
- ✅ LOC methodology: 4,184 code-only (not 6,021)
- ✅ HTTP client duplication identified → rbee-http-client opportunity
- ✅ secrets-management waste → REMOVE

**Reference:** `TEAM_130B_CROSS_BINARY_ANALYSIS.md`

---

## 📊 GROUND TRUTH METRICS

```bash
$ cloc bin/rbee-hive/src --quiet
Files: 33 | Code: 4,184 | Comments: 1,092 | Blanks: 866
Total Lines: 6,142
```

**Largest Files:**
1. registry.rs - 492 LOC (worker state)
2. http/workers.rs - 407 LOC (endpoints)
3. provisioner.rs - 389 LOC (model download)
4. commands/daemon.rs - 380 LOC (startup)
5. monitor.rs - 207 LOC (health checks)

---

## 🏗️ 10 PROPOSED CRATES

| # | Crate | LOC | Purpose | Risk |
|---|-------|-----|---------|------|
| 1 | registry | 644 | Worker state management | Low |
| 2 | http-server | 878¹ | HTTP endpoints | Medium |
| 3 | http-middleware | 130 | Auth + audit² | Low |
| 4 | provisioner | 624 | Model download | Low |
| 5 | monitor | 210 | Health monitoring | Low |
| 6 | resources | 247 | Resource limits | Low |
| 7 | shutdown | 248 | Graceful shutdown | Medium |
| 8 | metrics | 222 | Prometheus | Low |
| 9 | restart | 162 | Restart policy | Low |
| 10 | cli | 719 | CLI commands² | Medium |

**Total:** 4,084 LOC in libraries + 100 LOC binary

**Corrections from Team 131:**
- ¹ http-server: Was 576 LOC (43% underestimate) → Actual 878 LOC
- ² audit-logging: Team 131 claimed "NOT USED" → Actually used in http-middleware (6×) and cli (7×)

---

## 📦 CRATE SPECIFICATIONS

### CRATE 1: rbee-hive-registry (644 LOC)

**Purpose:** In-memory worker registry  
**Files:** registry.rs (492) + types (152)

**API:**
```rust
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

pub enum WorkerState { Spawning, Ready, Healthy, Unhealthy, Dead }

impl WorkerRegistry {
    pub fn register(&self, info: WorkerInfo) -> Result<()>;
    pub fn update_state(&self, worker_id: &str, state: WorkerState);
    pub fn list(&self) -> Vec<WorkerInfo>;
}
```

**Dependencies:** tokio, serde  
**Cross-Binary:** Similar to queen-rbee worker registry (ephemeral vs persistent difference)

---

### CRATE 2: rbee-hive-http-server (878 LOC)

**Purpose:** HTTP endpoints and routing  
**Files:** http/{server,routes,workers,models,health,heartbeat,metrics}.rs

**API:**
```rust
pub struct HiveHttpServer {
    registry: Arc<WorkerRegistry>,
    provisioner: Arc<Provisioner>,
    monitor: Arc<Monitor>,
}

pub async fn spawn_worker(...) -> Result<Json<SpawnResponse>>;
pub async fn list_workers(...) -> Result<Json<Vec<WorkerInfo>>>;
```

**Dependencies:** axum, tower, serde_json, audit-logging (CORRECTED)  
**Cross-Binary:** Worker spawn contract → `rbee-http-types` shared crate opportunity

---

### CRATE 3: rbee-hive-http-middleware (130 LOC)

**Purpose:** Authentication + audit logging  
**Files:** http/middleware/auth.rs

**API:**
```rust
pub async fn auth_middleware(...) -> Result<Response, StatusCode>;
```

**Critical Correction:**
```rust
// Team 131 missed this - 6 audit events in auth.rs:
logger.emit(audit_logging::AuditEvent::AuthFailure { ... }); // line 49
logger.emit(audit_logging::AuditEvent::AuthSuccess { ... }); // line 109
```

**Dependencies:** axum, auth-min, audit-logging (CORRECTED)  
**Cross-Binary:** Pattern matches queen-rbee (excellent) ✅

---

### CRATE 4: rbee-hive-provisioner (624 LOC)

**Purpose:** Model download and caching  
**Files:** provisioner.rs (389) + models_catalog.rs (84) + gpu_info.rs (95) + support (56)

**API:**
```rust
pub async fn provision_model(&self, model_ref: &str) -> Result<PathBuf>;
pub fn is_cached(&self, model_ref: &str) -> bool;
```

**Dependencies:** tokio, reqwest, model-catalog, gpu-info  
**Cross-Binary:** Used by llm-worker (model loading)

---

### CRATE 5: rbee-hive-monitor (210 LOC)

**Purpose:** Worker health monitoring  
**API:** Periodic polling of worker health endpoints, updates registry state

**Dependencies:** tokio, reqwest  
**Cross-Binary:** Different from llm-worker heartbeat (pull vs push)

---

### CRATE 6: rbee-hive-resources (247 LOC)

**Purpose:** Resource limit enforcement  
**API:** Check GPU VRAM, validate model fits, enforce spawn limits

**Dependencies:** gpu-info

---

### CRATE 7: rbee-hive-shutdown (248 LOC)

**Purpose:** Graceful shutdown coordination  
**API:** SIGTERM/SIGINT handling, coordinate worker cleanup, timeout enforcement

**Dependencies:** tokio  
**Cross-Binary:** Similar to queen-rbee shutdown pattern

---

### CRATE 8: rbee-hive-metrics (222 LOC)

**Purpose:** Prometheus metrics export  
**API:** Collect worker counts, cache hit rate, expose /metrics

**Dependencies:** prometheus

---

### CRATE 9: rbee-hive-restart (162 LOC)

**Purpose:** Worker restart policy  
**API:** Restart policies (Always, OnFailure, Never), exponential backoff

**Dependencies:** tokio

---

### CRATE 10: rbee-hive-cli (719 LOC)

**Purpose:** CLI command implementations  
**Files:** commands/{daemon,models,workers,status}.rs + cli.rs

**Critical Correction:**
```rust
// Team 131 missed this - 7 audit usages in daemon.rs:
let audit_config = audit_logging::AuditConfig { ... }; // line 84
let audit_logger = audit_logging::AuditLogger::new(...); // line 102
```

**Dependencies:** clap, colored, audit-logging (CORRECTED), ALL other crates  
**Cross-Binary:** Similar structure to rbee-keeper CLI

---

## 📊 DEPENDENCY GRAPH

```
Layer 0 (Standalone):
- restart (162 LOC)
- metrics (222 LOC)  
- resources (247 LOC)
- http-middleware (130 LOC)

Layer 1 (Core):
- registry (644 LOC)
- provisioner (624 LOC)

Layer 2 (Business Logic):
- monitor (210 LOC) → uses registry, restart
- shutdown (248 LOC) → uses registry

Layer 3 (HTTP):
- http-server (878 LOC) → uses registry, provisioner, monitor, http-middleware, metrics

Layer 4 (CLI):
- cli (719 LOC) → uses ALL

Binary (100 LOC) → uses cli
```

**No circular dependencies ✅**

---

## 🔗 CROSS-BINARY CONTEXT

**Shared Patterns:**
- Worker registry concept (rbee-hive persistent, queen-rbee ephemeral)
- Auth middleware with audit events (matches queen-rbee pattern ✅)
- Graceful shutdown coordination (similar to queen-rbee)
- CLI structure (similar to rbee-keeper)

**Integration Points:**
- Receives worker ready callbacks from queen-rbee
- Spawns llm-worker processes via tokio::process::Command
- HTTP client for worker health checks

**Opportunities (from Phase 1):**
1. Create `rbee-http-client` shared crate (eliminate duplicate reqwest patterns)
2. Create `rbee-http-types` for worker spawn contracts
3. Add `narration-core` (~30-50 narration points)
4. Remove unused `secrets-management` dependency

**Critical Fixes Applied:**
- audit-logging dependencies added (http-middleware, cli)
- LOC counts corrected (http-server 878 not 576)
- All shared crate usages verified

---

**Status:** Part 1 Complete - Metrics & Crate Design Established  
**Next:** Part 2 (Phase 3) - External Library Analysis  
**Reference:** TEAM_130B_CROSS_BINARY_ANALYSIS.md for full context
