# Complete Component Map - ALL Components
# Created by: TEAM-077
# Date: 2025-10-11
# Updated: 2025-10-11 14:24 - Clarified M0-M1 scope, M2+ features documented but deferred
# Sources: .business/stakeholders/, bin/.specs/, user feedback, milestone analysis

## Components Discovered

### Registries & Catalogs (6 components!)

#### 1. queen-rbee's Beehive Registry (SQLite) ✅
**File:** `010-ssh-registry-management.feature` (EXISTS)
**Storage:** SQLite at `~/.rbee/beehives.db`
**Purpose:** SSH connection details for rbee-hive nodes
**Schema:**
```sql
CREATE TABLE beehives (
    node_name TEXT PRIMARY KEY,
    ssh_host TEXT NOT NULL,
    ssh_user TEXT,
    ssh_key_path TEXT,
    capabilities JSON,  -- backends, gpus, ram
    last_connected_unix INTEGER
);
```

#### 2. queen-rbee's Worker Registry (In-Memory) ✅ M1!
**File:** `030-queen-rbee-worker-registry.feature` (M1 - Basic orchestration)
**Milestone:** M1 (Pool Manager Lifecycle)
**Storage:** In-memory (ephemeral)
**Purpose:** Global view of ALL workers across ALL rbee-hive instances - just HTTP endpoints!
**Status:** FOUNDATIONAL M1 feature - basic worker registry, NOT Rhai complexity
**Schema:**
```rust
struct WorkerRegistry {
    workers: HashMap<WorkerId, WorkerInfo>,
}

struct WorkerInfo {
    worker_id: String,
    rbee_hive_node: String,  // Which rbee-hive spawned it
    url: String,
    model_ref: String,
    backend: String,
    device: u32,
    capability: String,  // text-gen, image-gen, etc.
    state: WorkerState,  // loading, idle, busy
    last_heartbeat: Instant,
}
```

#### 3. rbee-hive's Worker Registry (In-Memory) ✅
**File:** `040-rbee-hive-worker-registry.feature` (RENAME from 060)
**Storage:** In-memory (ephemeral, lost on rbee-hive restart)
**Purpose:** Local worker lifecycle management on ONE rbee-hive
**Schema:**
```rust
struct WorkerRegistry {
    workers: HashMap<WorkerId, LocalWorkerInfo>,
}

struct LocalWorkerInfo {
    worker_id: String,
    pid: u32,
    port: u16,
    model_ref: String,
    backend: String,
    device: u32,
    state: WorkerState,
    spawned_at: SystemTime,
}
```

#### 4. Model Catalog (SQLite) ✅
**File:** `020-model-catalog.feature` (RENAME from model-provisioning)
**Storage:** SQLite at `~/.rbee/models.db`
**Purpose:** Model storage, download tracking, GGUF metadata
**Schema:**
```sql
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,  -- hf, file
    reference TEXT NOT NULL,
    local_path TEXT NOT NULL,
    size_bytes INTEGER,
    downloaded_at_unix INTEGER,
    gguf_metadata JSON
);
```

#### 5. Worker Provisioning + Binaries Catalog ⚠️ NEW FOR M1!
**File:** `025-worker-provisioning.feature` (NEW!)
**Milestone:** M1 (Pool Manager Lifecycle)
**Storage:** SQLite at `~/.rbee/worker_binaries.db` (or part of beehives.db)
**Purpose:** Build worker binaries from git + track which worker types are installed
**Schema:**
```sql
CREATE TABLE worker_binaries (
    id INTEGER PRIMARY KEY,
    rbee_hive_node TEXT NOT NULL,
    worker_type TEXT NOT NULL,  -- llm-worker-rbee, sd-worker-rbee, etc.
    version TEXT NOT NULL,
    binary_path TEXT NOT NULL,
    installed_at_unix INTEGER,
    UNIQUE(rbee_hive_node, worker_type)
);
```
**Why:** Each rbee-hive needs to know which workers it can spawn!

#### 6. Backend Catalog (Per rbee-hive) ✅
**File:** Part of `060-rbee-hive-preflight-checks.feature` (NEW!)
**Storage:** Detected dynamically or cached in `.runtime/backends.json`
**Purpose:** Track available backends (CUDA, Metal, CPU) on each rbee-hive
**Schema:**
```json
{
  "backends": [
    {"name": "cpu", "available": true, "priority": 3},
    {"name": "metal", "available": true, "priority": 1},
    {"name": "cuda", "available": false, "reason": "No NVIDIA GPU"}
  ]
}
```

### Preflight Checks (3 levels!)

#### 1. SSH Preflight Checks ⚠️ NEW FOR M1!
**File:** `050-ssh-preflight-checks.feature` (NEW!)
**Milestone:** M1 (Pool Manager Lifecycle)
**Component:** queen-rbee → rbee-hive SSH validation
**Purpose:** Validate SSH connectivity BEFORE spawning rbee-hive
**Checks:**
- SSH connection reachable
- SSH authentication works (key-based)
- SSH command execution works (`echo test`)
- Network latency acceptable (<100ms)
- Firewall allows connection
- rbee-hive binary exists on remote node

#### 2. rbee-hive Preflight Checks ⚠️ NEW FOR M1!
**File:** `060-rbee-hive-preflight-checks.feature` (NEW!)
**Milestone:** M1 (Pool Manager Lifecycle)
**Component:** rbee-hive readiness validation
**Purpose:** Validate rbee-hive is ready BEFORE spawning workers
**Checks:**
- rbee-hive HTTP API responding (GET /v1/health)
- rbee-hive version compatible with queen-rbee
- rbee-hive has required worker binaries (check worker_binaries catalog)
- rbee-hive has sufficient resources (RAM, disk)
- Backend catalog populated (CUDA/Metal/CPU detected)

#### 3. Worker Preflight Checks ✅
**File:** `070-worker-preflight-checks.feature` (RENAME from 030)
**Component:** Worker resource validation
**Purpose:** Validate resources BEFORE spawning worker
**Checks:**
- RAM available (model size + overhead)
- VRAM available (if GPU backend)
- Disk space available (for model download)
- Backend available (CUDA/Metal/CPU)
- Model exists in catalog

### Daemon Lifecycles (3 daemons!)

#### 1. worker-rbee Daemon Lifecycle ✅
**File:** `080-worker-rbee-lifecycle.feature` (RENAME from 040)
**Component:** worker-rbee daemon (llm-worker-rbee, sd-worker-rbee, etc.)
**Purpose:** Worker startup, registration, health monitoring
**Operations:**
- Spawn worker process
- Load model into VRAM/memory
- Send ready callback to rbee-hive
- Health check endpoints
- Loading progress streaming
- Graceful shutdown

#### 2. rbee-hive Daemon Lifecycle ✅
**File:** `090-rbee-hive-lifecycle.feature` (RENAME from 070)
**Component:** rbee-hive daemon
**Purpose:** rbee-hive startup, worker management, shutdown
**Operations:**
- Start HTTP daemon (port 9200)
- Monitor worker health (30s heartbeat)
- Enforce idle timeout (5 minutes)
- Cascading shutdown to workers
- Force-kill unresponsive workers

#### 3. queen-rbee Daemon Lifecycle ✅ M1!
**File:** `100-queen-rbee-lifecycle.feature` (M1 - Basic orchestration)
**Milestone:** M1 (Pool Manager Lifecycle)
**Component:** queen-rbee daemon
**Purpose:** Standard daemon lifecycle - industry standard stuff!
**Operations:**
- Ephemeral mode (rbee-keeper spawns queen-rbee)
- Persistent mode (queen-rbee pre-started)
- Cascading shutdown to all rbee-hive instances via SSH
- rbee-keeper exits after inference (CLI dies, daemons live)
**Status:** FOUNDATIONAL M1 feature - standard daemon lifecycle, NOT Rhai complexity

### Security Components (From SECURITY_ARCHITECTURE.md) - M3 FEATURES

#### 1. Authentication (auth-min) ⚠️ M3 FEATURE - DEFER!
**File:** `150-authentication.feature` (M3 - Documented but deferred)
**Milestone:** M3 (Security & Platform Readiness)
**Component:** auth-min crate
**Purpose:** Timing-safe token validation, bearer token parsing
**Features:**
- Timing-safe comparison (CWE-208 prevention)
- Token fingerprinting (SHA-256 → first 6 hex chars)
- Bind policy enforcement (no public bind without auth)
- RFC 6750 bearer token parsing

#### 2. Audit Logging (audit-logging) ⚠️ M3 FEATURE - DEFER!
**File:** `160-audit-logging.feature` (M3 - Documented but deferred)
**Milestone:** M3 (Security & Platform Readiness)
**Component:** audit-logging crate
**Purpose:** Immutable audit trail for compliance
**Features:**
- 32 event types (auth, resource ops, task lifecycle, VRAM ops, security incidents, compliance)
- Append-only storage with hash chains
- 7-year retention (GDPR requirement)
- Tamper detection (blockchain-style)

#### 3. Input Validation (input-validation) ⚠️ M3 FEATURE - DEFER!
**File:** `170-input-validation.feature` (M3 - Documented but deferred)
**Milestone:** M3 (Security & Platform Readiness)
**Component:** input-validation crate
**Purpose:** Injection attack prevention
**Features:**
- Identifier validation (no path traversal)
- Model reference validation (no command injection)
- Prompt validation (no VRAM exhaustion)
- Path validation (no directory traversal)
- String sanitization (no log injection)

#### 4. Secrets Management (secrets-management) ⚠️ M3 FEATURE - DEFER!
**File:** `180-secrets-management.feature` (M3 - Documented but deferred)
**Milestone:** M3 (Security & Platform Readiness)
**Component:** secrets-management crate
**Purpose:** Secure credential handling
**Features:**
- File-based loading (not environment variables)
- Memory zeroization (prevents memory dumps)
- Permission validation (rejects world-readable files)
- Timing-safe verification (constant-time comparison)
- Systemd credentials support

#### 5. Deadline Propagation (deadline-propagation) ⚠️ M3 FEATURE - DEFER!
**File:** `190-deadline-propagation.feature` (M3 - Documented but deferred)
**Milestone:** M3 (Security & Platform Readiness)
**Component:** deadline-propagation crate
**Purpose:** Resource exhaustion prevention
**Features:**
- Deadline propagation (client → orchestrator → pool → worker)
- Remaining time calculation at every hop
- Deadline enforcement (abort if insufficient time)
- 504 Gateway Timeout responses

### Scheduler & Orchestration (From RHAI_PROGRAMMABLE_SCHEDULER.md) - M2 FEATURES

#### 1. Rhai Programmable Scheduler ⚠️ M2 FEATURE - DEFER!
**File:** `200-rhai-scheduler.feature` (M2 - Documented but deferred)
**Milestone:** M2 (Orchestrator Scheduling)
**Component:** Rhai scripting engine in queen-rbee
**Purpose:** User-programmable routing logic
**Features:**
- Platform mode (immutable scheduler)
- Home/Lab mode (custom Rhai scripts)
- 40+ helper functions (worker selection, GPU queries, quota checks)
- YAML support (compiles to Rhai)
- Web UI policy builder

#### 2. Queue Management ⚠️ M2 FEATURE - DEFER!
**File:** `210-queue-management.feature` (M2 - Documented but deferred)
**Milestone:** M2 (Orchestrator Scheduling)
**Component:** queen-rbee admission control
**Purpose:** Job queue and priority management
**Features:**
- Priority classes (interactive, batch)
- Queue admission
- Quota enforcement (platform mode)
- Capacity management (reject with 429 when full)

### Execution & Operations

#### 1. Inference Execution ✅
**File:** `110-inference-execution.feature` (RENAME from 050)
**Component:** Inference request handling
**Purpose:** Token streaming, cancellation, error handling

#### 2. Input Validation ✅
**File:** `120-input-validation.feature` (RENAME from 090)
**Component:** CLI input validation
**Purpose:** Validate user inputs and authentication

#### 3. CLI Commands ✅
**File:** `130-cli-commands.feature` (RENAME from 100)
**Component:** rbee-keeper CLI
**Purpose:** Installation, configuration, management commands

#### 4. End-to-End Flows ✅
**File:** `140-end-to-end-flows.feature` (RENAME from 110)
**Component:** Integration tests
**Purpose:** Complete workflows from start to finish

## Summary

### Existing Files for M0-M1 (10 files - 1 deferred to M2)
- 010-ssh-registry-management.feature ✅ M1
- 020-model-catalog.feature ✅ M1 (rename from model-provisioning)
- 030-worker-preflight-checks.feature ✅ M1 (rename to 070)
- 040-worker-rbee-lifecycle.feature ✅ M0-M1 (rename to 080)
- 050-inference-execution.feature ✅ M0-M1 (rename to 110)
- 060-rbee-hive-management.feature ✅ M1 (rename to 040)
- 070-rbee-hive-lifecycle.feature ✅ M1 (rename to 090)
- 080-queen-rbee-lifecycle.feature ⚠️ M2 (defer, document for future)
- 090-input-validation.feature ✅ M1 (rename to 120)
- 100-cli-commands.feature ✅ M1 (rename to 130)
- 110-end-to-end-flows.feature ✅ M1 (rename to 140)

### New Files for M0-M1 (4 files!)
- 025-worker-provisioning.feature ⚠️ NEW! M1 (Build workers + catalog)
- 050-ssh-preflight-validation.feature ⚠️ NEW! M1 (SSH checks)
- 060-rbee-hive-preflight-validation.feature ⚠️ NEW! M1 (rbee-hive readiness)
- (070-worker-resource-preflight.feature already exists, rename from 030)

### M2+ Files (Documented but Deferred - 7 files)
- ~~030-queen-rbee-worker-registry.feature~~ ✅ MOVED TO M1
- ~~100-queen-rbee-lifecycle.feature~~ ✅ MOVED TO M1
- 150-authentication.feature ⚠️ M3 (security)
- 160-audit-logging.feature ⚠️ M3 (security)
- 170-input-validation.feature ⚠️ M3 (security)
- 180-secrets-management.feature ⚠️ M3 (security)
- 190-deadline-propagation.feature ⚠️ M3 (security)
- 200-rhai-scheduler.feature ⚠️ M2 (orchestrator)
- 210-queue-management.feature ⚠️ M2 (orchestrator)

### Total: 14 feature files for M1 + 7 for M2+

## Milestone Summary

- **M0 (v0.1.0):** Worker standalone - 2 files
- **M1 (v0.2.0):** Basic orchestrator + pool manager + worker - 14 files (includes queen-rbee!)
- **M2 (v0.3.0):** Rhai programmable scheduler - +2 files (ONLY Rhai complexity)
- **M3 (v0.4.0):** Security & platform - +5 files (auth, audit, secrets, etc.)

## Naming Conventions

### Current Naming (CORRECT)
- **queen-rbee** - Orchestrator daemon (not "orchestratord")
- **rbee-hive** - Pool manager daemon (not "pool-managerd")
- **worker-rbee** - Worker daemon (llm-worker-rbee, sd-worker-rbee, etc.)
- **rbee-keeper** - CLI/UI tool

### Old Naming (OUTDATED in specs)
- ❌ orchestratord → ✅ queen-rbee
- ❌ pool-managerd → ✅ rbee-hive
- ❌ worker-orcd → ✅ worker-rbee (llm-worker-rbee, etc.)

## Next Steps for M0-M1

1. **Renumber existing 10 files** to make room for new M1 components
2. **Create 4 new M1 feature files** (025, 050, 060, and rename 030→070)
3. **Separate preflight** - 3 distinct validation levels (SSH, rbee-hive, worker)
4. **Document but defer M2+ files** (keep 100-queen-rbee-lifecycle for future)
5. **Extract scenarios** from test-001.feature where applicable
6. **Write new scenarios** for M1 components (worker provisioning, preflight)
7. **Verify** all scenarios accounted for
8. **Compile** and test

## M2+ Features (Documented for Future)

- **M2 Rhai Scheduler:** ONLY Rhai programmable scheduler + queue management (2 files)
  - 200-rhai-scheduler.feature - Custom Rhai scripting
  - 210-queue-management.feature - Priority queues
- **M3 Security:** auth-min, audit-logging, input-validation, secrets-management, deadline-propagation (5 files)

**NOTE:** queen-rbee worker registry and lifecycle are M1, NOT M2!

This is the COMPLETE picture of rbee's components across all milestones!
