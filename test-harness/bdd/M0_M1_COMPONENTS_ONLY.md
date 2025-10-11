# M0-M1 Components ONLY
# Created by: TEAM-077
# Date: 2025-10-11
# Focus: Components needed NOW (not M2+ security/scheduler)

## Milestone Breakdown

### M0: Worker Haiku Test (v0.1.0) - IN PROGRESS
**Goal:** Worker binary runs standalone, passes haiku test
**Components:** worker-rbee only

### M1: Pool Manager Lifecycle (v0.2.0) - NEXT
**Goal:** rbee-hive can start/stop workers, hot-load models, report state
**Components:** rbee-hive + worker-rbee

### M2: Orchestrator Scheduling (v0.3.0) - FUTURE
**Goal:** queen-rbee with Rhai scheduler
**Components:** queen-rbee + Rhai scheduler ← **NOT IN SCOPE YET**

### M3: Security & Platform (v0.4.0) - FUTURE
**Goal:** auth, audit logging, multi-tenancy
**Components:** auth-min, audit-logging, secrets-management, etc. ← **NOT IN SCOPE YET**

## M0-M1 Components (What We Need NOW)

### Registries & Catalogs (5 components)

#### 1. queen-rbee's Beehive Registry (SQLite) ✅
**File:** `010-ssh-registry-management.feature` (EXISTS)
**Milestone:** M1 (queen-rbee needs to know which rbee-hive nodes exist)
**Storage:** SQLite at `~/.rbee/beehives.db`
**Purpose:** SSH connection details for rbee-hive nodes

#### 2. queen-rbee's Worker Registry (In-Memory) ✅
**File:** `030-queen-rbee-worker-registry.feature` (NEW!)
**Milestone:** M2 (queen-rbee orchestration) ← **DEFER TO M2**
**Storage:** In-memory
**Purpose:** Global view of ALL workers across ALL rbee-hive instances
**Decision:** **SKIP FOR NOW** - This is M2 orchestrator feature

#### 3. rbee-hive's Worker Registry (In-Memory) ✅
**File:** `040-rbee-hive-worker-registry.feature` (RENAME from 060)
**Milestone:** M1 (rbee-hive needs to track its workers)
**Storage:** In-memory (ephemeral)
**Purpose:** Local worker lifecycle management on ONE rbee-hive

**File:** `020-model-catalog.feature` (RENAME from model-provisioning)
**Milestone:** M1 (rbee-hive needs to track downloaded models)
**Storage:** SQLite at `~/.rbee/models.db`
**Purpose:** Model storage, download tracking, GGUF metadata

#### 5. Worker Provisioning + Binaries Catalog 
**File:** `025-worker-provisioning.feature` (NEW!)
**Milestone:** M1 (Pool Manager Lifecycle)
**Storage:** SQLite + git build system
**Purpose:** Build worker binaries from git + track which worker types installed
**Decision:** **NEEDED FOR M1** - rbee-hive must build and track workers

**File:** Part of `060-rbee-hive-preflight-checks.feature` (NEW!)
**Milestone:** M1 (rbee-hive needs to know CUDA/Metal/CPU availability)
**Storage:** Detected dynamically via NVML/Metal detection
{{ ... }}

### Preflight Validation (Consolidated) \u2705

**File:** `050-preflight-validation.feature` (NEW! - Consolidates 3 separate features)
**Milestone:** M1 (Pool Manager Lifecycle)
**Decision:** **NEEDED FOR M1** - ALL preflight checks in one logical feature

#### SSH Preflight Checks
**Component:** queen-rbee \u2192 rbee-hive SSH validation
**Checks:** Connection, auth, latency, binary exists

#### rbee-hive Preflight Checks
**Component:** rbee-hive readiness validation
**Checks:** HTTP API, version, backend catalog, resources

#### Worker Resource Preflight Checks
**Component:** Worker resource validation
**Checks:** RAM, VRAM (GPU FAIL FAST), disk, backend availability

### Daemon Lifecycles (3 daemons)

#### 1. worker-rbee Daemon Lifecycle ✅
**File:** `080-worker-rbee-lifecycle.feature` (RENAME from 040)
**Decision:** **NEEDED FOR M0-M1**

#### 2. rbee-hive Daemon Lifecycle ✅
**File:** `090-rbee-hive-lifecycle.feature` (RENAME from 070)
**Milestone:** M1 (rbee-hive daemon management)
**Decision:** **NEEDED FOR M1**

#### 3. queen-rbee Daemon Lifecycle ✅
**File:** `100-queen-rbee-lifecycle.feature` (RENAME from 080)
**Milestone:** M2 (orchestrator lifecycle) ← **DEFER TO M2**
**Decision:** **SKIP FOR NOW** - This is M2 orchestrator feature

### Execution & Operations

#### 1. Inference Execution ✅
**File:** `110-inference-execution.feature` (RENAME from 050)
**Milestone:** M0 (worker standalone) + M1 (via rbee-hive)
**Decision:** **NEEDED FOR M0-M1**

#### 2. Input Validation ✅
**File:** `120-input-validation.feature` (RENAME from 090)
**Milestone:** M1 (rbee-keeper CLI validation)
**Decision:** **NEEDED FOR M1**

#### 3. CLI Commands ✅
**File:** `130-cli-commands.feature` (RENAME from 100)
**Milestone:** M1 (rbee-keeper manages rbee-hive)
**Decision:** **NEEDED FOR M1**

#### 4. End-to-End Flows ✅
**File:** `140-end-to-end-flows.feature` (RENAME from 110)
**Milestone:** M1 (full stack integration)
**Decision:** **NEEDED FOR M1**

## Components to SKIP (M2+)

### M2 Features (Orchestrator Scheduling)
- ❌ 030-queen-rbee-worker-registry.feature - M2 orchestrator
- ❌ 100-queen-rbee-lifecycle.feature - M2 orchestrator
- ❌ 200-rhai-scheduler.feature - M2 Rhai scripting
- ❌ 210-queue-management.feature - M2 admission control

### M3 Features (Security & Platform)

### Existing Files (11 - need renumbering)
- 010-ssh-registry-management.feature ✅ M1
- 020-model-catalog.feature ✅ M1 (rename from model-provisioning)
- 030-worker-preflight-checks.feature ✅ M1 (rename to 070)
- 050-inference-execution.feature ✅ M0-M1 (rename to 110)
- 060-rbee-hive-management.feature ✅ M1 (rename to 040, was "pool-management")
- 080-queen-rbee-lifecycle.feature ❌ M2 (DELETE or defer)
- 090-input-validation.feature ✅ M1 (rename to 120)
- 100-cli-commands.feature ✅ M1 (rename to 130)
- 110-end-to-end-flows.feature ✅ M1 (rename to 140)

### New Files Needed for M0-M1 (2 files!)
- 025-worker-provisioning.feature ⚠️ NEW! M1 (Build workers + catalog)
- 050-preflight-validation.feature ⚠️ NEW! M1 (Consolidated preflight)

### Total: 12 feature files for M0-M1

## Action Plan

1. **Delete or defer M2+ file:**
   - Delete `080-queen-rbee-lifecycle.feature` (M2Orchestrator)

2. **Renumber existing files:**
   - Keep 010, 020 as-is
   - Create 025 (new)
   - Rename 060  040 (rbee-hive-worker-registry)
   - Create 050, 060 (new preflight files)
   - Rename 030  070 (worker-preflight)
   - Rename 040  080 (worker-rbee-lifecycle)
   - Rename 070  090 (rbee-hive-lifecycle)
   - Rename 050  110 (inference-execution)
   - Rename 090  120 (input-validation)
   - Rename 100  130 (cli-commands)
   - Rename 110  140 (end-to-end-flows)

3. **Create 2 new feature files:**
   - 025-worker-provisioning.feature
   - 050-preflight-validation.feature

4. **Verify compilation and scenario counts**

## Key Insights

1. **M0 = Worker only** - Just worker-rbee standalone
2. **M1 = rbee-hive + worker** - Pool manager Spawns workers
2. **M1 = rbee-hive + worker** - Pool manager spawns workers
3. **M2 = queen-rbee orchestration** - Smart scheduling with Rhai
4. **M3 = Security** - auth, audit, secrets

**We are currently in M0-M1, so skip all M2+ features!**

This keeps us focused on what's needed NOW.
