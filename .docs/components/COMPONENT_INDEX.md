# rbee Component Index

**Purpose:** Central index of all rbee ecosystem components  
**Last Updated:** TEAM-096 | 2025-10-18

## Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    rbee Ecosystem                        │
└─────────────────────────────────────────────────────────┘

User Interface:
  rbee-keeper (CLI) ──► Controls everything

Orchestration Layer:
  queen-rbee (Daemon) ──► Manages hives across machines

Worker Pool Layer:
  rbee-hive (Daemon) ──► Manages workers on one machine

Worker Layer:
  llm-worker-rbee ──► Performs inference
  vision-worker-rbee (future)
  audio-worker-rbee (future)

Cross-Cutting:
  Local vs Network Mode ──► Deployment strategy
```

## Core Binaries

### 1. rbee-keeper (CLI Tool)
**File:** `RBEE_KEEPER.md`  
**Status:** ✅ IMPLEMENTED (TEAM-043, TEAM-085)

**Purpose:** User-facing CLI to control rbee ecosystem

**Key Responsibilities:**
- ✅ Start/stop queen-rbee (queen_lifecycle.rs - TEAM-085)
- ✅ Send inference requests (commands/infer.rs)
- ✅ Manage workers (commands/workers.rs)
- ✅ Setup nodes (commands/setup.rs - TEAM-043)
- ✅ SSH support (ssh.rs)
- ✅ Hive management (commands/hive.rs)
- ✅ Install/logs (commands/install.rs, logs.rs)

**Files:**
- `src/queen_lifecycle.rs` - Auto-start queen
- `src/commands/` - All CLI commands
- `src/ssh.rs` - SSH utilities

---

### 2. queen-rbee (Orchestrator Daemon)
**File:** `QUEEN_RBEE.md`  
**Status:** ✅ IMPLEMENTED (TEAM-043, TEAM-046, TEAM-052, TEAM-080)

**Purpose:** Central orchestrator managing hives across machines

**Key Responsibilities:**
- ✅ Beehive registry (SQLite) - beehive_registry.rs (TEAM-043)
- ✅ Worker registry (RAM) - worker_registry.rs (TEAM-043, TEAM-046)
- ✅ SSH support - ssh.rs (TEAM-043)
- ✅ HTTP API - http/ directory
  - beehives.rs - Hive management
  - workers.rs - Worker management
  - inference.rs - Request routing
  - health.rs - Health checks
- ✅ Backend capabilities tracking (TEAM-052)

**Files:**
- `src/beehive_registry.rs` - SQLite hive registry
- `src/worker_registry.rs` - RAM worker registry
- `src/ssh.rs` - SSH connection validation
- `src/http/` - Complete HTTP API

---

### 3. rbee-hive (Worker Pool Manager)
**File:** `RBEE_HIVE.md`  
**Status:** 🟡 FUNCTIONAL but lifecycle gaps

**Purpose:** Manages worker processes on single machine

**Key Responsibilities:**
- Worker registry (RAM) - tightly coupled with lifecycle
- Worker lifecycle management
- Model provisioner
- Model catalog (SQLite)
- Worker provisioner (future)

**Current Gaps:**
- No PID tracking (critical)
- No force kill capability
- No restart policy
- No heartbeat mechanism
- See `LIFECYCLE_MANAGEMENT_GAPS.md`

**Recent Fixes:**
- ✅ TEAM-096: Smart port allocation
- ✅ TEAM-096: Fail-fast protocol

---

## Subsystems & Components

### 4. llm-worker-rbee (Inference Worker)
**File:** `LLM_WORKER_RBEE.md` (TODO)  
**Status:** ✅ FUNCTIONAL

**Purpose:** Performs LLM inference

**Key Responsibilities:**
- Load GGUF models
- Run inference
- Stream results
- Report to hive

---

### 5. Local vs Network Mode
**File:** `LOCAL_VS_NETWORK_MODE.md`  
**Status:** ✅ IMPLEMENTED (TEAM-043, TEAM-085)

**Purpose:** Deployment mode distinction

**Modes:**
- ✅ **Local:** queen-rbee auto-start (queen_lifecycle.rs - TEAM-085)
- ✅ **Network:** SSH to remote hives (ssh.rs - TEAM-043)

**Implementation:**
- ✅ SSH connection validation (queen-rbee/src/ssh.rs)
- ✅ SSH utilities (rbee-keeper/src/ssh.rs)
- ✅ Remote command execution
- ✅ Beehive registry stores SSH credentials
- ✅ Auto-start queen for local mode (TEAM-085)

---

### 6. Model Provisioner (rbee-hive)
**File:** `MODEL_PROVISIONER.md`  
**Status:** ✅ IMPLEMENTED (TEAM-029, TEAM-034)

**Purpose:** Download GGUF models from HuggingFace with progress tracking

**Location:** `bin/rbee-hive/src/provisioner/`

**Key Features:**
- ✅ HuggingFace integration
- ✅ SSE progress streaming (TEAM-034)
- ✅ Catalog integration
- ✅ Concurrent downloads

---

### 7. Model Catalog (rbee-hive)
**File:** `MODEL_CATALOG.md`  
**Status:** ✅ IMPLEMENTED (TEAM-030)

**Purpose:** Persistent SQLite database tracking downloaded models for THIS hive

**Location:** `bin/shared-crates/model-catalog/`

**Key Features:**
- ✅ SQLite persistence (.rbee/models/catalog.db)
- ✅ Composite key (reference + provider)
- ✅ Size tracking
- ✅ Async/await via sqlx

---

### 8. Worker Provisioner (rbee-hive)
**Status:** 🔴 STUB ONLY

**Purpose:** Download or build worker binaries

**Current:** Assumes binary exists  
**Needed:** Download pre-built, build from git

---

### 9. Beehive Registry (queen-rbee)
**File:** `BEEHIVE_REGISTRY.md`  
**Status:** ✅ IMPLEMENTED (TEAM-043, TEAM-052)

**Purpose:** Persistent SQLite database tracking registered hives with SSH credentials and capabilities

**Location:** `bin/queen-rbee/src/beehive_registry.rs`

**Key Features:**
- ✅ SQLite persistence (~/.rbee/beehives.db)
- ✅ SSH credential storage
- ✅ Backend capabilities (TEAM-052)
- ✅ Connection status tracking
- ✅ Complete CRUD operations

---

### 10. Worker Registry (queen-rbee)
**File:** `WORKER_REGISTRY_QUEEN.md`  
**Status:** ✅ IMPLEMENTED (TEAM-043, TEAM-046, TEAM-080)

**Purpose:** In-memory registry tracking workers across ALL hives for request routing

**Location:** `bin/queen-rbee/src/worker_registry.rs`

**Key Features:**
- ✅ Thread-safe concurrent access (TEAM-080)
- ✅ Node mapping (TEAM-046)
- ✅ VRAM tracking
- ✅ State management
- ✅ Fast in-memory lookups

---

### 11. Worker Registry (rbee-hive)
**File:** `WORKER_REGISTRY_HIVE.md`  
**Status:** 🟡 IMPLEMENTED (with lifecycle gaps)

**Purpose:** In-memory registry tracking THIS hive's workers, tightly coupled with lifecycle

**Location:** `bin/rbee-hive/src/registry.rs`

**Key Features:**
- ✅ Thread-safe concurrent access
- ✅ Fail-fast counter (TEAM-096)
- ✅ Idle timeout support
- ❌ No PID tracking (critical gap)

---

### 12. Worker Lifecycle Management
**File:** `WORKER_LIFECYCLE.md`  
**Status:** 🟡 PARTIAL (critical gaps)

**Purpose:** Complete worker process lifecycle from spawn to termination

**Location:** `bin/rbee-hive/src/` (multiple files)

**Key Features:**
- ✅ Spawn with smart port allocation (TEAM-096)
- ✅ Health monitoring (30s, TEAM-027)
- ✅ Fail-fast removal (TEAM-096)
- ✅ Idle timeout (TEAM-027)
- ❌ No PID tracking (critical gap)
- ❌ No force kill (critical gap)

---

### 13. SSE Streaming
**File:** `SSE_STREAMING.md`  
**Status:** ✅ IMPLEMENTED (TEAM-034)

**Purpose:** Real-time streaming of download progress and inference results

**Location:** Multiple locations

**Key Features:**
- ✅ Download progress streaming
- ✅ Inference token streaming
- ✅ Multiple subscribers
- ✅ Keep-alive support

---

### 14. Cascading Shutdown
**File:** `CASCADING_SHUTDOWN.md`  
**Status:** 🟡 PARTIAL (TEAM-030)

**Purpose:** Clean termination cascade from queen → hives → workers

**Location:** Multiple components

**Key Features:**
- ✅ rbee-hive → workers HTTP shutdown (TEAM-030)
- ✅ SIGTERM/SIGINT handlers
- ✅ Graceful HTTP server shutdown
- 🟡 queen-rbee → hives (scaffolded only)
- ❌ No force-kill (requires PID tracking)
- ❌ Sequential shutdown (slow)

---

### 15. Scheduler (queen-rbee)
**Status:** 🔴 FUTURE MILESTONE

**Purpose:** Intelligent load balancing

**Scope:**
- Multi-hive routing
- Auto-scaling
- Geographic distribution

---

### 16. Shared Crates Library
**File:** `SHARED_CRATES.md`  
**Status:** ✅ IMPLEMENTED (ready to integrate)

**Purpose:** Reusable security, validation, and utility libraries

**Security Crates (Production-Ready):**
- ✅ **auth-min** - Timing-safe auth, token fingerprinting
- ✅ **jwt-guardian** - JWT validation, revocation lists
- ✅ **secrets-management** - Secure credential loading, zeroization

**Validation & Safety:**
- ✅ **input-validation** - Log/path/command injection prevention
- ✅ **deadline-propagation** - Request timeout propagation
- ✅ **audit-logging** - Tamper-evident audit trails

**Utilities:**
- ✅ **gpu-info** - GPU detection (CUDA, Metal, CPU)
- ✅ **model-catalog** - Model tracking (SQLite)
- ✅ **hive-core** - Shared hive types

**Integration Status:** 🔴 Ready but not yet integrated into main components

---

## Status Legend

- ✅ **IMPLEMENTED** - Fully functional
- 🟡 **PARTIAL** - Core exists, gaps remain
- 🔴 **NOT IMPLEMENTED** - Design only or stub
- 🟠 **INCOMPLETE** - Started but major gaps

## Apology Note

**TEAM-096 initially misreported status** - I didn't thoroughly search the `bin/` folder and incorrectly marked many components as "NOT IMPLEMENTED". After proper investigation:

- ✅ queen-rbee: Beehive registry, worker registry, SSH - ALL IMPLEMENTED (TEAM-043, 046, 052, 080)
- ✅ rbee-keeper: Queen lifecycle, SSH, all commands - ALL IMPLEMENTED (TEAM-043, 085)
- ✅ Local/network mode: SSH support, auto-start - IMPLEMENTED (TEAM-043, 085)

**Only genuine gap:** rbee-hive worker lifecycle (PID tracking, force kill) - documented in LIFECYCLE_MANAGEMENT_GAPS.md

## Component Dependencies

```
rbee-keeper
    └─► queen-rbee
            ├─► Hive Registry (SQLite)
            ├─► Worker Registry (RAM)
            └─► rbee-hive (multiple)
                    ├─► Worker Registry (RAM)
                    ├─► Model Catalog (SQLite)
                    ├─► Model Provisioner
                    ├─► Worker Provisioner
                    └─► llm-worker-rbee (multiple)
```

## Priority Matrix

### P0 - Critical (Blocks Production)
1. Worker lifecycle PID tracking (rbee-hive) - ONLY REMAINING CRITICAL GAP
2. Force kill capability (rbee-hive) - ONLY REMAINING CRITICAL GAP
3. ~~Hive registry (queen-rbee)~~ ✅ DONE (TEAM-043)
4. ~~Mode detection (all components)~~ ✅ DONE (TEAM-043, 085)

### P1 - High (Needed for Reliability)
5. ~~Hive lifecycle management (queen-rbee)~~ ✅ DONE (TEAM-043, 085)
6. ~~SSH support (queen-rbee, rbee-keeper)~~ ✅ DONE (TEAM-043)
7. Worker restart policy (rbee-hive) - Still needed
8. Heartbeat mechanism (rbee-hive) - Still needed

### P2 - Medium (Needed for Operations)
9. Worker provisioner (rbee-hive)
10. Metrics collection (all)
11. Scheduler (queen-rbee)

## Historical Context

**Major Contributors:**
- TEAM-027: Initial spawn, monitoring, timeout (rbee-hive)
- TEAM-029: Model provisioning (rbee-hive)
- TEAM-030: Graceful shutdown, catalogs (rbee-hive)
- TEAM-035: Worker CLI args (rbee-hive)
- TEAM-043: Beehive registry, worker registry, SSH (queen-rbee, rbee-keeper)
- TEAM-046: Worker management extensions (queen-rbee)
- TEAM-052: Backend capabilities tracking (queen-rbee)
- TEAM-080: Concurrent testing support (queen-rbee)
- TEAM-085: Queen auto-start lifecycle (rbee-keeper)
- TEAM-087: Spawn diagnostics (rbee-hive)
- TEAM-088: Stdout/stderr inheritance (rbee-hive, rbee-keeper)
- TEAM-092: Model metadata (rbee-hive)
- TEAM-096: Port allocation, fail-fast, lifecycle analysis (rbee-hive)

## Detailed Component Documentation

### Core Binaries
- `RBEE_KEEPER.md` - CLI tool
- `QUEEN_RBEE.md` - Orchestrator daemon
- `RBEE_HIVE.md` - Worker pool manager

### Registries
- `BEEHIVE_REGISTRY.md` - Hive registration (SQLite)
- `WORKER_REGISTRY_QUEEN.md` - Queen's worker tracking (RAM)
- `WORKER_REGISTRY_HIVE.md` - Hive's worker tracking (RAM)
- `MODEL_CATALOG.md` - Model tracking (SQLite)

### Provisioning & Management
- `MODEL_PROVISIONER.md` - Model downloads
- `WORKER_LIFECYCLE.md` - Worker process management

### Communication & Lifecycle
- `SSE_STREAMING.md` - Real-time streaming
- `LOCAL_VS_NETWORK_MODE.md` - Deployment modes
- `CASCADING_SHUTDOWN.md` - Clean termination cascade (TEAM-030)

### Shared Libraries
- `SHARED_CRATES.md` - Security, validation, and utility crates (9 crates documented)
- `SHARED_CRATES_INTEGRATION.md` - How to integrate shared crates into components

### Analysis Documents
- `LIFECYCLE_MANAGEMENT_GAPS.md` - Detailed lifecycle analysis
- `TEAM_096_PORT_ALLOCATION_FIX.md` - Port allocation fix
- `TEAM_096_SUMMARY.md` - TEAM-096 work summary
- `RELEASE_CANDIDATE_CHECKLIST.md` - Production readiness checklist (v0.1.0)

### Production Release Plan
- `PLAN/START_HERE.md` - **START HERE** for production release work
- `PLAN/TEAM_097_BDD_P0_SECURITY.md` - Security tests (auth, secrets, validation)
- `PLAN/TEAM_098_BDD_P0_LIFECYCLE.md` - Lifecycle tests (PID, error handling)
- `PLAN/TEAM_099_BDD_P1_OPERATIONS.md` - Operations tests (audit, deadlines)
- `PLAN/TEAM_100_BDD_P2_OBSERVABILITY.md` - Observability tests (metrics, config)
- `PLAN/TEAM_101_IMPL_WORKER_LIFECYCLE.md` - Worker lifecycle implementation
- `PLAN/TEAM_102_IMPL_SECURITY.md` - Security implementation
- `PLAN/TEAM_103_IMPL_OPERATIONS.md` - Operations implementation
- `PLAN/TEAM_104_IMPL_OBSERVABILITY.md` - Observability implementation
- `PLAN/TEAM_105_IMPL_CASCADING_SHUTDOWN.md` - Cascading shutdown implementation
- `PLAN/TEAM_106_INTEGRATION_TESTING.md` - Integration testing
- `PLAN/TEAM_107_CHAOS_LOAD_TESTING.md` - Chaos & load testing
- `PLAN/TEAM_108_FINAL_VALIDATION.md` - Final RC sign-off

---

**Created by:** TEAM-096 | 2025-10-18  
**Purpose:** Provide comprehensive component overview
