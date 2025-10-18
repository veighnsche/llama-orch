# Comprehensive Feature Map
# Created by: TEAM-077
# Date: 2025-10-11
# Updated: 2025-10-11 14:23 - Filtered for M0-M1 Structure (14 files) + M2+ Features (Documented but Deferred)

**MORE FILES = BETTER CLARITY!** Each file = one concern, one stakeholder, one component.
# Status: COMPLETE COMPONENT MAPPING (M0-M1 focus)

## User Feedback: Missing Components (M0-M1 Scope)

The current structure is missing several critical M0-M1 components:

1. **SSH preflight validation** - queen-rbee → rbee-hive connection validation (M1)
2. **rbee-hive preflight validation** - rbee-hive readiness validation (M1)
3. **Worker resource validation** - Worker resource validation (RAM, VRAM, disk, backend)s validation (M1)
4. **Worker binaries catalog** - Which worker types installed on each rbee-hive (M1)
5. **queen-rbee's beehive registry** - SSH connection details (SQLite) (M1)
6. **queen-rbee's worker registry** - Worker connection details (in-memory) (M2)
7. **rbee-hive's worker registry** - Worker lifecycle management (in-memory) (M1)

## Complete Component Breakdown

### Registries & Catalogs (6 components)

#### 010-ssh-registry-management.feature (EXISTING)
**Component:** queen-rbee's beehive registry (SQLite)
**Purpose:** SSH connection details for rbee-hive nodes
**Operations:** Add node, remove node, list nodes, update capabilities
**Storage:** SQLite at `~/.rbee/beehives.db`

#### 020-model-catalog.feature (RENAME from model-provisioning)
**Component:** Model catalog (SQLite)
**Purpose:** Model storage, download, and metadata
**Operations:** Download, register, query, GGUF metadata
**Storage:** SQLite at `~/.rbee/models.db`

#### 025-worker-binaries-catalog.feature (NEW!)
**Component:** Worker binaries catalog (SQLite)
**Purpose:** Track which worker types are installed on each rbee-hive
**Storage:** SQLite at `~/.rbee/worker_binaries.db` (or part of beehives.db)
**Example entries:**
- rbee-hive: workstation, worker_type: llm-worker-rbee, version: 0.1.0, path: /usr/local/bin/llm-worker-rbee
- rbee-hive: workstation, worker_type: embedding-worker-rbee, version: 0.1.0, path: /usr/local/bin/embedding-worker-rbee

#### 2. 030-queen-rbee-worker-registry.feature (M1)
**Scenarios:**
- Query all workers across all rbee-hive instances
- Filter workers by capability (GPU, CPU)
- Aggregate worker statistics
- Worker registry updates when rbee-hive reports
- Stale worker cleanup
**Status:** M1 foundational feature - just HTTP endpoints!
**Storage:** In-memory (ephemeral)
**Scope:** Global view across all rbee-hive instances

#### 040-rbee-hive-worker-registry.feature (RENAME from rbee-hive-management)
**Component:** rbee-hive's worker registry (in-memory)
{{ ... }}
7. **Verify** all scenarios accounted for
8. **Compile** and test

## M2+ Features (Documented but Deferred)

### M2: Rhai Programmable Scheduler (ONLY Rhai complexity!)
- 200-rhai-scheduler.feature - Custom Rhai scripting
- 210-queue-management.feature - Priority queues

**NOTE:** queen-rbee worker registry and lifecycle are M1, NOT M2!

### M3: Security & Platform
- 150-authentication.feature - auth-min (timing-safe tokens)
- 160-audit-logging.feature - Immutable audit trail (GDPR)
- 170-input-validation.feature - Injection attack prevention
- 190-deadline-propagation.feature - Resource exhaustion prevention

**These are documented for future reference but not implemented in M0-M1.**

## Key Insights from User Feedback

1. **Preflight is multi-level:**
   - SSH level (queen-rbee → rbee-hive)
   - rbee-hive level (rbee-hive readiness)
   - Worker level (worker resources)

2. **Registries are distinct:**
   - queen-rbee has 2 registries (beehive + worker)
   - rbee-hive has 1 registry (worker)
   - Model catalog is separate
   - Worker binaries catalog is separate

3. **Worker binaries catalog is critical:**
   - Each rbee-hive needs to know which workers it has
   - Version tracking for compatibility
   - Installation verification

This is a much more complete picture of the system!
