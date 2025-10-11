# Feature File Correct Structure
# Created by: TEAM-077
# Date: 2025-10-11
# Updated: 2025-10-11 14:24 - Clarified M0-M1 scope, M2+ features documented but deferred
# Status: CORRECTED NAMING AND NUMBERING (M0-M1 FOCUS)

## User Feedback

1. **"Pool management" → "rbee-hive"** - Use consistent naming
2. **"Daemon lifecycle" is ambiguous** - There are 3 daemons (queen-rbee, rbee-hive, worker-rbee)
3. **Don't artificially limit to 10 features** - Create as many as needed
4. **Use XXY numbering format** - XX = feature group (00-99), Y = sub-feature (0-9)

## Correct Feature Structure

### Numbering Format: XXY
- **XX** = Feature group (00-99, allows 100 feature groups)
- **Y** = Sub-feature (0-9, allows 10 sub-features per group)
- **Example:** `010` = Feature group 01, sub-feature 0

### M0-M1 Features (13 files)

#### 010-ssh-registry-management.feature (10 scenarios) - M1
**Feature:** SSH connection setup and node registry management
**Milestone:** M1 (Pool Manager Lifecycle)
- SSH connections, authentication, node registration

#### 020-model-catalog.feature (13 scenarios) - M1
**Feature:** Model download, catalog, and GGUF support
**Milestone:** M1 (Pool Manager Lifecycle)
- Model download, catalog operations, GGUF metadata

#### 025-worker-provisioning.feature (? scenarios) - M1 NEW!
**Feature:** Worker provisioning + binaries catalog
**Milestone:** M1 (Pool Manager Lifecycle)
- Build worker binaries from git (cargo build --release --bin llm-cuda-worker-rbee)
- Track which worker types installed on each rbee-hive
- Version compatibility, binary paths
- Future: Download pre-built binaries (M2+)

#### 040-rbee-hive-worker-registry.feature (9 scenarios) - M1
**Feature:** rbee-hive's local worker registry
**Milestone:** M1 (Pool Manager Lifecycle)
- Local worker lifecycle management on ONE rbee-hive
- Explicitly: rbee-hive's in-memory registry (not queen-rbee's)

#### 050-preflight-validation.feature (? scenarios) - M1 NEW! (CONSOLIDATED)
**Feature:** ALL preflight validation checks
**Milestone:** M1 (Pool Manager Lifecycle)
**Consolidates:** SSH preflight + rbee-hive preflight + worker resource preflight
- SSH connection validation (connection, auth, latency, binary exists)
- rbee-hive readiness (HTTP API, version, backend catalog, resources)
- Worker resources (RAM, VRAM, disk, backend availability)

#### 080-worker-rbee-lifecycle.feature (11 scenarios) - M0-M1
**Feature:** worker-rbee daemon lifecycle
**Milestone:** M0 (standalone) + M1 (spawned by rbee-hive)
- Worker startup, registration, health checks, loading progress
- Explicitly: worker-rbee daemon (not queen-rbee, not rbee-hive)

#### 090-rbee-hive-lifecycle.feature (7 scenarios) - M1
**Feature:** rbee-hive daemon lifecycle
**Milestone:** M1 (Pool Manager Lifecycle)
- rbee-hive startup, shutdown, health monitoring, worker management
- Explicitly: rbee-hive daemon (not worker-rbee, not queen-rbee)

#### 110-inference-execution.feature (11 scenarios) - M0-M1
**Feature:** Inference request handling and token streaming
**Milestone:** M0 (standalone) + M1 (via rbee-hive)
- SSE streaming, cancellation, worker busy handling

#### 120-input-validation.feature (6 scenarios) - M1
**Feature:** CLI input validation and authentication
**Milestone:** M1 (Pool Manager Lifecycle)
- Model reference format, backend names, device numbers, API keys

#### 130-cli-commands.feature (9 scenarios) - M1
**Feature:** CLI command interface
**Milestone:** M1 (Pool Manager Lifecycle)
- Install, config, basic commands

#### 140-end-to-end-flows.feature (2 scenarios) - M1
**Feature:** Complete end-to-end workflows
**Milestone:** M1 (Pool Manager Lifecycle)
- Cold start, warm start integration tests

## M2+ Features (Documented but Deferred)

#### 030-queen-rbee-worker-registry.feature (? scenarios) - M2
**Feature:** queen-rbee's global worker registry
**Milestone:** M2 (Orchestrator Scheduling)
- Global view of ALL workers across ALL rbee-hive instances
- Explicitly: queen-rbee's in-memory registry (not rbee-hive's)
**Status:** Documented for M2, not implemented in M0-M1

#### 100-queen-rbee-lifecycle.feature (3 scenarios) - M2
**Feature:** queen-rbee daemon lifecycle
**Milestone:** M2 (Orchestrator Scheduling)
- queen-rbee startup, shutdown, orchestration
- Ephemeral vs persistent modes
- Explicitly: queen-rbee daemon (not rbee-hive, not worker-rbee)
**Status:** Documented for M2, not implemented in M0-M1

#### 150-authentication.feature (? scenarios) - M3
**Feature:** Authentication (auth-min)
**Milestone:** M3 (Security & Platform Readiness)
**Status:** Documented for M3, not implemented in M0-M1

#### 160-audit-logging.feature (? scenarios) - M3
**Feature:** Audit logging (audit-logging)
**Milestone:** M3 (Security & Platform Readiness)
**Status:** Documented for M3, not implemented in M0-M1

#### 170-input-validation.feature (? scenarios) - M3
**Feature:** Input validation (input-validation)
**Milestone:** M3 (Security & Platform Readiness)
**Status:** Documented for M3, not implemented in M0-M1

#### 180-secrets-management.feature (? scenarios) - M3
**Feature:** Secrets management (secrets-management)
**Milestone:** M3 (Security & Platform Readiness)
**Status:** Documented for M3, not implemented in M0-M1

#### 190-deadline-propagation.feature (? scenarios) - M3
**Feature:** Deadline propagation (deadline-propagation)
**Milestone:** M3 (Security & Platform Readiness)
**Status:** Documented for M3, not implemented in M0-M1

#### 200-rhai-scheduler.feature (? scenarios) - M2
**Feature:** Rhai programmable scheduler
**Milestone:** M2 (Orchestrator Scheduling)
**Status:** Documented for M2, not implemented in M0-M1

#### 210-queue-management.feature (? scenarios) - M2
**Feature:** Queue management and admission control
**Milestone:** M2 (Orchestrator Scheduling)
**Status:** Documented for M2, not implemented in M0-M1

## Current File: 07-daemon-lifecycle.feature Already Split

The old `07-daemon-lifecycle.feature` has been split into:
- **090-rbee-hive-lifecycle.feature** - rbee-hive daemon (M1)
- **100-queen-rbee-lifecycle.feature** - queen-rbee daemon (M2 - deferred)
- **080-worker-rbee-lifecycle.feature** - worker-rbee daemon (M0-M1)

## Action Plan for M0-M1

1. **Renumber existing 10 files** from `0X-` to `XX0-` format
2. **Create 2 new M1 files:** 025-worker-provisioning, 050-preflight-validation
3. **Consolidate preflight** - 3 separate concerns into 1 feature
4. **Keep M2+ files documented** but defer implementation:
   - 030-queen-rbee-worker-registry.feature (M2)
   - 100-queen-rbee-lifecycle.feature (M2)
   - 150-190 security features (M3)
   - 200-210 orchestrator features (M2)
   - "pool-management" → "rbee-hive-worker-registry"
   - "worker-lifecycle" → "worker-rbee-lifecycle"
   - "daemon-lifecycle" → split into specific daemons
5. **Extract scenarios** from test-001.feature for new M1 files

### Total Features

- **M0-M1:** 12 feature files (10 existing + 2 new)
- **M2+:** 9 feature files (documented but deferred)
- **Grand Total:** 21 feature files across all milestones

This is the correct approach - proper milestone scoping, no artificial limits, clear naming.
