# 🚀 MIGRATION MASTER PLAN
## Old Code → New Numbered Structure

**Created:** 2025-10-20  
**Status:** 🔴 NOT STARTED  
**Total Work Units:** 24 (12 BDD + 12 Code)  
**Estimated Time:** 40-60 hours  

---

## 📊 Migration Overview

### Source (OLD)
```
/home/vince/Projects/llama-orch/bin/
├── old.rbee-keeper/    (~15 files)
├── old.queen-rbee/     (~19 files)
└── old.rbee-hive/      (~37 files)
```

### Target (NEW)
```
/home/vince/Projects/llama-orch/bin/
├── 00_rbee_keeper/           (main binary)
├── 05_rbee_keeper_crates/    (4 crates)
├── 10_queen_rbee/            (main binary)
├── 15_queen_rbee_crates/     (8 crates)
├── 20_rbee_hive/             (main binary)
├── 25_rbee_hive_crates/      (9 crates)
├── 30_llm_worker_rbee/       (main binary)
└── 99_shared_crates/         (shared utilities)
```

---

## 🎯 Migration Strategy

### Phase 1: BDD Test Migration (12 units)
Move tests from `test-harness/bdd` to individual crate `bdd` folders

### Phase 2: Code Migration (12 units)
Migrate actual implementation from old structure to new numbered crates

---

## 📋 PHASE 1: BDD TEST MIGRATION

Total: **29 feature files** in `test-harness/bdd/tests/features/`

### UNIT 1-A: rbee-keeper BDD Tests (2-3 hours)
**Destination:** `bin/00_rbee_keeper/bdd/`

**Files:**
- `150-cli-commands.feature` → CLI command testing
- `360-configuration-management.feature` (keeper config portion)

**New crates to test:**
- `bin/05_rbee_keeper_crates/commands/bdd/`
- `bin/05_rbee_keeper_crates/config/bdd/`
- `bin/05_rbee_keeper_crates/queen-lifecycle/bdd/`

**Tasks:**
1. Read old feature files
2. Split scenarios by crate responsibility
3. Create feature files in respective bdd folders
4. Add step definitions in `tests/steps/`
5. Update Cargo.toml with cucumber dependencies

---

### UNIT 1-B: queen-rbee Registry BDD Tests (2-3 hours)
**Destination:** `bin/15_queen_rbee_crates/`

**Files:**
- `010-ssh-registry-management.feature` → `ssh-client/bdd/`
- `050-queen-rbee-worker-registry.feature` → `worker-registry/bdd/`
- `bin/15_queen_rbee_crates/hive-registry/bdd/`

**Tasks:**
1. Migrate SSH registry scenarios
2. Migrate worker registry scenarios
3. Create hive registry BDD tests (new)
4. Add step definitions

---

### UNIT 1-C: queen-rbee Lifecycle & Preflight BDD (2-3 hours)
**Destination:** `bin/15_queen_rbee_crates/`

**Files:**
- `120-queen-rbee-lifecycle.feature` → `hive-lifecycle/bdd/`
- `070-ssh-preflight-validation.feature` → `preflight/bdd/`

**Tasks:**
1. Migrate hive lifecycle scenarios
2. Migrate preflight validation scenarios
3. Add step definitions

---

### UNIT 1-D: rbee-hive Model Management BDD (2-3 hours)
**Destination:** `bin/25_rbee_hive_crates/`

**Files:**
- `020-model-catalog.feature` → `model-catalog/bdd/`
- `030-model-provisioner.feature` → `model-provisioner/bdd/`
- `bin/25_rbee_hive_crates/download-tracker/bdd/` (create new)

**Tasks:**
1. Migrate model catalog scenarios
2. Migrate model provisioner scenarios
3. Create download tracker BDD tests
4. Add step definitions

---

### UNIT 1-E: rbee-hive Worker Management BDD (2-3 hours)
**Destination:** `bin/25_rbee_hive_crates/`

**Files:**
- `040-worker-provisioning.feature` → `worker-catalog/` (create)
- `060-rbee-hive-worker-registry.feature` → `worker-registry/bdd/`
- `090-worker-resource-preflight.feature` → shared content

**Tasks:**
1. Migrate worker provisioning scenarios
2. Migrate worker registry scenarios
3. Add step definitions

---

### UNIT 1-F: rbee-hive Lifecycle & Monitor BDD (2-3 hours)
**Destination:** `bin/25_rbee_hive_crates/`

**Files:**
- `110-rbee-hive-lifecycle.feature` → `worker-lifecycle/bdd/`
- `080-rbee-hive-preflight-validation.feature` → split across crates
- Monitor tests (extract from `230-resource-management.feature`)

**Tasks:**
1. Migrate lifecycle scenarios
2. Create monitor BDD tests in `monitor/bdd/`
3. Migrate device detection tests to `device-detection/bdd/`
4. Add step definitions

---

### UNIT 1-G: Worker BDD Tests (2-3 hours)
**Destination:** `bin/30_llm_worker_rbee/bdd/`

**Files:**
- `100-worker-rbee-lifecycle.feature` → `bin/30_llm_worker_rbee/bdd/`
- `130-inference-execution.feature` → `bin/30_llm_worker_rbee/bdd/`

**Tasks:**
1. Migrate worker lifecycle scenarios
2. Migrate inference execution scenarios
3. Add step definitions

---

### UNIT 1-H: Shared Crates BDD - Auth & Secrets (2-3 hours)
**Destination:** `bin/99_shared_crates/`

**Files:**
- `300-authentication.feature` → `auth-min/` (if has bdd folder)
- `310-secrets-management.feature` → `secrets-management/bdd/`
- `330-audit-logging.feature` → `audit-logging/bdd/`

**Tasks:**
1. Migrate auth scenarios
2. Migrate secrets management scenarios
3. Migrate audit logging scenarios
4. Add step definitions

---

### UNIT 1-I: Shared Crates BDD - Validation & Narration (2-3 hours)
**Destination:** `bin/99_shared_crates/`

**Files:**
- `140-input-validation.feature` → `input-validation/bdd/`
- `340-deadline-propagation.feature` → `deadline-propagation/` (add bdd folder)
- Narration tests → `narration-core/bdd/`

**Tasks:**
1. Migrate input validation scenarios
2. Create deadline propagation BDD tests
3. Migrate narration scenarios
4. Add step definitions

---

### UNIT 1-J: Shared Crates BDD - HTTP & Types (2-3 hours)
**Destination:** `bin/99_shared_crates/`

**Files:**
- Create HTTP client tests → `rbee-http-client/bdd/`
- Create types tests → `rbee-types/bdd/`
- Heartbeat tests (already done)

**Tasks:**
1. Create HTTP client BDD tests
2. Create rbee-types BDD tests
3. Verify heartbeat BDD setup

---

### UNIT 1-K: Integration & Concurrency BDD (2-3 hours)
**Destination:** `bin/10_queen_rbee/bdd/` and `bin/20_rbee_hive/bdd/`

**Files:**
- `200-concurrency-scenarios.feature` → split across binaries
- `210-failure-recovery.feature` → split across binaries
- `320-error-handling.feature` → split across binaries

**Tasks:**
1. Categorize scenarios by component
2. Create integration tests in binary bdd folders
3. Add step definitions

---

### UNIT 1-L: End-to-End BDD (2-3 hours)
**Destination:** Keep in `test-harness/bdd/` (system-level)

**Files:**
- `160-end-to-end-flows.feature` (keep)
- `900-integration-e2e.feature` (keep)
- `910-full-stack-integration.feature` (keep)
- `920-integration-scenarios.feature` (keep)
- `230-resource-management.feature` (keep)
- `350-metrics-observability.feature` (keep)

**Tasks:**
1. Review and update feature files
2. Ensure they test full system
3. Add missing scenarios from architecture docs

---

## 📋 PHASE 2: CODE MIGRATION

### UNIT 2-A: rbee-keeper Commands (3-4 hours)
**Source:** `old.rbee-keeper/src/commands/`  
**Destination:** `bin/05_rbee_keeper_crates/commands/src/`

**Files to migrate:**
- `infer.rs` → inference command
- `start_queue.rs` → queue startup
- `stop_queue.rs` → queue shutdown
- Plus other command files

**Architecture mapping:**
- Phase 0: CLI entry (a_Claude_Sonnet_4_5_refined_this.md lines 32-42)
- Phase 2: Job submission (lines 79-95)

**Tasks:**
1. Copy command implementations
2. Adapt to new crate structure
3. Update imports for new shared crates
4. Add error handling using `thiserror`
5. Add unit tests

---

### UNIT 2-B: rbee-keeper Lifecycle & Config (3-4 hours)
**Source:** `old.rbee-keeper/src/`  
**Destination:** Multiple crates

**Files to migrate:**
- `queen_lifecycle.rs` → `bin/05_rbee_keeper_crates/queen-lifecycle/src/`
- `config.rs` → `bin/05_rbee_keeper_crates/config/src/`
- `pool_client.rs` → Use `rbee-http-client` instead

**Architecture mapping:**
- Phase 1: Start Queen (lines 45-75)

**Tasks:**
1. Migrate queen lifecycle management
2. Migrate configuration management
3. Replace custom HTTP client with `rbee-http-client`
4. Add polling logic to `bin/05_rbee_keeper_crates/polling/src/`
5. Add unit tests

---

### UNIT 2-C: queen-rbee SSH & Registries (3-4 hours)
**Source:** `old.queen-rbee/src/`  
**Destination:** `bin/15_queen_rbee_crates/`

**Files to migrate:**
- `ssh.rs` → `ssh-client/src/`
- `beehive_registry.rs` → `hive-registry/src/`
- `worker_registry.rs` → `worker-registry/src/`

**Architecture mapping:**
- Phase 3: Hive Discovery (lines 98-108)
- Phase 4: Device Detection (lines 136-164)

**Tasks:**
1. Migrate SSH client implementation
2. Migrate hive registry (catalog + RAM registry)
3. Migrate worker registry
4. Add unit tests

---

### UNIT 2-D: queen-rbee Lifecycle & Preflight (3-4 hours)
**Source:** `old.queen-rbee/src/preflight/`  
**Destination:** `bin/15_queen_rbee_crates/`

**Files to migrate:**
- `preflight/*.rs` → `preflight/src/`
- Hive lifecycle logic → `hive-lifecycle/src/`

**Architecture mapping:**
- Phase 5: Preflight Checks (lines 165-180)
- Phase 3: Start Hive (lines 109-135)

**Tasks:**
1. Migrate preflight validation logic
2. Migrate hive lifecycle management
3. Use `daemon-lifecycle` shared crate
4. Add unit tests

---

### UNIT 2-E: queen-rbee HTTP Server & Scheduler (3-4 hours)
**Source:** `old.queen-rbee/src/http/` and `main.rs`  
**Destination:** `bin/10_queen_rbee/src/` and `bin/15_queen_rbee_crates/scheduler/`

**Files to migrate:**
- `http/*.rs` → `bin/10_queen_rbee/src/http/`
- Create scheduler in `bin/15_queen_rbee_crates/scheduler/src/`
- `main.rs` → `bin/10_queen_rbee/src/main.rs`

**Architecture mapping:**
- Phase 7: Schedule Job (lines 190-216)
- Phase 11: Receive Inference (lines 315-328)

**Tasks:**
1. Migrate HTTP routes
2. Create scheduling logic
3. Wire up all crates in main.rs
4. Add health endpoint
5. Add unit tests

---

### UNIT 2-F: rbee-hive Model Management (3-4 hours)
**Source:** `old.rbee-hive/src/provisioner/`  
**Destination:** `bin/25_rbee_hive_crates/`

**Files to migrate:**
- `provisioner/model_provisioner.rs` → `model-provisioner/src/`
- `provisioner/*.rs` → `model-provisioner/src/`
- Model catalog logic → `model-catalog/src/` (may already exist in shared)
- `download_tracker.rs` → `download-tracker/src/`

**Architecture mapping:**
- Phase 8: Model Download (lines 217-263)

**Tasks:**
1. Migrate model provisioner
2. Migrate download tracker
3. Link with model-catalog
4. Add SSE support for progress
5. Add unit tests

---

### UNIT 2-G: rbee-hive Worker Management (3-4 hours)
**Source:** `old.rbee-hive/src/`  
**Destination:** `bin/25_rbee_hive_crates/`

**Files to migrate:**
- `worker_provisioner.rs` → Create `worker-catalog/src/`
- `registry.rs` → `worker-registry/src/`
- Worker lifecycle logic → `worker-lifecycle/src/`

**Architecture mapping:**
- Phase 9: Worker Spawning (lines 264-299)
- Phase 10: Heartbeats (lines 300-313)

**Tasks:**
1. Migrate worker provisioner
2. Migrate worker registry
3. Migrate worker lifecycle management
4. Use `heartbeat` shared crate
5. Add unit tests

---

### UNIT 2-H: rbee-hive Monitor & Resources (3-4 hours)
**Source:** `old.rbee-hive/src/`  
**Destination:** `bin/25_rbee_hive_crates/`

**Files to migrate:**
- `monitor.rs` → `monitor/src/`
- `resources.rs` → `device-detection/src/` and `vram-checker/` (create)
- `metrics.rs` → Add to monitor or main binary

**Architecture mapping:**
- Phase 4: Device Detection (lines 136-164)
- Phase 6: VRAM Check (lines 181-189)

**Tasks:**
1. Migrate monitor implementation
2. Create device detection crate
3. Create VRAM checker crate
4. Add metrics collection
5. Add unit tests

---

### UNIT 2-I: rbee-hive HTTP Server & Main (3-4 hours)
**Source:** `old.rbee-hive/src/`  
**Destination:** `bin/20_rbee_hive/src/`

**Files to migrate:**
- `http/*.rs` → `bin/20_rbee_hive/src/http/` (some already done)
- `main.rs` → `bin/20_rbee_hive/src/main.rs`
- `shutdown.rs`, `restart.rs`, `timeout.rs` → integrate into main

**Architecture mapping:**
- Phase 11: Worker Inference Relay (lines 329-354)

**Tasks:**
1. Complete HTTP endpoint migration
2. Wire up all hive crates in main.rs
3. Add shutdown handling
4. Add restart logic
5. Add unit tests

---

### UNIT 2-J: Shared Crate - daemon-lifecycle (3-4 hours)
**Source:** Multiple lifecycle implementations in old code  
**Destination:** `bin/99_shared_crates/daemon-lifecycle/src/`

**Extract from:**
- `old.rbee-keeper/src/queen_lifecycle.rs`
- `old.queen-rbee/` (hive lifecycle logic)
- `old.rbee-hive/` (worker lifecycle logic)

**Pattern to consolidate:**
- Spawn process
- Health check polling
- Shutdown handling
- Restart logic

**Tasks:**
1. Identify common patterns
2. Create generic lifecycle trait
3. Implement for different daemon types
4. Add unit tests

---

### UNIT 2-K: Shared Crate - rbee-http-client (3-4 hours)
**Source:** HTTP client usage across old code  
**Destination:** `bin/99_shared_crates/rbee-http-client/src/`

**Extract from:**
- `old.rbee-keeper/src/pool_client.rs`
- Old queen HTTP calls
- Old hive HTTP calls

**Features needed:**
- GET/POST/DELETE support
- Authentication headers
- Retry logic
- Timeout handling
- SSE support

**Tasks:**
1. Create unified HTTP client
2. Add authentication support
3. Add retry/timeout logic
4. Add SSE client
5. Add unit tests

---

### UNIT 2-L: Shared Crate - rbee-types (3-4 hours)
**Source:** Type definitions scattered across old code  
**Destination:** `bin/99_shared_crates/rbee-types/src/`

**Extract from:**
- Job types
- Worker types
- Model types
- Registry types
- Common request/response types

**Tasks:**
1. Identify all shared types
2. Organize into modules
3. Add serde derives
4. Add validation
5. Add unit tests

---

## 📊 Work Distribution

### Team Size: 3 people (you + 2 others)

**Person A (Specialist: BDD Tests)**
- UNIT 1-A: rbee-keeper BDD
- UNIT 1-D: rbee-hive Model Management BDD
- UNIT 1-G: Worker BDD
- UNIT 1-J: Shared HTTP & Types BDD

**Person B (Specialist: queen-rbee)**
- UNIT 1-B: queen-rbee Registry BDD
- UNIT 1-C: queen-rbee Lifecycle BDD
- UNIT 2-C: queen-rbee SSH & Registries Code
- UNIT 2-D: queen-rbee Lifecycle Code
- UNIT 2-E: queen-rbee HTTP Server Code

**Person C (Specialist: rbee-hive)**
- UNIT 1-E: rbee-hive Worker BDD
- UNIT 1-F: rbee-hive Lifecycle BDD
- UNIT 2-F: rbee-hive Model Management Code
- UNIT 2-G: rbee-hive Worker Management Code
- UNIT 2-H: rbee-hive Monitor Code
- UNIT 2-I: rbee-hive HTTP Server Code

**Person A + Person C (Shared Work)**
- UNIT 2-A: rbee-keeper Commands Code
- UNIT 2-B: rbee-keeper Lifecycle Code
- UNIT 2-J: daemon-lifecycle Shared Crate
- UNIT 2-K: rbee-http-client Shared Crate
- UNIT 2-L: rbee-types Shared Crate

**All Team (Review & Integration)**
- UNIT 1-H: Shared Auth & Secrets BDD
- UNIT 1-I: Shared Validation BDD
- UNIT 1-K: Integration BDD
- UNIT 1-L: End-to-End BDD

---

## 🎯 Success Criteria

### Phase 1 Complete (BDD):
- [ ] All 29 feature files migrated or kept
- [ ] Each crate has BDD folder with relevant tests
- [ ] All step definitions implemented
- [ ] All BDD tests pass: `cargo test --all`

### Phase 2 Complete (Code):
- [ ] All old code migrated to new structure
- [ ] All binaries compile: `cargo build --all`
- [ ] All unit tests pass: `cargo test --all`
- [ ] All BDD tests still pass
- [ ] Architecture docs match implementation

---

## 📅 Timeline Estimate

**Aggressive (3 people, full-time):**
- Week 1: BDD Migration (Units 1-A through 1-L)
- Week 2: Code Migration Part 1 (Units 2-A through 2-F)
- Week 3: Code Migration Part 2 (Units 2-G through 2-L)
- Week 4: Integration, testing, documentation

**Realistic (3 people, part-time):**
- 6-8 weeks total

**Solo (1 person, full-time):**
- 8-10 weeks total

---

## 🚨 Critical Dependencies

### Must Complete First:
1. ✅ Heartbeat crate (COMPLETE)
2. UNIT 2-J: daemon-lifecycle (needed by keepers)
3. UNIT 2-K: rbee-http-client (needed by all)
4. UNIT 2-L: rbee-types (needed by all)

### Parallel Work:
- All BDD units can be done in parallel
- Code units within each service can be done in parallel
- Services can be migrated in parallel

---

## 📝 Notes

**Architecture References:**
- `/home/vince/Projects/llama-orch/bin/a_human_wrote_this.md` - Original flow
- `/home/vince/Projects/llama-orch/bin/a_chatGPT_5_refined_this.md` - ChatGPT refinement
- `/home/vince/Projects/llama-orch/bin/a_Claude_Sonnet_4_5_refined_this.md` - Code-backed refinement (MOST ACCURATE)

**Old Code:**
- `/home/vince/Projects/llama-orch/bin/old.rbee-keeper/`
- `/home/vince/Projects/llama-orch/bin/old.queen-rbee/`
- `/home/vince/Projects/llama-orch/bin/old.rbee-hive/`

**Current BDD Tests:**
- `/home/vince/Projects/llama-orch/test-harness/bdd/tests/features/`

---

**END OF MASTER PLAN**  
**Ready for work assignment!**  
**Good luck team! 🚀**
