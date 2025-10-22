# BEHAVIOR DISCOVERY MASTER PLAN

**Mission:** Inventory ALL behaviors in `/bin/` to create comprehensive test coverage that freezes current functionality.

**Strategy:** Multi-phase concurrent discovery → Comprehensive test plan → Implementation

---

## Overview

### Scope

**Main Binaries (4):**
- `00_rbee_keeper` - CLI client
- `10_queen_rbee` - Queen daemon (orchestrator)
- `20_rbee_hive` - Hive daemon (manages workers)
- `30_llm_worker_rbee` - Worker daemon (runs models)

**Supporting Crates (30+):**
- `15_queen_rbee_crates/` - 3 crates (hive-lifecycle, hive-registry, ssh-client)
- `25_rbee_hive_crates/` - 7 crates (device-detection, download-tracker, model-catalog, model-provisioner, monitor, vram-checker, worker-catalog, worker-lifecycle, worker-registry)
- `99_shared_crates/` - 20+ crates (narration-core, daemon-lifecycle, rbee-http-client, etc.)

---

## Discovery Phases

### Phase 1: Main Binary Behaviors (TEAM-216 → TEAM-219)
**Duration:** 4 teams working concurrently  
**Output:** 4 behavior inventory documents

- **TEAM-216:** `rbee-keeper` CLI behaviors
- **TEAM-217:** `queen-rbee` daemon behaviors
- **TEAM-218:** `rbee-hive` daemon behaviors
- **TEAM-219:** `llm-worker-rbee` daemon behaviors

### Phase 2: Queen Crate Behaviors (TEAM-220 → TEAM-222)
**Duration:** 3 teams working concurrently  
**Output:** 3 behavior inventory documents

- **TEAM-220:** `hive-lifecycle` behaviors
- **TEAM-221:** `hive-registry` behaviors
- **TEAM-222:** `ssh-client` behaviors

### Phase 3: Hive Crate Behaviors (TEAM-223 → TEAM-229)
**Duration:** 7 teams working concurrently  
**Output:** 7 behavior inventory documents

- **TEAM-223:** `device-detection` behaviors
- **TEAM-224:** `download-tracker` behaviors
- **TEAM-225:** `model-catalog` behaviors
- **TEAM-226:** `model-provisioner` behaviors
- **TEAM-227:** `monitor` behaviors
- **TEAM-228:** `vram-checker` behaviors
- **TEAM-229:** `worker-catalog` + `worker-lifecycle` + `worker-registry` behaviors

### Phase 4: Shared Crate Behaviors (TEAM-230 → TEAM-237)
**Duration:** 8 teams working concurrently  
**Output:** 8 behavior inventory documents

- **TEAM-230:** `narration-core` + `narration-macros` behaviors
- **TEAM-231:** `daemon-lifecycle` behaviors
- **TEAM-232:** `rbee-http-client` behaviors
- **TEAM-233:** `rbee-config` + `rbee-operations` behaviors
- **TEAM-234:** `job-registry` + `deadline-propagation` behaviors
- **TEAM-235:** `auth-min` + `jwt-guardian` behaviors
- **TEAM-236:** `audit-logging` + `input-validation` behaviors
- **TEAM-237:** `heartbeat` + `auto-update` + `hive-core` behaviors

### Phase 5: Integration Behaviors (TEAM-238 → TEAM-241)
**Duration:** 4 teams working concurrently  
**Output:** 4 integration flow documents

- **TEAM-238:** Keeper → Queen flows
- **TEAM-239:** Queen → Hive flows
- **TEAM-240:** Hive → Worker flows
- **TEAM-241:** End-to-end inference flows

---

## Behavior Inventory Template

Each team MUST produce a document following this structure:

```markdown
# [COMPONENT] BEHAVIOR INVENTORY

**Team:** TEAM-XXX  
**Component:** [crate/binary name]  
**Date:** [date]

## 1. Public API Surface

### Functions/Methods
- List ALL public functions with signatures
- Document parameters, return types, error cases

### Endpoints (if applicable)
- List ALL HTTP endpoints
- Document request/response schemas
- Document status codes

### CLI Commands (if applicable)
- List ALL commands and subcommands
- Document flags, arguments, defaults

## 2. State Machine Behaviors

### States
- List ALL possible states
- Document state transitions
- Document invariants

### Lifecycle Events
- Startup behaviors
- Shutdown behaviors
- Crash/recovery behaviors

## 3. Data Flows

### Inputs
- Configuration files
- Environment variables
- Command-line arguments
- HTTP requests
- IPC messages

### Outputs
- Stdout/stderr
- Log files
- HTTP responses
- SSE streams
- File writes

## 4. Error Handling

### Error Types
- List ALL error variants
- Document error propagation paths
- Document retry logic

### Edge Cases
- Timeout behaviors
- Network failures
- Resource exhaustion
- Invalid inputs
- Race conditions

## 5. Integration Points

### Dependencies
- Which crates/services does this depend on?
- What happens if dependencies fail?

### Dependents
- Which crates/services depend on this?
- What contracts must be maintained?

## 6. Critical Invariants

- What MUST always be true?
- What safety guarantees exist?
- What performance characteristics?

## 7. Existing Test Coverage

### Unit Tests
- Count of tests
- Coverage gaps identified

### BDD Tests
- Existing features/scenarios
- Coverage gaps identified

### Integration Tests
- Existing test cases
- Coverage gaps identified

## 8. Behavior Checklist

- [ ] All public APIs documented
- [ ] All state transitions documented
- [ ] All error paths documented
- [ ] All integration points documented
- [ ] All edge cases documented
- [ ] Existing test coverage assessed
- [ ] Coverage gaps identified
```

---

## Success Criteria

### Per-Team Deliverables
1. **Behavior inventory document** following template
2. **Code signatures** (`// TEAM-XXX:`) on all investigated files
3. **Compilation verification** (`cargo check -p [package]`)
4. **No TODO markers** in deliverables

### Phase Completion Criteria
- [ ] All teams in phase have completed inventories
- [ ] All inventories follow template structure
- [ ] All coverage gaps identified
- [ ] Cross-references validated

### Master Plan Completion
- [ ] All 26 behavior inventories complete
- [ ] Comprehensive test plan created (Phase 6)
- [ ] Test implementation roadmap created (Phase 7)

---

## Next Steps After Discovery

### Phase 6: Test Plan Creation (TEAM-242+)
Based on behavior inventories, create comprehensive test plans:
- Unit test plans
- BDD test plans
- Integration test plans
- E2E test plans

### Phase 7: Test Implementation (TEAM-250+)
Implement tests to freeze all discovered behaviors:
- Write unit tests
- Write BDD scenarios
- Write integration tests
- Write E2E tests via xtask

---

## Coordination Rules

### Concurrent Work
- Teams within same phase work concurrently
- Teams in different phases can overlap
- No dependencies between teams in same phase

### Documentation Standards
- All docs in `/bin/.plan/TEAM_XXX_[component]_BEHAVIORS.md`
- Max 3 pages per inventory
- Use exact template structure
- Include code examples

### Code Signatures
- Add `// TEAM-XXX: Investigated [date]` to files examined
- Don't modify code during discovery phase
- Only document what exists

---

## Timeline Estimate

- **Phase 1:** 4 teams × 1 day = 4 team-days (concurrent = 1 day)
- **Phase 2:** 3 teams × 1 day = 3 team-days (concurrent = 1 day)
- **Phase 3:** 7 teams × 1 day = 7 team-days (concurrent = 1 day)
- **Phase 4:** 8 teams × 1 day = 8 team-days (concurrent = 1 day)
- **Phase 5:** 4 teams × 1 day = 4 team-days (concurrent = 1 day)

**Total Discovery:** 5 days (with perfect parallelism)

**Test Planning:** 2-3 days  
**Test Implementation:** 10-15 days

**Grand Total:** ~20-25 days to freeze all behaviors

---

## Document Index

### Phase 1 (Main Binaries)
- `.plan/TEAM_216_RBEE_KEEPER_BEHAVIORS.md`
- `.plan/TEAM_217_QUEEN_RBEE_BEHAVIORS.md`
- `.plan/TEAM_218_RBEE_HIVE_BEHAVIORS.md`
- `.plan/TEAM_219_LLM_WORKER_BEHAVIORS.md`

### Phase 2 (Queen Crates)
- `.plan/TEAM_220_HIVE_LIFECYCLE_BEHAVIORS.md`
- `.plan/TEAM_221_HIVE_REGISTRY_BEHAVIORS.md`
- `.plan/TEAM_222_SSH_CLIENT_BEHAVIORS.md`

### Phase 3 (Hive Crates)
- `.plan/TEAM_223_DEVICE_DETECTION_BEHAVIORS.md`
- `.plan/TEAM_224_DOWNLOAD_TRACKER_BEHAVIORS.md`
- `.plan/TEAM_225_MODEL_CATALOG_BEHAVIORS.md`
- `.plan/TEAM_226_MODEL_PROVISIONER_BEHAVIORS.md`
- `.plan/TEAM_227_MONITOR_BEHAVIORS.md`
- `.plan/TEAM_228_VRAM_CHECKER_BEHAVIORS.md`
- `.plan/TEAM_229_WORKER_MANAGEMENT_BEHAVIORS.md`

### Phase 4 (Shared Crates)
- `.plan/TEAM_230_NARRATION_BEHAVIORS.md`
- `.plan/TEAM_231_DAEMON_LIFECYCLE_BEHAVIORS.md`
- `.plan/TEAM_232_HTTP_CLIENT_BEHAVIORS.md`
- `.plan/TEAM_233_CONFIG_OPERATIONS_BEHAVIORS.md`
- `.plan/TEAM_234_JOB_DEADLINE_BEHAVIORS.md`
- `.plan/TEAM_235_AUTH_JWT_BEHAVIORS.md`
- `.plan/TEAM_236_AUDIT_VALIDATION_BEHAVIORS.md`
- `.plan/TEAM_237_HEARTBEAT_UPDATE_BEHAVIORS.md`

### Phase 5 (Integration Flows)
- `.plan/TEAM_238_KEEPER_QUEEN_INTEGRATION.md`
- `.plan/TEAM_239_QUEEN_HIVE_INTEGRATION.md`
- `.plan/TEAM_240_HIVE_WORKER_INTEGRATION.md`
- `.plan/TEAM_241_E2E_INFERENCE_FLOWS.md`

---

## Critical Notes

1. **This is DISCOVERY only** - No code changes during Phases 1-5
2. **Document what EXISTS** - Don't design new behaviors
3. **Identify gaps** - Test coverage gaps are valuable findings
4. **Follow template** - Consistency enables Phase 6 planning
5. **Work concurrently** - Teams in same phase are independent

---

**Status:** READY FOR TEAM-216  
**Next:** Create individual team guides for Phase 1
