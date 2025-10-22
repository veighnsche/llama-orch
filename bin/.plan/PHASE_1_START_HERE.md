# PHASE 1: Main Binary Behavior Discovery

**Teams:** TEAM-216, TEAM-217, TEAM-218, TEAM-219  
**Duration:** 1 day (all teams work concurrently)  
**Output:** 4 behavior inventory documents

---

## Overview

Phase 1 focuses on discovering ALL behaviors in the 4 main binaries:

1. **TEAM-216:** `rbee-keeper` (CLI client)
2. **TEAM-217:** `queen-rbee` (Queen daemon)
3. **TEAM-218:** `rbee-hive` (Hive daemon)
4. **TEAM-219:** `llm-worker-rbee` (Worker daemon)

These 4 teams work **concurrently** - no dependencies between them.

---

## Team Assignments

### TEAM-216: rbee-keeper
- **Component:** CLI client
- **Complexity:** Low-Medium
- **Key Areas:** CLI commands, HTTP client, SSE consumption
- **Guide:** `.plan/TEAM_216_GUIDE.md`
- **Output:** `.plan/TEAM_216_RBEE_KEEPER_BEHAVIORS.md`

### TEAM-217: queen-rbee
- **Component:** Queen daemon (orchestrator)
- **Complexity:** High
- **Key Areas:** HTTP API, SSE routing, job management, hive operations
- **Guide:** `.plan/TEAM_217_GUIDE.md`
- **Output:** `.plan/TEAM_217_QUEEN_RBEE_BEHAVIORS.md`

### TEAM-218: rbee-hive
- **Component:** Hive daemon
- **Complexity:** High
- **Key Areas:** Worker lifecycle, model provisioning, device detection, heartbeat
- **Guide:** `.plan/TEAM_218_GUIDE.md`
- **Output:** `.plan/TEAM_218_RBEE_HIVE_BEHAVIORS.md`

### TEAM-219: llm-worker-rbee
- **Component:** Worker daemon
- **Complexity:** High
- **Key Areas:** Model loading, inference, streaming, OpenAI compatibility
- **Guide:** `.plan/TEAM_219_GUIDE.md`
- **Output:** `.plan/TEAM_219_LLM_WORKER_BEHAVIORS.md`

---

## Workflow

### Step 1: Read Your Guide
Each team should start by reading their specific guide:
- `.plan/TEAM_216_GUIDE.md`
- `.plan/TEAM_217_GUIDE.md`
- `.plan/TEAM_218_GUIDE.md`
- `.plan/TEAM_219_GUIDE.md`

### Step 2: Investigate Component
Follow the investigation areas in your guide:
- Read all source files
- Understand all behaviors
- Identify all edge cases
- Document everything

### Step 3: Document Findings
Create your behavior inventory document following the template from:
`.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md`

### Step 4: Verify Deliverables
Check your deliverables against the checklist in your guide:
- [ ] Document follows template
- [ ] All behaviors documented
- [ ] All edge cases identified
- [ ] All error paths documented
- [ ] Test coverage gaps identified
- [ ] Code signatures added
- [ ] Max 3 pages
- [ ] No TODO markers

---

## Template Reminder

Your behavior inventory MUST include these sections:

1. **Public API Surface** - Functions, endpoints, CLI commands
2. **State Machine Behaviors** - States, transitions, lifecycle
3. **Data Flows** - Inputs, outputs, transformations
4. **Error Handling** - Error types, propagation, recovery
5. **Integration Points** - Dependencies, dependents, contracts
6. **Critical Invariants** - What must always be true
7. **Existing Test Coverage** - Unit, BDD, integration tests
8. **Behavior Checklist** - Verification of completeness

See `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md` for full template.

---

## Critical Rules

### DO:
- ✅ Document what EXISTS (not what should exist)
- ✅ Include code examples with line numbers
- ✅ Identify test coverage gaps
- ✅ Add code signatures (`// TEAM-XXX: Investigated`)
- ✅ Follow template exactly
- ✅ Keep document ≤3 pages
- ✅ Work concurrently with other teams

### DON'T:
- ❌ Modify code during discovery
- ❌ Add TODO markers
- ❌ Skip edge cases
- ❌ Skip error paths
- ❌ Exceed 3 pages
- ❌ Wait for other teams

---

## Coordination

### No Dependencies
All 4 teams are independent:
- TEAM-216 doesn't need TEAM-217
- TEAM-217 doesn't need TEAM-218
- TEAM-218 doesn't need TEAM-219
- All can work in parallel

### Shared Resources
All teams use the same template:
- `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md`

### Communication
If you discover cross-component behaviors:
- Document them in your own inventory
- Note them for Phase 5 (integration flows)

---

## Success Criteria

### Individual Team Success
Each team must deliver:
1. ✅ Behavior inventory document
2. ✅ Following template structure
3. ✅ Max 3 pages
4. ✅ Code signatures added
5. ✅ All behaviors documented
6. ✅ All edge cases identified
7. ✅ Test coverage gaps identified

### Phase 1 Success
All 4 teams must:
1. ✅ Complete their inventories
2. ✅ Document ALL behaviors
3. ✅ Identify coverage gaps
4. ✅ Hand off to Phase 6 (test planning)

---

## Timeline

### Day 1
- All 4 teams start investigation
- All 4 teams work concurrently
- All 4 teams complete inventories

### After Phase 1
- Phase 2 starts (Queen crates)
- Phase 3 starts (Hive crates)
- Phase 4 starts (Shared crates)
- Phase 5 starts (Integration flows)
- Phase 6 starts (Test planning)

---

## Questions?

See:
- `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md` - Overall plan
- `.plan/TEAM_XXX_GUIDE.md` - Your specific guide
- `bin/engineering-rules.md` - Engineering standards

---

## Next Phase

After Phase 1 completes, the following teams start:

**Phase 2 (Queen Crates):**
- TEAM-220: hive-lifecycle
- TEAM-221: hive-registry
- TEAM-222: ssh-client

**Phase 3 (Hive Crates):**
- TEAM-223: device-detection
- TEAM-224: download-tracker
- TEAM-225: model-catalog
- TEAM-226: model-provisioner
- TEAM-227: monitor
- TEAM-228: vram-checker
- TEAM-229: worker-management

**Phase 4 (Shared Crates):**
- TEAM-230: narration
- TEAM-231: daemon-lifecycle
- TEAM-232: http-client
- TEAM-233: config-operations
- TEAM-234: job-deadline
- TEAM-235: auth-jwt
- TEAM-236: audit-validation
- TEAM-237: heartbeat-update

**Phase 5 (Integration):**
- TEAM-238: keeper-queen
- TEAM-239: queen-hive
- TEAM-240: hive-worker
- TEAM-241: e2e-inference

**Phase 6 (Test Planning):**
- TEAM-242+: Create comprehensive test plans

**Phase 7 (Test Implementation):**
- TEAM-250+: Implement all tests

---

**Status:** READY TO START  
**Start Date:** TBD  
**Expected Completion:** 1 day (concurrent work)
