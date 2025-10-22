# PHASE 5: Integration Flow Discovery

**Teams:** TEAM-238 to TEAM-241  
**Duration:** 1 day (all teams work concurrently)  
**Output:** 4 integration flow documents

---

## Overview

Phase 5 documents **cross-component behaviors** and **end-to-end flows**. This is different from Phases 1-4 which documented individual components.

### Focus
- How components interact
- How data flows between components
- How errors propagate across boundaries
- How distributed state is managed

---

## TEAM-238: keeper-queen Integration

**Components:** `rbee-keeper` ↔ `queen-rbee`  
**Complexity:** High  
**Output:** `.plan/TEAM_238_KEEPER_QUEEN_INTEGRATION.md`

### Investigation Areas

#### 1. Request Flow
- Document CLI command → HTTP request
- Document request serialization
- Document request routing in queen
- Document response handling in keeper

#### 2. SSE Stream Flow
- Document job creation in queen
- Document job_id generation
- Document SSE stream establishment
- Document narration event flow
- Document stream consumption in keeper

#### 3. Operation Flows
Document complete flows for:
- HiveList: CLI → Queen → Response → Display
- HiveStart: CLI → Queen → Hive spawn → Health poll → Capabilities → SSE → CLI
- HiveStop: CLI → Queen → SIGTERM → SIGKILL → SSE → CLI
- All other hive operations

#### 4. Error Propagation
- Document HTTP errors
- Document SSE errors
- Document timeout errors
- Document how errors flow to CLI
- Document error display in keeper

#### 5. State Synchronization
- Document job state tracking
- Document stream lifecycle
- Document cleanup

#### 6. Edge Cases
- Queen unreachable
- SSE stream closes early
- Multiple clients same job_id
- Network failures
- Timeout scenarios

---

## TEAM-239: queen-hive Integration

**Components:** `queen-rbee` ↔ `rbee-hive`  
**Complexity:** High  
**Output:** `.plan/TEAM_239_QUEEN_HIVE_INTEGRATION.md`

### Investigation Areas

#### 1. Hive Lifecycle
- Document hive spawn by queen
- Document hive registration with queen
- Document heartbeat flow
- Document capabilities discovery

#### 2. SSH Integration
- Document SSH connection establishment
- Document remote hive start
- Document remote command execution
- Document SSH error handling

#### 3. Worker Operations
- Document queen → hive worker requests
- Document worker status reporting
- Document worker lifecycle delegation

#### 4. Heartbeat System
- Document heartbeat frequency
- Document heartbeat payload
- Document staleness detection
- Document hive re-registration

#### 5. Capabilities Flow
- Document capabilities refresh trigger
- Document device detection
- Document capabilities caching
- Document timeout handling

#### 6. Error Propagation
- Document hive unreachable
- Document SSH failures
- Document heartbeat failures
- Document capabilities timeout

#### 7. State Synchronization
- Document hive registry state
- Document hive status tracking
- Document worker status aggregation

---

## TEAM-240: hive-worker Integration

**Components:** `rbee-hive` ↔ `llm-worker-rbee`  
**Complexity:** High  
**Output:** `.plan/TEAM_240_HIVE_WORKER_INTEGRATION.md`

### Investigation Areas

#### 1. Worker Lifecycle
- Document worker spawn by hive
- Document worker registration
- Document worker heartbeat to hive
- Document worker shutdown

#### 2. Model Provisioning
- Document model discovery
- Document model download coordination
- Document model validation
- Document model loading in worker

#### 3. Inference Coordination
- Document inference request routing
- Document response streaming
- Document worker slot management
- Document VRAM allocation

#### 4. Resource Management
- Document GPU assignment
- Document VRAM tracking
- Document worker capacity reporting
- Document resource cleanup

#### 5. Heartbeat System
- Document worker → hive heartbeat
- Document worker status reporting
- Document worker failure detection

#### 6. Error Propagation
- Document worker spawn failures
- Document model load failures
- Document inference failures
- Document VRAM exhaustion

#### 7. State Synchronization
- Document worker registry state
- Document worker status tracking
- Document slot availability

---

## TEAM-241: e2e-inference Flow

**Components:** Full system (keeper → queen → hive → worker)  
**Complexity:** Very High  
**Output:** `.plan/TEAM_241_E2E_INFERENCE_FLOWS.md`

### Investigation Areas

#### 1. Happy Path Flow
Document complete flow:
1. User runs `rbee-keeper inference --prompt "..."`
2. Keeper sends HTTP request to queen
3. Queen generates job_id
4. Queen establishes SSE stream
5. Queen routes to hive
6. Hive routes to worker
7. Worker loads model (if needed)
8. Worker generates tokens
9. Worker streams tokens
10. Tokens flow: worker → hive → queen → SSE → keeper
11. User sees streaming output
12. Inference completes
13. SSE stream closes
14. Job marked complete

#### 2. Narration Flow
Document narration events through system:
- Which components emit narration?
- How does job_id enable SSE routing?
- How do events flow to client?
- What narration goes to stdout vs SSE?

#### 3. Error Scenarios
Document complete error flows:
- Hive not running
- Worker not available
- Model not found
- Model load failure
- VRAM exhaustion
- Network timeout
- Client disconnect

#### 4. State Management
Document distributed state:
- Job state (queen)
- Hive state (queen + hive)
- Worker state (hive + worker)
- Model state (worker)

#### 5. Timeout Handling
Document timeout at each layer:
- Client timeout
- Queen timeout
- Hive timeout
- Worker timeout
- How timeouts propagate

#### 6. Resource Cleanup
Document cleanup flow:
- Normal completion
- Error completion
- Client disconnect
- Timeout expiration

#### 7. Edge Cases
- Multiple concurrent requests
- Worker crash mid-inference
- Hive crash mid-operation
- Queen restart
- Network partitions

---

## Investigation Methodology

### Step 1: Review Component Inventories
Read all inventories from Phases 1-4 to understand components.

### Step 2: Trace Flows
Follow code paths across component boundaries:
- Start at entry point (CLI or HTTP)
- Trace through all components
- Document each hop
- Document transformations

### Step 3: Test Happy Flows
```bash
# Use existing test scripts
bash bin/test_happy_flow.sh
bash bin/test_keeper_queen_sse.sh
```

### Step 4: Test Error Flows
- Kill processes mid-flow
- Inject timeouts
- Simulate network failures
- Document observed behaviors

### Step 5: Document Everything
- Complete flow diagrams
- State machine diagrams
- Error propagation paths
- Data transformations

---

## Deliverables Template

```markdown
# [INTEGRATION] FLOW INVENTORY

**Team:** TEAM-XXX  
**Components:** [component A] ↔ [component B]  
**Date:** [date]

## 1. Happy Path Flows
[Complete end-to-end flows with no errors]

## 2. Data Transformations
[How data changes across boundaries]

## 3. State Synchronization
[How distributed state is managed]

## 4. Error Propagation
[How errors flow across boundaries]

## 5. Timeout Handling
[How timeouts work at each layer]

## 6. Resource Cleanup
[How cleanup happens across components]

## 7. Edge Cases
[All failure scenarios and race conditions]

## 8. Critical Invariants
[What must always be true across system]

## 9. Existing Test Coverage
[Integration tests, E2E tests, gaps]

## 10. Flow Checklist
- [ ] All happy paths documented
- [ ] All error paths documented
- [ ] All state transitions documented
- [ ] All cleanup flows documented
- [ ] All edge cases documented
- [ ] Test coverage gaps identified
```

---

## Deliverables Checklist

Each team must deliver:
- [ ] Integration flow document
- [ ] Follows template structure
- [ ] Max 4 pages (integration is complex)
- [ ] All flows diagrammed
- [ ] All error paths documented
- [ ] All edge cases documented
- [ ] Test coverage gaps identified
- [ ] Code signatures added (`// TEAM-XXX: Investigated`)

---

## Success Criteria

### Per-Team
- ✅ Complete integration flow inventory
- ✅ All happy paths documented
- ✅ All error paths documented
- ✅ All edge cases identified
- ✅ Test gaps identified

### Phase 5
- ✅ All 4 teams completed
- ✅ All integration flows documented
- ✅ Ready for Phase 6 (test planning)

---

## Coordination

### Concurrent Work
- All 4 teams work independently
- No dependencies between teams
- Can start as soon as Phase 4 completes

### Cross-References
- Teams should reference component inventories from Phases 1-4
- Teams should cite specific files and line numbers
- Teams should document assumptions

---

## Critical Notes

### This is the MOST IMPORTANT Phase
- Integration bugs are the hardest to find
- Most test gaps exist at boundaries
- Most production issues are integration issues
- Document EVERYTHING you observe

### Test Thoroughly
- Run existing tests
- Inject failures
- Simulate timeouts
- Document all behaviors

### Document Assumptions
- What do components assume about each other?
- What contracts exist?
- What happens when assumptions break?

---

**Status:** READY (after Phase 4)  
**Next:** Phase 6 (Test planning)
