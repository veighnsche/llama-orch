# Foundation Team - Complete Story List

**Agent**: Foundation-Alpha (Autonomous Development Agent)  
**Total Stories**: 49 stories (47 original + 2 from spec coverage)  
**Total Estimated Effort**: ~87 agent-days (sequential execution)  
**Timeline**: ~87 calendar days (agent works one story at a time)

---

## Sprint 1: HTTP Foundation (5 stories, 9 agent-days)

**Goal**: Establish HTTP server with basic endpoints and SSE streaming

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| FT-001 | HTTP Server Setup | M | 2 | M0-W-1100, M0-W-1320 |
| FT-002 | POST /execute Endpoint (Skeleton) | M | 2 | M0-W-1300, M0-W-1302 |
| FT-003 | SSE Streaming Infrastructure | M | 3 | M0-W-1310, M0-W-1311 |
| FT-004 | Correlation ID Middleware | S | 1 | (Implied in spec) |
| FT-005 | Request Validation Framework | S | 1 | M0-W-1302 |

**Sequential Execution**: Agent completes FT-001 fully before starting FT-002, etc.  
**Timeline**: Days 1-9  
**Blocks**: None (first sprint)

---

## Sprint 2: FFI Layer (7 stories, 13 agent-days) 🔴 CRITICAL

**Goal**: Lock FFI interface to unblock Llama-Beta and GPT-Gamma

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| FT-006 | FFI Interface Definition (C Header) | M | 2 | M0-W-1052 |
| FT-007 | Rust FFI Bindings | M | 2 | M0-W-1052 |
| FT-008 | Error Code System (C++ Side) | S | 1 | M0-W-1501 |
| FT-009 | Error Code to Result Conversion (Rust) | S | 1 | M0-W-1501 |
| FT-010 | CUDA Context Initialization | M | 3 | M0-W-1400, M0-W-1010 |
| FT-011 | VRAM-Only Enforcement | M | 2 | M0-W-1010 |
| FT-012 | FFI Integration Tests | M | 2 | M0-W-1006 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 10-22  
**Critical Milestone**: After FT-007, publish `FFI_INTERFACE_LOCKED.md` to unblock other agents  
**Blocks**: Llama-Beta and GPT-Gamma cannot start until FFI locked (day 15)

---

## Sprint 3: Shared Kernels + Logging (10 stories, 16 agent-days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-013 | DeviceMemory RAII Wrapper | M | 2 | C++ Lead | M0-W-1220 |
| FT-014 | VRAM Residency Verification | M | 2 | C++ Lead | M0-W-1012 |
| FT-015 | Embedding Lookup Kernel | S | 1 | C++ Lead | M0-W-1430 |
| FT-016 | cuBLAS GEMM Wrapper | M | 2 | C++ Lead | M0-W-1430, M0-W-1008 |
| FT-017 | Temperature Scaling Kernel | S | 1 | C++ Lead | M0-W-1032, M0-W-1421 |
| FT-018 | Greedy Sampling (temp=0) | S | 1 | C++ Lead | M0-W-1030, M0-W-1421 |
| FT-019 | Stochastic Sampling (temp>0) | M | 2 | C++ Lead | M0-W-1421 |
| FT-020 | Seeded RNG Implementation | M | 3 | M0-W-1030 |
| FT-049 | Model Load Progress Events | S | 1 | M0-W-1621 |
| FT-050 | Narration-Core Logging Implementation | S | 1 | M0-W-1900 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 23-38  
**Dependencies**: FT-013 (DeviceMemory) needed before FT-015-020 (kernels)  
**Note**: FT-049 and FT-050 added from spec coverage analysis

---

## Sprint 4: Integration & Gate 1 (7 stories, 14 agent-days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-021 | KV Cache Allocation | M | 3 | C++ Lead | M0-W-1010 (implied) |
| FT-022 | KV Cache Management (Update/Free) | M | 2 | C++ Lead | (Implied in spec) |
| FT-023 | Integration Test Framework Setup | M | 3 | DevOps | M0-W-1011 |
| FT-024 | HTTP → FFI → CUDA Integration Test | M | 2 | DevOps | M0-W-1011 |
| FT-025 | Gate 1 Validation Tests | M | 2 | DevOps | Gate 1 criteria |
| FT-026 | Error Handling Integration | M | 2 | Rust Lead | M0-W-1510 |
| FT-027 | **Gate 1 Checkpoint** | - | - | Gate 1 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 39-52  
**Critical Milestone**: Gate 1 validation - Foundation infrastructure complete  
**Blocks**: Integration framework (FT-023) needed by Llama-Beta and GPT-Gamma for their integration tests

---

## Sprint 5: Support & Adapter Prep (5 stories, 8 agent-days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-028 | Support Llama Team Integration | M | 2 | C++ Lead | (Support) |
| FT-029 | Support GPT Team Integration | M | 2 | C++ Lead | (Support) |
| FT-030 | Bug Fixes from Integration | M | 2 | All | (Reactive) |
| FT-031 | Performance Baseline Prep | M | 2 | DevOps | M0-W-1012 (FT-012) |
| FT-032 | **Gate 2 Checkpoint** | - | - | Gate 2 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 53-60  
**Note**: Lower story count - agent available for bug fixes and support as other agents integrate

---

## Sprint 6: Adapter Coordination (6 stories, 11 agent-days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-033 | InferenceAdapter Interface Design | M | 3 | C++ Lead + Teams 2&3 | M0-W-1213 |
| FT-034 | Adapter Factory Pattern | M | 2 | C++ Lead | M0-W-1213 |
| FT-035 | Architecture Detection Integration | M | 2 | C++ Lead | M0-W-1212 |
| FT-036 | Update Integration Tests for Adapters | M | 2 | DevOps | (Testing) |
| FT-037 | API Documentation | M | 2 | Rust Lead | M0-W-1013 (FT-013) |
| FT-038 | **Gate 3 Checkpoint** | - | - | Gate 3 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 61-71  
**Critical**: Adapter pattern must be complete for Llama-Beta and GPT-Gamma to refactor

---

## Sprint 7: Final Integration (9 stories, 16 agent-days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-039 | CI/CD Pipeline Setup | L | 4 | DevOps | M0-W-1014 (FT-014) |
| FT-040 | Performance Baseline Measurements | M | 2 | DevOps | M0-W-1012 (FT-012) |
| FT-041 | All Models Integration Test | M | 3 | DevOps | Gate 4 |
| FT-042 | OOM Recovery Test | M | 2 | DevOps | M0-W-1021 |
| FT-043 | UTF-8 Streaming Edge Cases Test | M | 2 | Rust Lead | M0-W-1312 |
| FT-044 | Cancellation Integration Test | M | 2 | Rust Lead | M0-W-1330 |
| FT-045 | Documentation Complete | S | 1 | All | (Docs) |
| FT-046 | Final Bug Fixes | - | - | All | (Reactive) |
| FT-047 | **Gate 4 Checkpoint (M0 Complete)** | - | - | Gate 4 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 72-87  
**Note**: Foundation-Alpha finishes before GPT-Gamma (who is on critical path)

---

## Agent Execution Reality

### ✅ **NO OVERCOMMITMENT ISSUE**

**Reality**: Foundation-Alpha is a single agent working sequentially, not a team with parallel capacity.

**Agent Characteristics**:
- Works sequentially through all 49 stories
- Completes each story fully before moving to next
- Can work on multiple files simultaneously within a story
- No parallel work across stories
- No "overcommitment" - agent works until done

**Timeline**: 87 agent-days = 87 calendar days (assuming full-time work)

**Critical Path Position**: Foundation-Alpha (87 days) is NOT the critical path - GPT-Gamma (102 days) is longer

### Key Milestones

**Day 15 (After FT-007)**: FFI Interface Locked
- **Critical**: Unblocks Llama-Beta and GPT-Gamma
- **Action**: Publish `FFI_INTERFACE_LOCKED.md`
- **Impact**: Other agents idle until this point

**Day 52 (After FT-027)**: Gate 1 Complete
- **Validation**: Foundation infrastructure complete
- **Enables**: Integration testing for other agents

**Day 71 (After FT-038)**: Gate 3 Complete
- **Validation**: Adapter pattern complete
- **Enables**: Other agents refactor to use adapters

**Day 87 (After FT-047)**: Foundation-Alpha Complete
- **Note**: GPT-Gamma still working (finishes day 102)

---

## Dependency Analysis

### Sequential Execution Chain

```
Day 1-9: HTTP Foundation (FT-001 → FT-005)
  ↓
Day 10-15: FFI Interface (FT-006 → FT-007) ← LOCK POINT
  ↓ (Llama-Beta and GPT-Gamma start here)
Day 16-22: Error Handling + CUDA (FT-008 → FT-012)
  ↓
Day 23-38: Shared Kernels + Logging (FT-013 → FT-050)
  ↓
Day 39-52: Integration + Gate 1 (FT-021 → FT-027)
  ↓
Day 53-60: Support & Prep (FT-028 → FT-032)
  ↓
Day 61-71: Adapter Pattern + Gate 3 (FT-033 → FT-038)
  ↓
Day 72-87: Final Integration + Gate 4 (FT-039 → FT-047)
```

**Total Duration**: 87 agent-days

**Critical Dependencies**:
- **Day 15**: FFI lock blocks Llama-Beta and GPT-Gamma
- **Day 52**: Integration framework blocks other agents' integration tests
- **Day 71**: Adapter pattern blocks other agents' refactoring

**No "Slack Time"**: Agent works sequentially until done - concept of utilization doesn't apply

---

## Risk Register (Revised for AI Agent Reality)

### High Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| FFI lock delayed beyond day 15 | Llama-Beta and GPT-Gamma blocked | Prioritize FT-006, FT-007; publish lock document |
| FFI interface changes after lock | Other agents must refactor | Absolute lock after day 15 - no changes allowed |
| Story estimates incorrect | Timeline extends | Accept reality - agent works until done |

### Medium Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUDA context bugs | Integration delays | Valgrind tests, VRAM tracking from day 1 |
| SSE UTF-8 edge cases | Client-facing bugs | Comprehensive test vectors, fuzzing |
| Integration framework delays | Blocks other agents' tests | Prioritize FT-023 in Sprint 4 |

### ❌ Not Risks (Human Team Assumptions)

- ~~"Week 7 overcommitted"~~ - Agent works sequentially, no overcommitment
- ~~"Need 4th person"~~ - Cannot scale agent count
- ~~"Team burnout"~~ - Not applicable to AI agent

---

## Key Actions for Foundation-Alpha

### 1. **Prioritize FFI Lock (Day 15)** 🔴 CRITICAL

**Action**: Complete FT-006 and FT-007 by day 15
- Publish `FFI_INTERFACE_LOCKED.md` immediately after FT-007
- Include C header, Rust bindings, and usage examples
- Notify Llama-Beta and GPT-Gamma to begin work

### 2. **Maintain Sequential Execution**

**Reality**: Agent completes one story fully before next
- No parallel work across stories
- Can work on multiple files within a story
- Timeline is 87 agent-days, period

### 3. **Document Interface Decisions**

**Action**: Create clear documentation for other agents
- FFI interface patterns
- Error handling conventions
- Shared kernel usage examples
- Integration test patterns

---

## Timeline Summary

**Total Duration**: 87 agent-days  
**Critical Path**: GPT-Gamma (102 days) - Foundation-Alpha finishes first  
**Key Milestone**: Day 15 FFI lock (unblocks other agents)

**No "weeks" or "sprints"** - these are organizational conveniences. Agent works sequentially through 49 stories.

---

**Status**: ✅ Revised for AI Agent Reality  
**Next Action**: Begin FT-001 when ready  
**Owner**: Foundation-Alpha

---

*Built by Foundation-Alpha 🏗️*
