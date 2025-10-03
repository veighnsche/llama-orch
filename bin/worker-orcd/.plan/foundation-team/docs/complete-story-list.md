# Foundation Team - Complete Story List

**Total Stories**: 47 stories across 7 weeks  
**Total Estimated Effort**: ~85 days (with 3-4 people = realistic for 7 weeks)

---

## Week 1: HTTP Foundation (5 stories, 9 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-001 | HTTP Server Setup | M | 2 | Rust Lead | M0-W-1100, M0-W-1320 |
| FT-002 | POST /execute Endpoint (Skeleton) | M | 2 | Rust Lead | M0-W-1300, M0-W-1302 |
| FT-003 | SSE Streaming Infrastructure | M | 3 | Rust Lead | M0-W-1310, M0-W-1311 |
| FT-004 | Correlation ID Middleware | S | 1 | Rust Lead | (Implied in spec) |
| FT-005 | Request Validation Framework | S | 1 | Rust Lead | M0-W-1302 |

**Week 1 Capacity**: 3 people × 5 days = 15 days available, 9 days committed (60% utilization - good for first sprint)

---

## Week 2: FFI Layer (7 stories, 13 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-006 | FFI Interface Definition (C Header) | M | 2 | C++ Lead | M0-W-1052 |
| FT-007 | Rust FFI Bindings | M | 2 | Rust Lead | M0-W-1052 |
| FT-008 | Error Code System (C++ Side) | S | 1 | C++ Lead | M0-W-1501 |
| FT-009 | Error Code to Result Conversion (Rust) | S | 1 | Rust Lead | M0-W-1501 |
| FT-010 | CUDA Context Initialization | M | 3 | C++ Lead | M0-W-1400, M0-W-1010 |
| FT-011 | VRAM-Only Enforcement | M | 2 | C++ Lead | M0-W-1010 |
| FT-012 | FFI Integration Tests | M | 2 | DevOps | M0-W-1006 |

**Week 2 Capacity**: 15 days, 13 days committed (87% utilization)

---

## Week 3: Shared Kernels (8 stories, 14 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-013 | DeviceMemory RAII Wrapper | M | 2 | C++ Lead | M0-W-1220 |
| FT-014 | VRAM Residency Verification | M | 2 | C++ Lead | M0-W-1012 |
| FT-015 | Embedding Lookup Kernel | S | 1 | C++ Lead | M0-W-1430 |
| FT-016 | cuBLAS GEMM Wrapper | M | 2 | C++ Lead | M0-W-1430, M0-W-1008 |
| FT-017 | Temperature Scaling Kernel | S | 1 | C++ Lead | M0-W-1032, M0-W-1421 |
| FT-018 | Greedy Sampling (temp=0) | S | 1 | C++ Lead | M0-W-1030, M0-W-1421 |
| FT-019 | Stochastic Sampling (temp>0) | M | 2 | C++ Lead | M0-W-1421 |
| FT-020 | Seeded RNG Implementation | M | 3 | C++ Lead | M0-W-1030 |

**Week 3 Capacity**: 15 days, 14 days committed (93% utilization)

---

## Week 4: Integration & Gate 1 (7 stories, 14 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-021 | KV Cache Allocation | M | 3 | C++ Lead | M0-W-1010 (implied) |
| FT-022 | KV Cache Management (Update/Free) | M | 2 | C++ Lead | (Implied in spec) |
| FT-023 | Integration Test Framework Setup | M | 3 | DevOps | M0-W-1011 |
| FT-024 | HTTP → FFI → CUDA Integration Test | M | 2 | DevOps | M0-W-1011 |
| FT-025 | Gate 1 Validation Tests | M | 2 | DevOps | Gate 1 criteria |
| FT-026 | Error Handling Integration | M | 2 | Rust Lead | M0-W-1510 |
| FT-027 | **Gate 1 Checkpoint** | - | - | Team Lead | Gate 1 |

**Week 4 Capacity**: 15 days, 14 days committed (93% utilization)  
**Critical**: Gate 1 must pass or Week 5 blocked

---

## Week 5: Support Role (5 stories, 8 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-028 | Support Llama Team Integration | M | 2 | C++ Lead | (Support) |
| FT-029 | Support GPT Team Integration | M | 2 | C++ Lead | (Support) |
| FT-030 | Bug Fixes from Integration | M | 2 | All | (Reactive) |
| FT-031 | Performance Baseline Prep | M | 2 | DevOps | M0-W-1012 (FT-012) |
| FT-032 | **Gate 2 Checkpoint** | - | - | Team Lead | Gate 2 |

**Week 5 Capacity**: 15 days, 8 days committed (53% utilization - intentional, support role)

---

## Week 6: Adapter Coordination (6 stories, 11 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| FT-033 | InferenceAdapter Interface Design | M | 3 | C++ Lead + Teams 2&3 | M0-W-1213 |
| FT-034 | Adapter Factory Pattern | M | 2 | C++ Lead | M0-W-1213 |
| FT-035 | Architecture Detection Integration | M | 2 | C++ Lead | M0-W-1212 |
| FT-036 | Update Integration Tests for Adapters | M | 2 | DevOps | (Testing) |
| FT-037 | API Documentation | M | 2 | Rust Lead | M0-W-1013 (FT-013) |
| FT-038 | **Gate 3 Checkpoint** | - | - | Team Lead | Gate 3 |

**Week 6 Capacity**: 15 days, 11 days committed (73% utilization)

---

## Week 7: Final Integration (9 stories, 16 days)

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
| FT-047 | **Gate 4 Checkpoint (M0 Complete)** | - | - | Team Lead | Gate 4 |

**Week 7 Capacity**: 15 days, 16 days committed (107% utilization - **OVERCOMMITTED**)

---

## Planning Gap Analysis

### ⚠️ **CRITICAL FINDING: Week 7 Overcommitted**

**Problem**: Week 7 has 16 days of work committed but only 15 days available (107% utilization).

**Impact**: 
- Risk of Gate 4 failure
- M0 delivery delayed
- Team burnout

**Options**:

#### Option A: Extend to 8 Weeks (Recommended)
- Move FT-039 (CI/CD) to Week 6 (can start earlier)
- Move FT-040 (Performance baseline) to Week 6
- Week 7 becomes 12 days (80% utilization)
- **Total timeline**: 8 weeks instead of 7

#### Option B: Reduce Scope
- Make CI/CD optional (manual testing for M0)
- Defer performance baseline to M1
- Week 7 becomes 10 days (67% utilization)
- **Risk**: Lower quality M0

#### Option C: Add Developer
- Add 4th developer for Week 7 only
- Week 7 capacity becomes 20 days
- 16 days / 20 days = 80% utilization
- **Cost**: Additional resource

### 🟡 **MEDIUM FINDING: Week 5 Underutilized**

**Observation**: Week 5 only 53% utilized (support role)

**Opportunity**: 
- Pull forward some Week 6 work (API docs, adapter design prep)
- Start CI/CD setup early
- More buffer for unexpected issues

**Recommendation**: Use Week 5 slack time as buffer, don't add more work

### 🟢 **POSITIVE: Weeks 2-4 Well Balanced**

**Observation**: Weeks 2-4 are 87-93% utilized

**Analysis**: Good balance, leaves room for:
- Unexpected bugs
- Integration issues
- Learning curve

---

## Dependency Analysis

### Critical Path

```
Week 1: HTTP Server (FT-001)
  ↓
Week 1: Execute Endpoint (FT-002)
  ↓
Week 2: FFI Interface (FT-006, FT-007)
  ↓
Week 2: CUDA Context (FT-010)
  ↓
Week 3: Shared Kernels (FT-015, FT-016, FT-017, FT-018)
  ↓
Week 4: Integration Tests (FT-023, FT-024)
  ↓
Week 4: **GATE 1** (FT-027) ← CRITICAL MILESTONE
  ↓
Week 5: Support Teams 2 & 3 (FT-028, FT-029)
  ↓
Week 6: Adapter Pattern (FT-033, FT-034)
  ↓
Week 6: **GATE 3** (FT-038)
  ↓
Week 7: Final Integration (FT-041)
  ↓
Week 7: **GATE 4** (FT-047) ← M0 COMPLETE
```

**Critical Path Duration**: 7 weeks (minimum)

**Slack Time**: 
- Week 1: 6 days slack (60% util)
- Week 5: 7 days slack (53% util)
- **Total slack**: 13 days

**Analysis**: With 13 days of slack, we can absorb some delays, but Week 7 overcommitment is still a risk.

---

## Risk Register

### High Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FFI interface changes after Week 2 | Blocks Teams 2 & 3 | Medium | Lock interface by Week 2 end, no changes |
| Gate 1 failure (Week 4) | Entire project delayed | Low | Weekly integration tests, early detection |
| Week 7 overcommitment | Gate 4 failure, M0 delay | **HIGH** | **Extend to 8 weeks (Option A)** |

### Medium Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| CUDA context bugs | Integration delays | Medium | Valgrind tests, VRAM tracking from day 1 |
| SSE UTF-8 edge cases | Client-facing bugs | Medium | Comprehensive test vectors, fuzzing |
| KV cache complexity | Week 4 delays | Medium | Start early (Week 3 if possible) |

---

## Recommendations

### 1. **Extend Timeline to 8 Weeks** (Critical)

**Rationale**: Week 7 is overcommitted (107%). This is not sustainable.

**Action**:
- Move FT-039 (CI/CD) to Week 6
- Move FT-040 (Performance baseline) to Week 6
- Week 7 becomes focused on final integration and bug fixes

**New Timeline**:
- Weeks 1-6: As planned
- Week 7: Integration + testing (12 days, 80% util)
- Week 8: Buffer + Gate 4 (if needed)

### 2. **Lock FFI Interface by Week 2 End** (Critical)

**Rationale**: Any changes after Week 2 block Teams 2 & 3.

**Action**:
- FT-006 (FFI interface) must be reviewed by all teams
- Sign-off meeting Friday Week 2
- Document in `interfaces.md` with "LOCKED" status

### 3. **Start CI/CD in Week 5** (Recommended)

**Rationale**: Week 5 has slack time (53% util), CI/CD is on critical path.

**Action**:
- Move FT-039 (CI/CD) from Week 7 to Week 5-6 (split across 2 weeks)
- Reduces Week 7 pressure

### 4. **Add Integration Tests Early** (Recommended)

**Rationale**: Catch issues before Gate 1.

**Action**:
- Add smoke test in Week 2 (HTTP → FFI → CUDA context init)
- Don't wait until Week 4 for first integration test

---

## Revised Timeline (8 Weeks)

| Week | Stories | Days | Util % | Gate |
|------|---------|------|--------|------|
| 1 | 5 | 9 | 60% | - |
| 2 | 7 | 13 | 87% | - |
| 3 | 8 | 14 | 93% | - |
| 4 | 7 | 14 | 93% | Gate 1 ✓ |
| 5 | 5 + CI start | 11 | 73% | Gate 2 ✓ |
| 6 | 6 + CI + Perf | 15 | 100% | Gate 3 ✓ |
| 7 | 6 (integration) | 12 | 80% | - |
| 8 | 3 (buffer + Gate 4) | 5 | 33% | Gate 4 ✓ |

**Total**: 8 weeks, 93 days of work, 120 days available (3 people × 8 weeks × 5 days)

**Utilization**: 93 / 120 = 78% (healthy for project with unknowns)

---

## Next Actions

1. **Immediate**: Review this analysis with team leads
2. **Immediate**: Decide on 7 weeks (risky) vs 8 weeks (recommended)
3. **Week 0**: Create all 47 story cards in `stories/backlog/`
4. **Week 0**: Story sizing workshop (validate estimates)
5. **Week 1**: Sprint planning, commit to first 5 stories

---

**Status**: ✅ Analysis Complete  
**Recommendation**: **Extend to 8 weeks**  
**Next Action**: Stakeholder decision on timeline
