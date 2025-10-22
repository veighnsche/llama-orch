# BEHAVIOR DISCOVERY & TEST FREEZE - INDEX

**Mission:** Inventory ALL behaviors in `/bin/` and freeze them with comprehensive test coverage.

**Status:** Phase 1 READY TO START  
**Total Teams:** 50+  
**Total Duration:** ~25 days

---

## Quick Navigation

### Planning Documents
- **Master Plan:** [BEHAVIOR_DISCOVERY_MASTER_PLAN.md](BEHAVIOR_DISCOVERY_MASTER_PLAN.md) - Complete strategy
- **Quick Reference:** [QUICK_REFERENCE_TESTING_PLAN.md](QUICK_REFERENCE_TESTING_PLAN.md) - Summary
- **This Index:** [00_INDEX.md](00_INDEX.md) - Navigation hub

### Phase Guides
- **Phase 1:** [PHASE_1_START_HERE.md](PHASE_1_START_HERE.md) - Main binaries (READY NOW ✅)
- **Phase 2:** [PHASE_2_GUIDES.md](PHASE_2_GUIDES.md) - Queen crates
- **Phase 3:** [PHASE_3_GUIDES.md](PHASE_3_GUIDES.md) - Hive crates
- **Phase 4:** [PHASE_4_GUIDES.md](PHASE_4_GUIDES.md) - Shared crates
- **Phase 5:** [PHASE_5_GUIDES.md](PHASE_5_GUIDES.md) - Integration flows

---

## Phase 1: Main Binaries (READY TO START)

**Duration:** 1 day (4 teams concurrent)

| Team | Component | Guide | Output |
|------|-----------|-------|--------|
| TEAM-216 | rbee-keeper | [TEAM_216_GUIDE.md](TEAM_216_GUIDE.md) | TEAM_216_RBEE_KEEPER_BEHAVIORS.md |
| TEAM-217 | queen-rbee | [TEAM_217_GUIDE.md](TEAM_217_GUIDE.md) | TEAM_217_QUEEN_RBEE_BEHAVIORS.md |
| TEAM-218 | rbee-hive | [TEAM_218_GUIDE.md](TEAM_218_GUIDE.md) | TEAM_218_RBEE_HIVE_BEHAVIORS.md |
| TEAM-219 | llm-worker-rbee | [TEAM_219_GUIDE.md](TEAM_219_GUIDE.md) | TEAM_219_LLM_WORKER_BEHAVIORS.md |

**Start Here:** [PHASE_1_START_HERE.md](PHASE_1_START_HERE.md)

---

## Phase 2: Queen Crates

**Duration:** 1 day (3 teams concurrent)  
**Depends On:** Phase 1 complete

| Team | Component | Output |
|------|-----------|--------|
| TEAM-220 | hive-lifecycle | TEAM_220_HIVE_LIFECYCLE_BEHAVIORS.md |
| TEAM-221 | hive-registry | TEAM_221_HIVE_REGISTRY_BEHAVIORS.md |
| TEAM-222 | ssh-client | TEAM_222_SSH_CLIENT_BEHAVIORS.md |

**Guide:** [PHASE_2_GUIDES.md](PHASE_2_GUIDES.md)

---

## Phase 3: Hive Crates

**Duration:** 1 day (7 teams concurrent)  
**Depends On:** Phase 2 complete

| Team | Component | Output |
|------|-----------|--------|
| TEAM-223 | device-detection | TEAM_223_DEVICE_DETECTION_BEHAVIORS.md |
| TEAM-224 | download-tracker | TEAM_224_DOWNLOAD_TRACKER_BEHAVIORS.md |
| TEAM-225 | model-catalog | TEAM_225_MODEL_CATALOG_BEHAVIORS.md |
| TEAM-226 | model-provisioner | TEAM_226_MODEL_PROVISIONER_BEHAVIORS.md |
| TEAM-227 | monitor | TEAM_227_MONITOR_BEHAVIORS.md |
| TEAM-228 | vram-checker | TEAM_228_VRAM_CHECKER_BEHAVIORS.md |
| TEAM-229 | worker-management | TEAM_229_WORKER_MANAGEMENT_BEHAVIORS.md |

**Guide:** [PHASE_3_GUIDES.md](PHASE_3_GUIDES.md)

---

## Phase 4: Shared Crates

**Duration:** 1 day (9 teams concurrent)  
**Depends On:** Phase 3 complete

| Team | Component | Output |
|------|-----------|--------|
| TEAM-230 | narration-core + narration-macros | TEAM_230_NARRATION_BEHAVIORS.md |
| TEAM-231 | daemon-lifecycle | TEAM_231_DAEMON_LIFECYCLE_BEHAVIORS.md |
| TEAM-232 | rbee-http-client | TEAM_232_HTTP_CLIENT_BEHAVIORS.md |
| TEAM-233 | rbee-config + rbee-operations | TEAM_233_CONFIG_OPERATIONS_BEHAVIORS.md |
| TEAM-234 | job-registry + deadline-propagation | TEAM_234_JOB_DEADLINE_BEHAVIORS.md |
| TEAM-235 | auth-min + jwt-guardian | TEAM_235_AUTH_JWT_BEHAVIORS.md |
| TEAM-236 | audit-logging + input-validation | TEAM_236_AUDIT_VALIDATION_BEHAVIORS.md |
| TEAM-237 | heartbeat + auto-update + timeout-enforcer | TEAM_237_HEARTBEAT_UPDATE_BEHAVIORS.md |
| TEAM-238 | secrets-management + sse-relay + model-catalog | TEAM_238_SECRETS_SSE_MODEL_BEHAVIORS.md |

**Guide:** [PHASE_4_GUIDES.md](PHASE_4_GUIDES.md)

---

## Phase 5: Integration Flows

**Duration:** 1 day (4 teams concurrent)  
**Depends On:** Phase 4 complete

| Team | Flow | Output |
|------|------|--------|
| TEAM-239 | keeper ↔ queen | TEAM_239_KEEPER_QUEEN_INTEGRATION.md |
| TEAM-240 | queen ↔ hive | TEAM_240_QUEEN_HIVE_INTEGRATION.md |
| TEAM-241 | hive ↔ worker | TEAM_241_HIVE_WORKER_INTEGRATION.md |
| TEAM-242 | End-to-end inference | TEAM_242_E2E_INFERENCE_FLOWS.md |

**Guide:** [PHASE_5_GUIDES.md](PHASE_5_GUIDES.md)

---

## Phase 6: Test Planning

**Duration:** 3 days (~8 teams)  
**Depends On:** Phase 5 complete  
**Input:** 27 behavior inventory documents  
**Output:** Comprehensive test plans

### Test Plan Types
1. **Unit Test Plans** - Per-crate unit tests
2. **BDD Test Plans** - Gherkin scenarios for user-facing behaviors
3. **Integration Test Plans** - Cross-crate integration tests
4. **E2E Test Plans** - Full system flows via xtask

**Teams:** TEAM-243 through TEAM-250 (TBD)

---

## Phase 7: Test Implementation

**Duration:** 15 days (~15 teams)  
**Depends On:** Phase 6 complete  
**Input:** Test plans from Phase 6  
**Output:** Full test suite

### Test Types
1. **Unit Tests** - Freeze function-level behaviors
2. **BDD Tests** - Freeze user-facing behaviors
3. **Integration Tests** - Freeze component interactions
4. **E2E Tests** - Freeze end-to-end flows

**Teams:** TEAM-251 through TEAM-265 (TBD)

---

## Timeline

```
Week 1: Discovery
├─ Day 1: Phase 1 (Main binaries)
├─ Day 2: Phase 2 (Queen crates)
├─ Day 3: Phase 3 (Hive crates)
├─ Day 4: Phase 4 (Shared crates)
└─ Day 5: Phase 5 (Integration flows)

Week 2: Test Planning
└─ Days 6-8: Phase 6 (Create test plans)

Weeks 3-5: Test Implementation
└─ Days 9-23: Phase 7 (Implement tests)
```

**Total:** ~25 days to freeze all behaviors

---

## Team Count Summary

| Phase | Teams | Components | Type |
|-------|-------|------------|------|
| 1 | 4 | Main binaries | Discovery |
| 2 | 3 | Queen crates | Discovery |
| 3 | 7 | Hive crates | Discovery |
| 4 | 9 | Shared crates | Discovery |
| 5 | 4 | Integration flows | Discovery |
| **Total Discovery** | **27** | **All components** | **5 days** |
| 6 | ~8 | Test plans | Planning |
| 7 | ~15 | Test implementation | Implementation |
| **Grand Total** | **~50** | **Full test suite** | **~25 days** |

---

## Document Structure

### Discovery Documents (Phases 1-5)
All behavior inventories follow this structure:

1. **Public API Surface** - Functions, endpoints, CLI commands
2. **State Machine Behaviors** - States, transitions, lifecycle
3. **Data Flows** - Inputs, outputs, transformations
4. **Error Handling** - Error types, propagation, recovery
5. **Integration Points** - Dependencies, dependents, contracts
6. **Critical Invariants** - What must always be true
7. **Existing Test Coverage** - Unit, BDD, integration tests
8. **Behavior Checklist** - Verification of completeness

**Max Length:** 3 pages (4 pages for Phase 5 integration flows)

### Integration Documents (Phase 5)
Integration flow documents include:

1. **Happy Path Flows** - Complete end-to-end flows
2. **Data Transformations** - Cross-boundary data changes
3. **State Synchronization** - Distributed state management
4. **Error Propagation** - Cross-boundary error flows
5. **Timeout Handling** - Multi-layer timeout behavior
6. **Resource Cleanup** - Cross-component cleanup
7. **Edge Cases** - Failure scenarios and race conditions
8. **Critical Invariants** - System-wide invariants
9. **Existing Test Coverage** - Integration/E2E test gaps
10. **Flow Checklist** - Verification of completeness

---

## Success Metrics

### After Phase 5 (Discovery)
- ✅ 27 behavior inventory documents
- ✅ Complete understanding of ALL behaviors
- ✅ All test coverage gaps identified
- ✅ Ready for test planning

### After Phase 6 (Test Planning)
- ✅ Comprehensive unit test plans
- ✅ Comprehensive BDD test plans
- ✅ Comprehensive integration test plans
- ✅ Comprehensive E2E test plans

### After Phase 7 (Test Implementation)
- ✅ Full test suite implemented
- ✅ All behaviors frozen
- ✅ 80%+ code coverage
- ✅ All critical flows covered
- ✅ No regressions possible

---

## Critical Rules

### Discovery Phase (Phases 1-5)
- ❌ NO code changes during discovery
- ✅ Document existing behaviors only
- ✅ Identify test coverage gaps
- ✅ Add code signatures (`// TEAM-XXX: Investigated`)
- ✅ Follow template exactly
- ✅ Max 3 pages per document (4 for Phase 5)
- ✅ No TODO markers

### Test Planning Phase (Phase 6)
- ✅ Create test plans from inventories
- ✅ Cover all discovered behaviors
- ✅ Include all edge cases
- ✅ Include all error paths
- ✅ Prioritize by criticality

### Test Implementation Phase (Phase 7)
- ✅ Implement all planned tests
- ✅ Verify all behaviors
- ✅ Freeze all behaviors
- ✅ No regressions possible
- ✅ Document test outputs

---

## Getting Started

### For Phase 1 Teams (START HERE)
1. Read [PHASE_1_START_HERE.md](PHASE_1_START_HERE.md)
2. Read your team guide:
   - TEAM-216: [TEAM_216_GUIDE.md](TEAM_216_GUIDE.md)
   - TEAM-217: [TEAM_217_GUIDE.md](TEAM_217_GUIDE.md)
   - TEAM-218: [TEAM_218_GUIDE.md](TEAM_218_GUIDE.md)
   - TEAM-219: [TEAM_219_GUIDE.md](TEAM_219_GUIDE.md)
3. Investigate your component
4. Document all behaviors
5. Deliver inventory document

### For Later Phases
- Wait for your phase to start
- Read the phase guide
- Follow the investigation methodology
- Deliver your inventory document

---

## Resources

### Documentation
- [BEHAVIOR_DISCOVERY_MASTER_PLAN.md](BEHAVIOR_DISCOVERY_MASTER_PLAN.md) - Full strategy
- [QUICK_REFERENCE_TESTING_PLAN.md](QUICK_REFERENCE_TESTING_PLAN.md) - Quick summary
- Phase guides (PHASE_1 through PHASE_5)
- Team guides (TEAM_216 through TEAM_241)

### Engineering Standards
- `/home/vince/Projects/llama-orch/.windsurf/rules/engineering-rules.md`
- `/home/vince/Projects/llama-orch/.windsurf/rules/debugging-rules.md`

### Existing Tests
- `test-harness/bdd/` - BDD tests
- Per-crate tests in `*/tests/`
- Per-crate BDD in `*/bdd/`

---

## Questions?

- **Overall strategy:** [BEHAVIOR_DISCOVERY_MASTER_PLAN.md](BEHAVIOR_DISCOVERY_MASTER_PLAN.md)
- **Quick summary:** [QUICK_REFERENCE_TESTING_PLAN.md](QUICK_REFERENCE_TESTING_PLAN.md)
- **Your phase:** [PHASE_X_GUIDES.md or PHASE_X_START_HERE.md]
- **Your team:** [TEAM_XXX_GUIDE.md]

---

## Status Tracking

### Discovery Phases

| Phase | Status | Teams | Expected Completion |
|-------|--------|-------|---------------------|
| Phase 1 | ✅ READY | 4 | TBD |
| Phase 2 | ⏳ Waiting | 3 | After Phase 1 |
| Phase 3 | ⏳ Waiting | 7 | After Phase 2 |
| Phase 4 | ⏳ Waiting | 8 | After Phase 3 |
| Phase 5 | ⏳ Waiting | 4 | After Phase 4 |

### Planning & Implementation Phases

| Phase | Status | Teams | Expected Completion |
|-------|--------|-------|---------------------|
| Phase 6 | ⏳ Waiting | ~8 | After Phase 5 |
| Phase 7 | ⏳ Waiting | ~15 | After Phase 6 |

---

## Expected Deliverables

### Discovery (Phases 1-5)
- 26 behavior inventory documents
- All behaviors documented
- All test gaps identified

### Planning (Phase 6)
- Unit test plan
- BDD test plan
- Integration test plan
- E2E test plan

### Implementation (Phase 7)
- Complete unit test suite
- Complete BDD test suite
- Complete integration test suite
- Complete E2E test suite
- 80%+ code coverage
- All behaviors frozen

---

**Current Phase:** Phase 1  
**Current Status:** READY TO START  
**Next Action:** TEAM-216, 217, 218, 219 begin investigation

---

**Last Updated:** Oct 22, 2025  
**Version:** 1.0  
**Owner:** Multi-team effort (TEAM-216 through TEAM-264+)
