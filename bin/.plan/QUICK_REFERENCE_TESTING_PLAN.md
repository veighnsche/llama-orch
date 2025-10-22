# TESTING PLAN QUICK REFERENCE

**Mission:** Freeze ALL current behaviors with comprehensive test coverage

---

## Plan Overview

```
Discovery (Phases 1-5) → Test Planning (Phase 6) → Test Implementation (Phase 7)
     26 teams                  ~8 teams                    ~15 teams
     5 days                    3 days                      15 days
```

**Total:** ~50 teams, ~25 days

---

## Discovery Phases (TEAM-216 to TEAM-241)

### Phase 1: Main Binaries (4 teams, 1 day)
- **TEAM-216:** rbee-keeper (CLI)
- **TEAM-217:** queen-rbee (daemon)
- **TEAM-218:** rbee-hive (daemon)
- **TEAM-219:** llm-worker-rbee (daemon)

### Phase 2: Queen Crates (3 teams, 1 day)
- **TEAM-220:** hive-lifecycle
- **TEAM-221:** hive-registry
- **TEAM-222:** ssh-client

### Phase 3: Hive Crates (7 teams, 1 day)
- **TEAM-223:** device-detection
- **TEAM-224:** download-tracker
- **TEAM-225:** model-catalog
- **TEAM-226:** model-provisioner
- **TEAM-227:** monitor
- **TEAM-228:** vram-checker
- **TEAM-229:** worker-catalog + worker-lifecycle + worker-registry

### Phase 4: Shared Crates (8 teams, 1 day)
- **TEAM-230:** narration-core + narration-macros
- **TEAM-231:** daemon-lifecycle
- **TEAM-232:** rbee-http-client
- **TEAM-233:** rbee-config + rbee-operations
- **TEAM-234:** job-registry + deadline-propagation
- **TEAM-235:** auth-min + jwt-guardian
- **TEAM-236:** audit-logging + input-validation
- **TEAM-237:** heartbeat + auto-update + hive-core

### Phase 5: Integration Flows (4 teams, 1 day)
- **TEAM-238:** keeper → queen flows
- **TEAM-239:** queen → hive flows
- **TEAM-240:** hive → worker flows
- **TEAM-241:** end-to-end inference flows

---

## Test Planning Phase (TEAM-242+)

**Input:** 26 behavior inventory documents  
**Output:** Comprehensive test plans  
**Duration:** 3 days

### Test Plan Types
1. **Unit Test Plans** - Per-crate unit tests
2. **BDD Test Plans** - Gherkin scenarios
3. **Integration Test Plans** - Cross-crate integration
4. **E2E Test Plans** - Full system flows via xtask

---

## Test Implementation Phase (TEAM-250+)

**Input:** Test plans from Phase 6  
**Output:** Full test suite  
**Duration:** 15 days

### Test Types
1. **Unit Tests** - Freeze function behaviors
2. **BDD Tests** - Freeze user-facing behaviors
3. **Integration Tests** - Freeze component interactions
4. **E2E Tests** - Freeze end-to-end flows

---

## Deliverables Per Team (Phases 1-5)

### Required Document Structure
1. Public API Surface
2. State Machine Behaviors
3. Data Flows
4. Error Handling
5. Integration Points
6. Critical Invariants
7. Existing Test Coverage
8. Behavior Checklist

### Quality Gates
- [ ] Max 3 pages
- [ ] Follows template
- [ ] All behaviors documented
- [ ] All edge cases identified
- [ ] Test coverage gaps identified
- [ ] Code signatures added (`// TEAM-XXX: Investigated`)
- [ ] No TODO markers

---

## Key Documents

### Master Plan
- `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md` - Complete strategy

### Phase 1 (Ready Now)
- `.plan/PHASE_1_START_HERE.md` - Phase 1 overview
- `.plan/TEAM_216_GUIDE.md` - rbee-keeper guide
- `.plan/TEAM_217_GUIDE.md` - queen-rbee guide
- `.plan/TEAM_218_GUIDE.md` - rbee-hive guide
- `.plan/TEAM_219_GUIDE.md` - llm-worker-rbee guide

### Future Phases (To Be Created)
- `.plan/PHASE_2_START_HERE.md`
- `.plan/TEAM_220_GUIDE.md` through `.plan/TEAM_241_GUIDE.md`

---

## Timeline

### Week 1: Discovery
- **Day 1:** Phase 1 (main binaries)
- **Day 2:** Phase 2 (queen crates)
- **Day 3:** Phase 3 (hive crates)
- **Day 4:** Phase 4 (shared crates)
- **Day 5:** Phase 5 (integration flows)

### Week 2: Test Planning
- **Days 6-8:** Create comprehensive test plans

### Weeks 3-5: Test Implementation
- **Days 9-23:** Implement all tests

---

## Success Metrics

### Coverage Targets
- **Unit Tests:** 80%+ code coverage per crate
- **BDD Tests:** All user-facing scenarios covered
- **Integration Tests:** All component interactions covered
- **E2E Tests:** All critical user flows covered

### Behavioral Freeze
- ✅ All current behaviors documented
- ✅ All current behaviors tested
- ✅ No regressions possible without test failures

---

## Getting Started

### For Phase 1 Teams (TEAM-216 to TEAM-219)
1. Read `.plan/PHASE_1_START_HERE.md`
2. Read your specific guide (`.plan/TEAM_XXX_GUIDE.md`)
3. Investigate your component
4. Document all behaviors
5. Deliver inventory document

### For Later Teams
- Wait for your phase to start
- Guides will be created as phases progress

---

## Critical Rules

### Discovery Phase (Phases 1-5)
- ❌ NO code changes
- ✅ Document existing behaviors
- ✅ Identify test gaps
- ✅ Add code signatures
- ✅ Max 3 pages per document

### Test Planning Phase (Phase 6)
- ✅ Create test plans from inventories
- ✅ Cover all discovered behaviors
- ✅ Include edge cases
- ✅ Include error paths

### Test Implementation Phase (Phase 7)
- ✅ Implement all planned tests
- ✅ Verify all behaviors
- ✅ Freeze all behaviors
- ✅ No regressions possible

---

## Team Count by Phase

| Phase | Teams | Components | Days |
|-------|-------|------------|------|
| 1 | 4 | Main binaries | 1 |
| 2 | 3 | Queen crates | 1 |
| 3 | 7 | Hive crates | 1 |
| 4 | 8 | Shared crates | 1 |
| 5 | 4 | Integration flows | 1 |
| **Total Discovery** | **26** | **All components** | **5** |
| 6 | ~8 | Test planning | 3 |
| 7 | ~15 | Test implementation | 15 |
| **Grand Total** | **~50** | **Full test suite** | **~25** |

---

## Expected Outcomes

### After Phase 5 (Discovery)
- ✅ 26 behavior inventory documents
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

## Questions?

- **Overall strategy:** `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md`
- **Phase 1 details:** `.plan/PHASE_1_START_HERE.md`
- **Your team guide:** `.plan/TEAM_XXX_GUIDE.md`
- **Engineering rules:** `bin/engineering-rules.md`

---

**Status:** Phase 1 ready to start  
**Next:** TEAM-216, 217, 218, 219 begin investigation
