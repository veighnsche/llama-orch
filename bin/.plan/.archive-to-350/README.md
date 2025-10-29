# Test Freeze Planning System

**Status:** ✅ COMPLETE - Ready for TEAM-216 to start  
**Created:** Oct 22, 2025  
**Purpose:** Freeze ALL current behaviors with comprehensive test coverage

---

## What This Is

A complete multi-phase plan to:

1. **Discover** ALL behaviors in `/bin/` (26 components)
2. **Plan** comprehensive test coverage
3. **Implement** tests to freeze all behaviors

**Goal:** Make regressions impossible without test failures.

---

## Quick Start

### For Phase 1 Teams (START HERE)

**You are:** TEAM-216, TEAM-217, TEAM-218, or TEAM-219

**Read this:** [PHASE_1_START_HERE.md](PHASE_1_START_HERE.md)

**Then read your guide:**
- TEAM-216: [TEAM_216_GUIDE.md](TEAM_216_GUIDE.md) - rbee-keeper
- TEAM-217: [TEAM_217_GUIDE.md](TEAM_217_GUIDE.md) - queen-rbee
- TEAM-218: [TEAM_218_GUIDE.md](TEAM_218_GUIDE.md) - rbee-hive
- TEAM-219: [TEAM_219_GUIDE.md](TEAM_219_GUIDE.md) - llm-worker-rbee

### For Everyone Else

**Read this:** [00_INDEX.md](00_INDEX.md) - Complete navigation

---

## Document Structure

```
.plan/
├── README.md (you are here)
├── 00_INDEX.md (navigation hub)
│
├── BEHAVIOR_DISCOVERY_MASTER_PLAN.md (complete strategy)
├── QUICK_REFERENCE_TESTING_PLAN.md (summary)
│
├── PHASE_1_START_HERE.md (Phase 1 overview)
├── PHASE_2_GUIDES.md (Phase 2 guide)
├── PHASE_3_GUIDES.md (Phase 3 guide)
├── PHASE_4_GUIDES.md (Phase 4 guide)
├── PHASE_5_GUIDES.md (Phase 5 guide)
│
├── TEAM_216_GUIDE.md (rbee-keeper)
├── TEAM_217_GUIDE.md (queen-rbee)
├── TEAM_218_GUIDE.md (rbee-hive)
├── TEAM_219_GUIDE.md (llm-worker-rbee)
│
└── [Future: TEAM_220+ guides and deliverables]
```

---

## The Plan in 30 Seconds

### Week 1: Discovery (5 days, 26 teams)
- **Day 1:** 4 teams document main binaries
- **Day 2:** 3 teams document queen crates
- **Day 3:** 7 teams document hive crates
- **Day 4:** 8 teams document shared crates
- **Day 5:** 4 teams document integration flows

**Output:** 26 behavior inventory documents

### Week 2: Test Planning (3 days, ~8 teams)
- Create comprehensive test plans
- Cover all discovered behaviors
- Prioritize by criticality

**Output:** Complete test plans

### Weeks 3-5: Test Implementation (15 days, ~15 teams)
- Implement unit tests
- Implement BDD tests
- Implement integration tests
- Implement E2E tests

**Output:** Full test suite, behaviors frozen

---

## Key Documents

### Start Here
- [00_INDEX.md](00_INDEX.md) - Navigation hub
- [PHASE_1_START_HERE.md](PHASE_1_START_HERE.md) - Phase 1 teams start here

### Strategy
- [BEHAVIOR_DISCOVERY_MASTER_PLAN.md](BEHAVIOR_DISCOVERY_MASTER_PLAN.md) - Complete plan
- [QUICK_REFERENCE_TESTING_PLAN.md](QUICK_REFERENCE_TESTING_PLAN.md) - Quick summary

### Phase Guides
- [PHASE_1_START_HERE.md](PHASE_1_START_HERE.md) - Main binaries (READY NOW)
- [PHASE_2_GUIDES.md](PHASE_2_GUIDES.md) - Queen crates
- [PHASE_3_GUIDES.md](PHASE_3_GUIDES.md) - Hive crates
- [PHASE_4_GUIDES.md](PHASE_4_GUIDES.md) - Shared crates
- [PHASE_5_GUIDES.md](PHASE_5_GUIDES.md) - Integration flows

### Team Guides (Phase 1)
- [TEAM_216_GUIDE.md](TEAM_216_GUIDE.md) - rbee-keeper investigation
- [TEAM_217_GUIDE.md](TEAM_217_GUIDE.md) - queen-rbee investigation
- [TEAM_218_GUIDE.md](TEAM_218_GUIDE.md) - rbee-hive investigation
- [TEAM_219_GUIDE.md](TEAM_219_GUIDE.md) - llm-worker-rbee investigation

---

## Team Assignments

### Phase 1 (READY NOW)
- **TEAM-216:** rbee-keeper (CLI client)
- **TEAM-217:** queen-rbee (Queen daemon)
- **TEAM-218:** rbee-hive (Hive daemon)
- **TEAM-219:** llm-worker-rbee (Worker daemon)

### Phase 2
- **TEAM-220:** hive-lifecycle crate
- **TEAM-221:** hive-registry crate
- **TEAM-222:** ssh-client crate

### Phase 3
- **TEAM-223:** device-detection crate
- **TEAM-224:** download-tracker crate
- **TEAM-225:** model-catalog crate
- **TEAM-226:** model-provisioner crate
- **TEAM-227:** monitor crate
- **TEAM-228:** vram-checker crate
- **TEAM-229:** worker-management crates (3 crates)

### Phase 4
- **TEAM-230:** narration crates
- **TEAM-231:** daemon-lifecycle crate
- **TEAM-232:** rbee-http-client crate
- **TEAM-233:** config/operations crates
- **TEAM-234:** job/deadline crates
- **TEAM-235:** auth/jwt crates
- **TEAM-236:** audit/validation crates
- **TEAM-237:** heartbeat/update/core crates

### Phase 5
- **TEAM-238:** keeper ↔ queen integration
- **TEAM-239:** queen ↔ hive integration
- **TEAM-240:** hive ↔ worker integration
- **TEAM-241:** End-to-end inference flows

### Phase 6-7 (TBD)
- **TEAM-242+:** Test planning
- **TEAM-250+:** Test implementation

---

## Success Metrics

### After Discovery (Phase 5)
- ✅ 26 behavior inventories complete
- ✅ All behaviors documented
- ✅ All test gaps identified

### After Planning (Phase 6)
- ✅ Comprehensive test plans created
- ✅ All behaviors have test coverage plan

### After Implementation (Phase 7)
- ✅ Full test suite implemented
- ✅ 80%+ code coverage
- ✅ All behaviors frozen
- ✅ No regressions possible

---

## Rules

### Discovery (Phases 1-5)
- ❌ NO code changes
- ✅ Document existing behaviors
- ✅ Identify test gaps
- ✅ Max 3 pages (4 for Phase 5)
- ✅ Add code signatures

### Test Planning (Phase 6)
- ✅ Cover all discovered behaviors
- ✅ Include edge cases
- ✅ Include error paths

### Test Implementation (Phase 7)
- ✅ Implement all planned tests
- ✅ Freeze all behaviors
- ✅ No regressions possible

---

## Getting Help

- **Navigation:** [00_INDEX.md](00_INDEX.md)
- **Strategy:** [BEHAVIOR_DISCOVERY_MASTER_PLAN.md](BEHAVIOR_DISCOVERY_MASTER_PLAN.md)
- **Quick Reference:** [QUICK_REFERENCE_TESTING_PLAN.md](QUICK_REFERENCE_TESTING_PLAN.md)
- **Your phase:** See phase guide (PHASE_X_GUIDES.md)
- **Your team:** See team guide (TEAM_XXX_GUIDE.md)

---

## Status

**Current Phase:** Phase 1  
**Status:** ✅ READY TO START  
**Next Action:** TEAM-216, 217, 218, 219 begin investigation

**Documents Created:** 11  
**Teams Defined:** 26 (discovery) + ~23 (planning/implementation)  
**Timeline:** ~25 days total

---

**This is a complete, ready-to-execute plan.**

**Next step:** TEAM-216 starts investigating rbee-keeper.
