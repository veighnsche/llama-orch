# CRATE DECOMPOSITION PROJECT - START HERE

**Date:** 2025-10-19  
**Project:** Binary Decomposition into Focused Crates  
**Status:** 🚀 READY TO START

---

## 🎯 PROJECT OVERVIEW

**Goal:** Decompose 4 monolithic binaries into 25 focused library crates

**Why:**
- 93% faster compilation (1m 42s → 8s per crate)
- Perfect test isolation
- Future-proof architecture
- Clear ownership boundaries

**Binaries:**
1. `rbee-hive` (4,184 LOC) → 10 crates
2. `queen-rbee` (~3,100 LOC) → 4 crates
3. `llm-worker-rbee` (~2,550 LOC) → 6 crates
4. `rbee-keeper` (1,252 LOC) → 5 crates

---

## 📋 3-PHASE APPROACH

### Phase 1: INVESTIGATION (Week 1)
**Teams:** 131, 132, 133, 134  
**Goal:** Deep analysis, no code changes  
**Output:** Investigation reports

### Phase 2: PREPARATION (Week 2)
**Teams:** 135, 136, 137, 138  
**Goal:** Create structure, plan migration  
**Output:** Crate skeletons, migration scripts

### Phase 3: IMPLEMENTATION (Week 3)
**Teams:** 139, 140, 141, 142  
**Goal:** Execute migration, verify tests  
**Output:** Working decomposed binaries

---

## 👥 TEAM ASSIGNMENTS

### Phase 1: INVESTIGATION (NO CODE CHANGES!)

| Team | Binary | LOC | Crates | Duration |
|------|--------|-----|--------|----------|
| **TEAM-131** | rbee-hive | 4,184 | 10 | 5 days |
| **TEAM-132** | queen-rbee | ~3,100 | 4 | 5 days |
| **TEAM-133** | llm-worker-rbee | ~2,550 | 6 | 5 days |
| **TEAM-134** | rbee-keeper | 1,252 | 5 | 5 days |

**Deliverables:**
- `TEAM_13x_[binary]_INVESTIGATION.md`
- Crate boundary analysis
- Dependency mapping
- Shared crate usage analysis
- Risk assessment

### Phase 2: PREPARATION (CREATE STRUCTURE)

| Team | Binary | Tasks | Duration |
|------|--------|-------|----------|
| **TEAM-135** | rbee-hive | Create 10 crate dirs, Cargo.toml files | 5 days |
| **TEAM-136** | queen-rbee | Create 4 crate dirs, Cargo.toml files | 5 days |
| **TEAM-137** | llm-worker-rbee | Create 6 crate dirs, Cargo.toml files | 5 days |
| **TEAM-138** | rbee-keeper | Create 5 crate dirs, Cargo.toml files | 5 days |

**Deliverables:**
- `TEAM_13x_[binary]_PREPARATION.md`
- Empty crate directories
- Cargo.toml files
- Migration scripts
- Test plan

### Phase 3: IMPLEMENTATION (EXECUTE MIGRATION)

| Team | Binary | Tasks | Duration |
|------|--------|-------|----------|
| **TEAM-139** | rbee-hive | Move code, update imports, verify tests | 5 days |
| **TEAM-140** | queen-rbee | Move code, update imports, verify tests | 5 days |
| **TEAM-141** | llm-worker-rbee | Move code, update imports, verify tests | 5 days |
| **TEAM-142** | rbee-keeper | Move code, update imports, verify tests | 5 days |

**Deliverables:**
- `TEAM_13x_[binary]_IMPLEMENTATION.md`
- Working decomposed binary
- All tests passing
- BDD suite per crate
- Documentation

---

## 📊 PROJECT TIMELINE

```
Week 1: INVESTIGATION (Teams 131-134)
├─ Day 1-2: Deep code analysis
├─ Day 3-4: Dependency mapping
└─ Day 5: Risk assessment & report

Week 2: PREPARATION (Teams 135-138)
├─ Day 1-2: Create crate structure
├─ Day 3-4: Write Cargo.toml files
└─ Day 5: Migration scripts & test plan

Week 3: IMPLEMENTATION (Teams 139-142)
├─ Day 1-2: Move code to crates
├─ Day 3-4: Update imports & tests
└─ Day 5: Verification & documentation

Week 4: INTEGRATION & CLEANUP
├─ Day 1-2: CI/CD updates
├─ Day 3-4: Integration testing
└─ Day 5: Final verification
```

---

## 🎯 PHASE 1 FOCUS: INVESTIGATION

### What Teams 131-134 Must Do:

#### 1. **Code Analysis** (Day 1-2)
- [ ] Read every file in the binary
- [ ] Identify logical modules
- [ ] Map dependencies between modules
- [ ] Identify circular dependencies
- [ ] Document current architecture

#### 2. **Crate Boundary Analysis** (Day 2-3)
- [ ] Propose crate boundaries
- [ ] Justify each crate split
- [ ] Identify shared code
- [ ] Map public APIs
- [ ] Document data flow

#### 3. **Shared Crate Analysis** (Day 3-4)
- [ ] Audit ALL shared crates usage
- [ ] Identify missing shared crate opportunities
- [ ] Check for duplicate code
- [ ] Verify shared crate versions
- [ ] Document shared crate strategy

#### 4. **Risk Assessment** (Day 4-5)
- [ ] Identify breaking changes
- [ ] Assess migration complexity
- [ ] Identify test gaps
- [ ] Document rollback plan
- [ ] Estimate effort

#### 5. **Report Writing** (Day 5)
- [ ] Complete investigation report
- [ ] Include code examples
- [ ] Add dependency diagrams
- [ ] Document recommendations
- [ ] Get peer review

### What Teams 131-134 Must NOT Do:

- ❌ **NO CODE CHANGES** - Investigation only!
- ❌ **NO REFACTORING** - Document as-is!
- ❌ **NO ASSUMPTIONS** - Verify everything!
- ❌ **NO SHORTCUTS** - Deep analysis required!

---

## 📚 INVESTIGATION CHECKLIST

### For Each Binary:

- [ ] **Current State Analysis**
  - [ ] Total LOC count
  - [ ] File structure documented
  - [ ] Module dependencies mapped
  - [ ] External dependencies listed
  - [ ] Test coverage measured

- [ ] **Proposed Crate Structure**
  - [ ] Crate names defined
  - [ ] Crate boundaries justified
  - [ ] Public APIs designed
  - [ ] Dependencies mapped
  - [ ] LOC per crate estimated

- [ ] **Shared Crate Audit**
  - [ ] Current usage documented
  - [ ] Missing opportunities identified
  - [ ] Duplicate code found
  - [ ] Version conflicts checked
  - [ ] Integration points mapped

- [ ] **Migration Strategy**
  - [ ] Step-by-step plan
  - [ ] Breaking changes identified
  - [ ] Test strategy defined
  - [ ] Rollback plan documented
  - [ ] Timeline estimated

- [ ] **Risk Assessment**
  - [ ] Technical risks listed
  - [ ] Mitigation strategies defined
  - [ ] Dependencies on other teams
  - [ ] Blocking issues identified
  - [ ] Contingency plans

---

## 🔍 SHARED CRATES TO AUDIT

### Existing Shared Crates (Must Use!)

| Crate | Purpose | Current Users |
|-------|---------|---------------|
| `hive-core` | Core hive types | rbee-hive |
| `model-catalog` | Model management | rbee-hive, queen-rbee |
| `gpu-info` | GPU detection | rbee-hive, llm-worker-rbee |
| `auth-min` | Authentication | rbee-hive, queen-rbee |
| `secrets-management` | Secrets | rbee-hive, queen-rbee |
| `input-validation` | Input validation | All binaries |
| `audit-logging` | Audit logs | rbee-hive, queen-rbee |
| `deadline-propagation` | Deadlines | All binaries |
| `narration-core` | Observability | All binaries |
| `jwt-guardian` | JWT handling | rbee-hive, queen-rbee |

### Questions to Answer:

1. **Are we using ALL shared crates where appropriate?**
2. **Is there duplicate code that should be in shared crates?**
3. **Are there new shared crates we should create?**
4. **Are version conflicts preventing shared crate usage?**
5. **Can we consolidate similar functionality?**

---

## 📖 INVESTIGATION REPORT TEMPLATE

Each team must produce:

```markdown
# TEAM-13x [BINARY] INVESTIGATION REPORT

## Executive Summary
- Current state
- Proposed crates
- Key findings
- Recommendations

## Current Architecture
- File structure
- Module dependencies
- External dependencies
- Test coverage

## Proposed Crate Structure
- Crate 1: [name]
  - Purpose
  - LOC
  - Public API
  - Dependencies
  - Justification
- Crate 2: [name]
  ...

## Shared Crate Analysis
- Current usage
- Missing opportunities
- Duplicate code
- Recommendations

## Migration Strategy
- Step-by-step plan
- Breaking changes
- Test strategy
- Timeline

## Risk Assessment
- Technical risks
- Mitigation strategies
- Dependencies
- Contingency plans

## Recommendations
- Go/No-Go decision
- Alternative approaches
- Next steps
```

---

## 🚀 SUCCESS CRITERIA

### Phase 1 Complete When:

- [ ] All 4 investigation reports complete
- [ ] Peer reviews done
- [ ] Shared crate audit complete
- [ ] Migration strategies defined
- [ ] Risks documented
- [ ] Go/No-Go decision made

### Quality Gates:

- ✅ **Completeness:** Every file analyzed
- ✅ **Accuracy:** No assumptions, all verified
- ✅ **Clarity:** Clear recommendations
- ✅ **Actionable:** Ready for Phase 2
- ✅ **Reviewed:** Peer-reviewed by another team

---

## 📞 COMMUNICATION

### Daily Standups:
- What did you analyze yesterday?
- What will you analyze today?
- Any blockers or questions?

### Weekly Sync:
- Progress review
- Cross-team dependencies
- Shared findings
- Risk updates

### Slack Channels:
- `#team-131-rbee-hive`
- `#team-132-queen-rbee`
- `#team-133-llm-worker-rbee`
- `#team-134-rbee-keeper`
- `#crate-decomposition-all` (cross-team)

---

## 📂 FOLDER STRUCTURE

```
bin/.plan/
├── START_HERE.md                           # This file
├── TEAM_131_rbee-hive_INVESTIGATION.md     # Team 131 report
├── TEAM_132_queen-rbee_INVESTIGATION.md    # Team 132 report
├── TEAM_133_llm-worker-rbee_INVESTIGATION.md # Team 133 report
└── TEAM_134_rbee-keeper_INVESTIGATION.md   # Team 134 report

# After Phase 1, add:
├── TEAM_135_rbee-hive_PREPARATION.md
├── TEAM_136_queen-rbee_PREPARATION.md
├── TEAM_137_llm-worker-rbee_PREPARATION.md
├── TEAM_138_rbee-keeper_PREPARATION.md

# After Phase 2, add:
├── TEAM_139_rbee-hive_IMPLEMENTATION.md
├── TEAM_140_queen-rbee_IMPLEMENTATION.md
├── TEAM_141_llm-worker-rbee_IMPLEMENTATION.md
└── TEAM_142_rbee-keeper_IMPLEMENTATION.md
```

---

## 🎯 NEXT STEPS

### For Team Leads:

1. **Read this document completely**
2. **Review your binary's current code**
3. **Read the team-specific investigation guide**
4. **Set up daily standups**
5. **Start Day 1 analysis**

### For Project Manager:

1. **Assign teams to binaries**
2. **Set up Slack channels**
3. **Schedule daily standups**
4. **Schedule weekly syncs**
5. **Monitor progress**

---

## ✅ READY TO START!

**Phase 1 starts NOW!**

**Teams 131-134: Read your investigation guides and begin!**

**Remember:**
- 🔍 **INVESTIGATION ONLY** - No code changes!
- 📊 **DEEP ANALYSIS** - Verify everything!
- 🤝 **COLLABORATE** - Share findings!
- 📝 **DOCUMENT** - Write everything down!

---

**Let's decompose these binaries and make them FAST! 🚀**
