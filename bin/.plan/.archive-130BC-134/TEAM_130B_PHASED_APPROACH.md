# TEAM-130B: PHASED SYNTHESIS APPROACH

**Duration:** 9-12 days (3 phases × 3-4 days each)  
**Team Size:** 2-3 people  
**Status:** 🎯 PHASED EXECUTION

---

## 🎯 CORE PRINCIPLE

**Team 130B MUST have ENTIRE CONTEXT of ALL 4 binaries before writing anything!**

**Why:** Cross-binary shared crate opportunities are the most valuable findings!

**Therefore:**
- Phase 1 = Read EVERYTHING, build mental model of entire system
- Phase 2 & 3 = Write final investigations with full context

---

## 📊 DOCUMENT STRUCTURE (Avoid Token Limits)

Instead of one 40-60 page file per binary, split into **3 parts:**

### Per Binary (3 files × 4 binaries = 12 files):

**Part 1: Metrics & Crate Decomposition** (15-20 pages)
- `TEAM_130B_FINAL_[binary]_PART1_METRICS.md`
- Ground truth metrics
- Proposed crate decomposition
- Each crate detailed

**Part 2: Shared Crates & External Libraries** (15-20 pages)
- `TEAM_130B_FINAL_[binary]_PART2_LIBRARIES.md`
- Shared crate usage analysis
- External Rust library recommendations
- Cross-binary recommendations

**Part 3: Migration & Approval** (10-15 pages)
- `TEAM_130B_FINAL_[binary]_PART3_MIGRATION.md`
- Architecture decisions
- Step-by-step migration strategy
- Risk assessment
- Test coverage
- Recommendations & approval

**Total per binary:** 40-55 pages (split into 3 manageable files)

---

## 🔄 THREE PHASES

### Phase 1: Foundation & Cross-Binary Analysis (Days 1-4)

**Goal:** Build complete mental model of entire system

**Duration:** 3-4 days  
**Output:** 1 comprehensive cross-binary analysis document

#### Day 1-2: Read EVERYTHING (20+ documents)
- [ ] Read all 4 team investigations (16+ docs)
- [ ] Read all 8 peer reviews
- [ ] Take notes on conflicts
- [ ] List all open questions
- [ ] Document all shared crate mentions

#### Day 3: Reconciliation & Verification
- [ ] Reconcile LOC conflicts (run cloc yourself)
- [ ] Reconcile architecture conflicts (analyze code)
- [ ] Reconcile dependency conflicts (cargo tree)
- [ ] Document ground truth for each binary

#### Day 4: Write Cross-Binary Analysis
- [ ] **File:** `TEAM_130B_CROSS_BINARY_ANALYSIS.md` (20-25 pages)

**Sections:**
1. Executive Summary (2 pages)
2. System-Wide Metrics (3 pages)
3. Shared Crate Audit Matrix (5-7 pages)
   - ALL 10+ shared crates analyzed across ALL 4 binaries
4. Cross-Binary Opportunities (5-7 pages)
   - auth-min standardization
   - SSH client sharing
   - Type sharing opportunities
5. External Library Recommendations (3-4 pages)
   - Workspace-level dependencies
   - Common testing libraries
6. Architecture-Level Decisions (2-3 pages)

**✅ Checkpoint:** You now have FULL CONTEXT of entire system!

---

### Phase 2: First Two Binaries (Days 5-8)

**Goal:** Write final investigations for rbee-hive and queen-rbee

**Duration:** 3-4 days  
**Output:** 6 files (3 parts × 2 binaries)

#### Why These First?
- rbee-hive = Largest, most complex (10 crates)
- queen-rbee = CRITICAL auth-min and russh fixes needed
- These two interact most closely

#### Day 5: rbee-hive Part 1 (Metrics & Crates)
**File:** `TEAM_130B_FINAL_rbee-hive_PART1_METRICS.md` (15-20 pages)

**Sections:**
1. Executive Summary (2 pages)
2. Ground Truth Metrics (3-4 pages)
3. Proposed Crate Decomposition (10-14 pages)
   - All 10 crates detailed with:
     - LOC, purpose, files, dependencies
     - Justification
     - External library suggestions for THIS crate
     - Approval status

**Estimated Time:** 6-8 hours

---

#### Day 6: rbee-hive Parts 2 & 3 (Libraries & Migration)

**File 1:** `TEAM_130B_FINAL_rbee-hive_PART2_LIBRARIES.md` (15-20 pages)

**Sections:**
1. Shared Crate Usage (8-10 pages)
   - All 10+ shared crates analyzed
2. Cross-Binary Shared Crate Recommendations (4-5 pages)
3. External Rust Library Recommendations (3-5 pages)

**File 2:** `TEAM_130B_FINAL_rbee-hive_PART3_MIGRATION.md` (10-15 pages)

**Sections:**
1. Architecture Decisions (2-3 pages)
2. Migration Strategy (6-8 pages) - Step-by-step
3. Risk Assessment (2-3 pages)
4. Test Coverage (1-2 pages)
5. Recommendations & Approval (1-2 pages)

**Estimated Time:** 8-10 hours total

---

#### Day 7: queen-rbee Part 1 (Metrics & Crates)
**File:** `TEAM_130B_FINAL_queen-rbee_PART1_METRICS.md` (15-20 pages)

Same structure as rbee-hive Part 1, but for 4 crates

**Estimated Time:** 5-7 hours

---

#### Day 8: queen-rbee Parts 2 & 3 (Libraries & Migration)

**File 1:** `TEAM_130B_FINAL_queen-rbee_PART2_LIBRARIES.md` (15-20 pages)

**CRITICAL Sections:**
- auth-min gap analysis (manual auth → auth-min migration)
- russh replacement (Command SSH → russh - security fix)
- Cross-binary impact (affects rbee-hive integration)

**File 2:** `TEAM_130B_FINAL_queen-rbee_PART3_MIGRATION.md` (10-15 pages)

**Estimated Time:** 8-10 hours total

**✅ Checkpoint:** 2 binaries complete (6 files, ~100 pages)

---

### Phase 3: Final Two Binaries (Days 9-12)

**Goal:** Write final investigations for llm-worker-rbee and rbee-keeper

**Duration:** 3-4 days  
**Output:** 6 files (3 parts × 2 binaries)

#### Why These Last?
- llm-worker-rbee = Worker pattern established, can reference queen-rbee
- rbee-keeper = CLI tool, more independent, benefits from seeing all others

#### Day 9: llm-worker-rbee Part 1 (Metrics & Crates)
**File:** `TEAM_130B_FINAL_llm-worker-rbee_PART1_METRICS.md` (15-20 pages)

**Focus:**
- 6 crates under `worker-rbee-crates/`
- Future-proofing for other worker types
- Reusability matrix

**Estimated Time:** 6-8 hours

---

#### Day 10: llm-worker-rbee Parts 2 & 3 (Libraries & Migration)

**File 1:** `TEAM_130B_FINAL_llm-worker-rbee_PART2_LIBRARIES.md` (15-20 pages)

**Focus:**
- Performance libraries (simd-json for inference)
- Prepare for future auth-min usage
- Model loading optimization

**File 2:** `TEAM_130B_FINAL_llm-worker-rbee_PART3_MIGRATION.md` (10-15 pages)

**Estimated Time:** 8-10 hours total

---

#### Day 11: rbee-keeper Part 1 (Metrics & Crates)
**File:** `TEAM_130B_FINAL_rbee-keeper_PART1_METRICS.md` (15-20 pages)

**Focus:**
- 5 crates
- CLI-focused design

**Estimated Time:** 5-7 hours

---

#### Day 12: rbee-keeper Parts 2 & 3 (Libraries & Migration)

**File 1:** `TEAM_130B_FINAL_rbee-keeper_PART2_LIBRARIES.md` (15-20 pages)

**Focus:**
- CLI libraries (clap, indicatif, console, dialoguer)
- SSH sharing with queen-rbee
- UX improvements

**File 2:** `TEAM_130B_FINAL_rbee-keeper_PART3_MIGRATION.md` (10-15 pages)

**Estimated Time:** 8-10 hours total

**✅ Final Checkpoint:** All 4 binaries complete!

---

## 📦 FINAL DELIVERABLES

### Total: 13 Files (~220 pages)

#### Cross-Binary (1 file):
1. `TEAM_130B_CROSS_BINARY_ANALYSIS.md` (20-25 pages)

#### Per Binary (3 files each × 4 binaries = 12 files):

**rbee-hive:**
2. `TEAM_130B_FINAL_rbee-hive_PART1_METRICS.md` (15-20 pages)
3. `TEAM_130B_FINAL_rbee-hive_PART2_LIBRARIES.md` (15-20 pages)
4. `TEAM_130B_FINAL_rbee-hive_PART3_MIGRATION.md` (10-15 pages)

**queen-rbee:**
5. `TEAM_130B_FINAL_queen-rbee_PART1_METRICS.md` (15-20 pages)
6. `TEAM_130B_FINAL_queen-rbee_PART2_LIBRARIES.md` (15-20 pages)
7. `TEAM_130B_FINAL_queen-rbee_PART3_MIGRATION.md` (10-15 pages)

**llm-worker-rbee:**
8. `TEAM_130B_FINAL_llm-worker-rbee_PART1_METRICS.md` (15-20 pages)
9. `TEAM_130B_FINAL_llm-worker-rbee_PART2_LIBRARIES.md` (15-20 pages)
10. `TEAM_130B_FINAL_llm-worker-rbee_PART3_MIGRATION.md` (10-15 pages)

**rbee-keeper:**
11. `TEAM_130B_FINAL_rbee-keeper_PART1_METRICS.md` (15-20 pages)
12. `TEAM_130B_FINAL_rbee-keeper_PART2_LIBRARIES.md` (15-20 pages)
13. `TEAM_130B_FINAL_rbee-keeper_PART3_MIGRATION.md` (10-15 pages)

---

## ⏱️ TIME BREAKDOWN

| Phase | Duration | Deliverables | Effort |
|-------|----------|--------------|--------|
| Phase 1 | 3-4 days | 1 cross-binary doc | 24-32 hours |
| Phase 2 | 3-4 days | 6 files (2 binaries) | 24-32 hours |
| Phase 3 | 3-4 days | 6 files (2 binaries) | 24-32 hours |
| **TOTAL** | **9-12 days** | **13 files** | **72-96 hours** |

**Per person (2-3 people):** 24-48 hours each over ~2 weeks

---

## ✅ BENEFITS OF PHASED APPROACH

### Manageable Chunks
- ✅ No 60-page documents in one sitting
- ✅ 15-20 page chunks are token-friendly
- ✅ Clear stopping points

### Full Context Maintained
- ✅ Phase 1 builds complete mental model
- ✅ Cross-binary analysis done FIRST
- ✅ Can reference earlier phases when writing later binaries

### Parallel Work Possible
- ✅ Phase 2: Person A writes rbee-hive, Person B writes queen-rbee
- ✅ Phase 3: Person A writes llm-worker, Person B writes rbee-keeper

### Progressive Delivery
- ✅ After Phase 1: Teams have cross-binary guidance
- ✅ After Phase 2: 2 binaries ready for Phase 2 (Preparation)
- ✅ After Phase 3: All 4 binaries ready

---

## 🎯 PHASE FOCUS

### Phase 1: CONTEXT IS KING
- Read everything
- Understand the system as a whole
- Identify cross-binary patterns
- No rushing to write

### Phase 2: CRITICAL PATH
- rbee-hive (largest complexity)
- queen-rbee (critical security fixes)
- Set the quality bar

### Phase 3: COMPLETE THE PICTURE
- llm-worker-rbee (worker pattern)
- rbee-keeper (CLI tool)
- Apply learnings from Phase 2

---

## 🚫 ANTI-PATTERNS TO AVOID

**DON'T:**
- ❌ Start writing before reading everything (Phase 1 is mandatory!)
- ❌ Write one binary in isolation (lose cross-binary opportunities)
- ❌ Skip cross-binary analysis (most valuable output!)
- ❌ Try to write all 3 parts of one binary in one day (too much!)

**DO:**
- ✅ Complete Phase 1 fully before starting Phase 2
- ✅ Write with full system context
- ✅ Reference cross-binary analysis in every Part 2
- ✅ Take breaks between parts

---

## 📞 COORDINATION

### Phase Gates:
1. **End of Phase 1:** Review cross-binary analysis as team
2. **End of Phase 2:** Review first 2 binaries, adjust approach if needed
3. **End of Phase 3:** Final review all 13 documents

### Daily Standups:
- What did you read/write yesterday?
- What conflicts did you find?
- What cross-binary opportunities emerged?
- Any blockers?

### Slack Channels:
- `#team-130b-phase1` - Reading & analysis
- `#team-130b-phase2` - Writing rbee-hive & queen-rbee
- `#team-130b-phase3` - Writing llm-worker & rbee-keeper

---

## 🚀 GETTING STARTED

### Week 1 (Phase 1):
1. Read `TEAM_130B_FINAL_SYNTHESIS.md`
2. Gather all 20+ input documents
3. Read everything (Days 1-2)
4. Reconcile conflicts (Day 3)
5. Write cross-binary analysis (Day 4)

### Week 2 (Phase 2):
1. Write rbee-hive 3 parts (Days 5-6)
2. Write queen-rbee 3 parts (Days 7-8)

### Week 3 (Phase 3):
1. Write llm-worker-rbee 3 parts (Days 9-10)
2. Write rbee-keeper 3 parts (Days 11-12)

---

**TEAM-130B: Build the complete mental model first, then write with full context! 🧠**
