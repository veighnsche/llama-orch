# PLAN FOLDER STRUCTURE

**Location:** `bin/.plan/`  
**Created:** 2025-10-19  
**Status:** ✅ COMPLETE

---

## 📂 COMPLETE FOLDER STRUCTURE

```
bin/.plan/
│
├── README.md                                    # Folder overview
├── PROJECT_SUMMARY.md                           # Complete project summary
├── FOLDER_STRUCTURE.md                          # This file
├── START_HERE.md                                # 🎯 START HERE!
│
├── TEAM_130_BDD_SCALABILITY_INVESTIGATION.md    # Background: Original problem
├── TEAM_130_CRATE_DECOMPOSITION_ANALYSIS.md     # Background: Detailed solution
├── TEAM_130_FINAL_RECOMMENDATION.md             # Background: Architecture decision
├── TEAM_130_INVESTIGATION_COMPLETE.md           # Background: Summary
│
├── TEAM_131_rbee-hive_INVESTIGATION.md          # Phase 1: rbee-hive (10 crates)
├── TEAM_132_queen-rbee_INVESTIGATION.md         # Phase 1: queen-rbee (4 crates)
├── TEAM_133_llm-worker-rbee_INVESTIGATION.md    # Phase 1: llm-worker-rbee (6 crates)
└── TEAM_134_rbee-keeper_INVESTIGATION.md        # Phase 1: rbee-keeper (5 crates)

# After Phase 1, add:
├── TEAM_135_rbee-hive_PREPARATION.md            # Phase 2: rbee-hive
├── TEAM_136_queen-rbee_PREPARATION.md           # Phase 2: queen-rbee
├── TEAM_137_llm-worker-rbee_PREPARATION.md      # Phase 2: llm-worker-rbee
└── TEAM_138_rbee-keeper_PREPARATION.md          # Phase 2: rbee-keeper

# After Phase 2, add:
├── TEAM_139_rbee-hive_IMPLEMENTATION.md         # Phase 3: rbee-hive
├── TEAM_140_queen-rbee_IMPLEMENTATION.md        # Phase 3: queen-rbee
├── TEAM_141_llm-worker-rbee_IMPLEMENTATION.md   # Phase 3: llm-worker-rbee
└── TEAM_142_rbee-keeper_IMPLEMENTATION.md       # Phase 3: rbee-keeper
```

---

## 📖 DOCUMENT GUIDE

### 🎯 For New Team Members

**Read in this order:**
1. **START_HERE.md** - Project overview and team assignments
2. **Your team's investigation guide** (TEAM_13x_[binary]_INVESTIGATION.md)
3. **PROJECT_SUMMARY.md** - Complete project details

### 📊 For Project Managers

**Read in this order:**
1. **PROJECT_SUMMARY.md** - Executive overview
2. **START_HERE.md** - Team structure and timeline
3. **TEAM_130_FINAL_RECOMMENDATION.md** - Architecture decision

### 🔍 For Technical Leads

**Read in this order:**
1. **TEAM_130_CRATE_DECOMPOSITION_ANALYSIS.md** - Detailed technical analysis
2. **Your team's investigation guide** - Specific tasks
3. **TEAM_130_BDD_SCALABILITY_INVESTIGATION.md** - Original problem context

---

## 📋 DOCUMENT DESCRIPTIONS

### Getting Started
- **README.md** (5 KB) - Quick overview of folder contents
- **START_HERE.md** (10 KB) - Complete project overview, team assignments, timeline
- **PROJECT_SUMMARY.md** (11 KB) - Executive summary and complete plan
- **FOLDER_STRUCTURE.md** (This file) - Navigation guide

### Background (TEAM-130)
- **TEAM_130_BDD_SCALABILITY_INVESTIGATION.md** (13 KB)
  - Original problem analysis
  - Current state metrics
  - Why we need decomposition

- **TEAM_130_CRATE_DECOMPOSITION_ANALYSIS.md** (18 KB)
  - Detailed decomposition strategy
  - All 4 binaries analyzed
  - Crate proposals with justification
  - Migration strategy

- **TEAM_130_FINAL_RECOMMENDATION.md** (8 KB)
  - Final architecture recommendation
  - Migration timeline
  - Success criteria

- **TEAM_130_INVESTIGATION_COMPLETE.md** (5 KB)
  - Summary of all findings
  - Ready for approval

### Phase 1: Investigation (Week 1)

- **TEAM_131_rbee-hive_INVESTIGATION.md** (13 KB)
  - rbee-hive analysis guide
  - 10 proposed crates
  - 4,184 LOC to decompose
  - Shared crate audit checklist

- **TEAM_132_queen-rbee_INVESTIGATION.md** (11 KB)
  - queen-rbee analysis guide
  - 4 proposed crates
  - ~3,100 LOC to decompose
  - Integration points with rbee-hive

- **TEAM_133_llm-worker-rbee_INVESTIGATION.md** (15 KB)
  - llm-worker-rbee analysis guide
  - 6 proposed crates under `worker-rbee-crates/`
  - ~2,550 LOC to decompose
  - **CRITICAL:** Future-proof for all worker types!

- **TEAM_134_rbee-keeper_INVESTIGATION.md** (14 KB)
  - rbee-keeper analysis guide
  - 5 proposed crates
  - 1,252 LOC to decompose
  - CLI user experience focus

---

## 🎯 READING PATHS

### Path 1: Quick Start (15 minutes)
1. START_HERE.md (5 min)
2. Your team's investigation guide (10 min)
3. Begin work!

### Path 2: Complete Understanding (45 minutes)
1. PROJECT_SUMMARY.md (10 min)
2. START_HERE.md (10 min)
3. TEAM_130_FINAL_RECOMMENDATION.md (10 min)
4. Your team's investigation guide (15 min)

### Path 3: Deep Dive (2 hours)
1. TEAM_130_BDD_SCALABILITY_INVESTIGATION.md (20 min)
2. TEAM_130_CRATE_DECOMPOSITION_ANALYSIS.md (30 min)
3. TEAM_130_FINAL_RECOMMENDATION.md (15 min)
4. PROJECT_SUMMARY.md (15 min)
5. START_HERE.md (15 min)
6. Your team's investigation guide (25 min)

---

## 📊 DOCUMENT STATS

### Total Documents: 12 (Phase 1)
- Getting Started: 4 docs
- Background: 4 docs
- Investigation Guides: 4 docs

### Total Size: ~160 KB
- Background investigation: ~54 KB
- Getting started: ~26 KB
- Investigation guides: ~52 KB

### Total Pages: ~80 pages
- Comprehensive coverage
- No assumptions
- Ready for execution

---

## ✅ DOCUMENT CHECKLIST

### Phase 1 Complete:
- [x] START_HERE.md
- [x] PROJECT_SUMMARY.md
- [x] README.md
- [x] FOLDER_STRUCTURE.md
- [x] TEAM_130 background (4 docs)
- [x] TEAM_131 investigation guide
- [x] TEAM_132 investigation guide
- [x] TEAM_133 investigation guide
- [x] TEAM_134 investigation guide

### Phase 2 Needed:
- [ ] TEAM_135 preparation guide
- [ ] TEAM_136 preparation guide
- [ ] TEAM_137 preparation guide
- [ ] TEAM_138 preparation guide

### Phase 3 Needed:
- [ ] TEAM_139 implementation guide
- [ ] TEAM_140 implementation guide
- [ ] TEAM_141 implementation guide
- [ ] TEAM_142 implementation guide

---

## 🚀 NEXT STEPS

### For Teams Starting Phase 1:
1. Read START_HERE.md
2. Read your investigation guide
3. Set up daily standups
4. Begin Day 1 analysis

### For Project Manager:
1. Review PROJECT_SUMMARY.md
2. Assign teams to binaries
3. Set up Slack channels
4. Monitor progress

---

## 📞 QUESTIONS?

- **Slack:** `#crate-decomposition-all`
- **Project Lead:** [Name]
- **Documentation:** This folder!

---

**TEAM-130: Complete documentation delivered. Ready to execute! 🚀**
