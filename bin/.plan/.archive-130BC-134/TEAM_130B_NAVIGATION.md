# TEAM-130B: DOCUMENT NAVIGATION GUIDE

**Too many documents? Not sure where to start? Use this guide! 📖**

---

## 🎯 START HERE (In Order)

### 1. **TEAM_130B_README.md** (5 min read)
**Purpose:** Quick overview of the team's mission  
**Read when:** First time looking at Team 130B work  
**Contains:**
- Mission statement
- Timeline (9-12 days, 3 phases)
- Deliverables overview (13 files)
- Quick start

---

### 2. **TEAM_130B_SUMMARY.md** (10 min read)
**Purpose:** Executive summary of the 3-phase approach  
**Read when:** You want a high-level plan overview  
**Contains:**
- 3 phases explained
- Timeline breakdown
- Deliverables summary
- Quality gates

---

### 3. **TEAM_130B_PHASED_APPROACH.md** ⭐ PRIMARY (30 min read)
**Purpose:** THE MAIN EXECUTION GUIDE  
**Read when:** You're ready to actually do the work  
**Contains:**
- Complete 3-phase breakdown
- Day-by-day activities
- File-by-file deliverables
- Time estimates per task
- Validation criteria

**🚨 THIS IS YOUR PRIMARY GUIDE FOR EXECUTION!**

---

## 📚 REFERENCE DOCUMENTS (Read as Needed)

### 4. **TEAM_130B_EXAMPLE_STRUCTURE.md** (10 min)
**Purpose:** Template for file structure  
**Read when:** Writing a final investigation file  
**Contains:**
- Part 1 structure (Metrics & Crates)
- Part 2 structure (Libraries)
- Part 3 structure (Migration)
- What goes in each section

---

### 5. **TEAM_130B_RUST_LIBRARY_SUGGESTIONS.md** (20 min)
**Purpose:** Library recommendations reference  
**Read when:** Writing Part 2 (Libraries) sections  
**Contains:**
- 20+ Rust crate suggestions
- Testing libraries (rstest, proptest, etc.)
- CLI libraries (clap, indicatif, etc.)
- SSH libraries (russh - CRITICAL)
- Code examples and versions

---

### 6. **TEAM_130B_FINAL_SYNTHESIS.md** (Reference Only)
**Purpose:** Detailed methodology reference  
**Read when:** You need deep-dive on synthesis process  
**Contains:**
- Detailed reconciliation methodology
- Cross-binary analysis techniques
- Template structures (now outdated)

**⚠️ WARNING:** This document describes methodology in detail but is NOT the execution guide. Use TEAM_130B_PHASED_APPROACH.md for actual execution!

---

## 🗺️ DOCUMENT MAP BY TASK

### Task: "I just joined Team 130B, where do I start?"
**Read:**
1. TEAM_130B_README.md
2. TEAM_130B_SUMMARY.md
3. TEAM_130B_PHASED_APPROACH.md (Day 1 section)

---

### Task: "I'm starting Phase 1 (Days 1-4)"
**Read:**
1. TEAM_130B_PHASED_APPROACH.md → Phase 1 section
2. TEAM_130B_FINAL_SYNTHESIS.md → Data Collection section (reference)

**Focus:** Read all 20+ input documents, build mental model

---

### Task: "I'm writing Part 1 (Metrics & Crates)"
**Read:**
1. TEAM_130B_PHASED_APPROACH.md → Your specific day
2. TEAM_130B_EXAMPLE_STRUCTURE.md → Part 1 section
3. Cross-binary analysis (from Phase 1)

**Write:** `TEAM_130B_FINAL_[binary]_PART1_METRICS.md`

---

### Task: "I'm writing Part 2 (Libraries)"
**Read:**
1. TEAM_130B_PHASED_APPROACH.md → Your specific day
2. TEAM_130B_EXAMPLE_STRUCTURE.md → Part 2 section
3. TEAM_130B_RUST_LIBRARY_SUGGESTIONS.md
4. Cross-binary analysis (from Phase 1)

**Write:** `TEAM_130B_FINAL_[binary]_PART2_LIBRARIES.md`

---

### Task: "I'm writing Part 3 (Migration)"
**Read:**
1. TEAM_130B_PHASED_APPROACH.md → Your specific day
2. TEAM_130B_EXAMPLE_STRUCTURE.md → Part 3 section

**Write:** `TEAM_130B_FINAL_[binary]_PART3_MIGRATION.md`

---

### Task: "I need library suggestions"
**Read:**
1. TEAM_130B_RUST_LIBRARY_SUGGESTIONS.md
2. Cross-binary analysis → External library recommendations section

---

### Task: "I need to understand the synthesis process"
**Read:**
1. TEAM_130B_FINAL_SYNTHESIS.md → Reference sections

---

### Task: "What are the deliverables again?"
**Read:**
1. TEAM_130B_PHASED_APPROACH.md → "Final Deliverables" section
2. TEAM_130B_SUMMARY.md → Deliverables section

---

## 📅 DOCUMENT USAGE BY PHASE

### Phase 1 (Days 1-4) - Foundation
**Primary:**
- TEAM_130B_PHASED_APPROACH.md (Phase 1 section)

**Reference:**
- TEAM_130B_FINAL_SYNTHESIS.md (methodology)

**Output:**
- TEAM_130B_CROSS_BINARY_ANALYSIS.md

---

### Phase 2 (Days 5-8) - rbee-hive + queen-rbee
**Primary:**
- TEAM_130B_PHASED_APPROACH.md (Phase 2 section)
- TEAM_130B_EXAMPLE_STRUCTURE.md (template)
- Cross-binary analysis (your Phase 1 output)

**Reference:**
- TEAM_130B_RUST_LIBRARY_SUGGESTIONS.md

**Output:**
- 6 files (3 per binary)

---

### Phase 3 (Days 9-12) - llm-worker + rbee-keeper
**Primary:**
- TEAM_130B_PHASED_APPROACH.md (Phase 3 section)
- TEAM_130B_EXAMPLE_STRUCTURE.md (template)
- Cross-binary analysis (your Phase 1 output)

**Reference:**
- TEAM_130B_RUST_LIBRARY_SUGGESTIONS.md

**Output:**
- 6 files (3 per binary)

---

## ⚠️ COMMON MISTAKES

### ❌ WRONG: "I'll read TEAM_130B_FINAL_SYNTHESIS.md for the plan"
**Problem:** That's reference material, not the execution plan!  
**✅ RIGHT:** Read TEAM_130B_PHASED_APPROACH.md for the actual plan

---

### ❌ WRONG: "I'll write one 60-page file per binary"
**Problem:** Token limits! Plus that's outdated structure  
**✅ RIGHT:** Write 3 separate files per binary (PART1, PART2, PART3)

---

### ❌ WRONG: "I'll skip Phase 1 and start writing"
**Problem:** You'll miss cross-binary opportunities!  
**✅ RIGHT:** Complete Phase 1 fully, build complete mental model

---

### ❌ WRONG: "Each team can work independently"
**Problem:** Need shared understanding for cross-binary analysis  
**✅ RIGHT:** Collaborate in Phase 1, can split work in Phase 2 & 3

---

## 🎯 QUICK REFERENCE

| Document | Purpose | When to Read |
|----------|---------|--------------|
| README | Overview | First time |
| SUMMARY | High-level plan | Manager review |
| PHASED_APPROACH ⭐ | Execution guide | ALWAYS (primary) |
| EXAMPLE_STRUCTURE | File template | Writing files |
| RUST_LIBRARY_SUGGESTIONS | Library ideas | Writing Part 2 |
| FINAL_SYNTHESIS | Methodology | Deep-dive reference |

---

## 📞 STILL CONFUSED?

**Ask yourself:**
1. "What am I trying to do right now?"
2. Check the "Document Map by Task" section above
3. Read the recommended documents for that task

**Slack:** `#team-130b-questions`

---

**TEAM-130B: When in doubt, TEAM_130B_PHASED_APPROACH.md is your guide! 🎯**
