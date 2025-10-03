# Revised Folder Structure - AI Agent Reality

**Date**: 2025-10-04  
**Context**: Folder structure revised to reflect autonomous AI agents, not human teams  
**Status**: 🔄 **RESTRUCTURING REQUIRED**

---

## Executive Summary

**Current Structure**: Based on human team assumptions (weeks, backlog, in-progress, review)  
**Revised Structure**: Based on AI agent reality (sequential sprints, day tracking, dependencies)

**Key Changes**:
1. Sprints named by goal + day range (not "week-1")
2. Stories organized by ID range (sequential execution)
3. New `execution/` folder for day-by-day tracking
4. Remove "backlog/in-progress/review/done" (agent works sequentially)

---

## Foundation Team (Foundation-Alpha)

### Revised Structure

```
foundation-team/
├── README.md                           (Agent profile, 49 stories, 89 days)
├── TEAM_PERSONALITY.md                 (Foundation-Alpha characteristics)
├── REVISION_SUMMARY.md                 (What changed from human assumptions)
│
├── docs/
│   ├── complete-story-list.md          (All 49 stories with day ranges)
│   ├── PLANNING_GAP_ANALYSIS.md        (Revised for AI agent reality)
│   └── team-charter.md                 (Agent capabilities & constraints)
│
├── sprints/
│   ├── sprint-1-http-foundation/       (Days 1-9: 5 stories)
│   │   ├── README.md                   (Sprint goal, stories, timeline)
│   │   ├── FT-001.md                   (HTTP Server Setup)
│   │   ├── FT-002.md                   (POST /execute Endpoint)
│   │   ├── FT-003.md                   (SSE Streaming)
│   │   ├── FT-004.md                   (Correlation ID Middleware)
│   │   └── FT-005.md                   (Request Validation)
│   │
│   ├── sprint-2-ffi-layer/             (Days 10-22: 7 stories) ← FFI LOCK DAY 15
│   │   ├── README.md                   (Sprint goal, FFI lock milestone)
│   │   ├── FT-006.md                   (FFI Interface Definition)
│   │   ├── FT-007.md                   (Rust FFI Bindings) ← LOCK AFTER THIS
│   │   ├── FT-008.md                   (Error Code System C++)
│   │   ├── FT-009.md                   (Error Code to Result Rust)
│   │   ├── FT-010.md                   (CUDA Context Init)
│   │   ├── FT-011.md                   (VRAM-Only Enforcement)
│   │   └── FT-012.md                   (FFI Integration Tests)
│   │
│   ├── sprint-3-shared-kernels/        (Days 23-38: 10 stories)
│   │   ├── README.md
│   │   ├── FT-013.md → FT-020.md       (Kernels)
│   │   ├── FT-049.md                   (Model Load Progress Events) ← NEW
│   │   └── FT-050.md                   (Narration-Core Logging) ← NEW
│   │
│   ├── sprint-4-integration-gate1/     (Days 39-52: 7 stories) ← GATE 1
│   │   ├── README.md
│   │   ├── FT-021.md → FT-027.md       (KV Cache + Integration + Gate 1)
│   │   └── GATE_1_CHECKLIST.md         (Validation criteria)
│   │
│   ├── sprint-5-support-prep/          (Days 53-60: 5 stories)
│   │   ├── README.md
│   │   └── FT-028.md → FT-032.md       (Support + Gate 2)
│   │
│   ├── sprint-6-adapter-gate3/         (Days 61-71: 6 stories) ← GATE 3
│   │   ├── README.md
│   │   ├── FT-033.md → FT-038.md       (Adapter + Gate 3)
│   │   └── GATE_3_CHECKLIST.md
│   │
│   └── sprint-7-final-integration/     (Days 72-89: 9 stories) ← GATE 4
│       ├── README.md
│       ├── FT-039.md → FT-047.md       (CI/CD + Final Integration + Gate 4)
│       └── GATE_4_CHECKLIST.md
│
├── execution/
│   ├── day-tracker.md                  (Current day, current story, progress)
│   ├── dependencies.md                 (Blocking: none, Blocked by: none)
│   ├── milestones.md                   (FFI lock status, gate status)
│   └── FFI_INTERFACE_LOCKED.md         (Published after FT-007)
│
└── integration-gates/
    ├── gate-1-foundation-complete.md   (Day 52 validation)
    ├── gate-3-adapter-complete.md      (Day 71 validation)
    └── gate-4-m0-complete.md           (Day 89 validation)
```

---

## Llama Team (Llama-Beta)

### Revised Structure

```
llama-team/
├── README.md                           (Agent profile, 39 stories, 90 days)
├── TEAM_PERSONALITY.md                 (Llama-Beta characteristics)
├── REVISION_SUMMARY.md
│
├── docs/
│   ├── complete-story-list.md          (All 39 stories with day ranges)
│   └── team-charter.md
│
├── sprints/
│   ├── sprint-0-prep-work/             (Days 1-3: 1 story) ← WAITING FOR FFI
│   │   ├── README.md                   (Research llama.cpp, GGUF format)
│   │   └── LT-000.md                   (GGUF Format Research) ← NEW
│   │
│   ├── sprint-1-gguf-foundation/       (Days 15-25: 6 stories) ← STARTS DAY 15
│   │   ├── README.md                   (Depends on FFI lock)
│   │   └── LT-001.md → LT-006.md       (GGUF Loader)
│   │
│   ├── sprint-2-gguf-bpe-tokenizer/    (Days 26-34: 4 stories)
│   │   ├── README.md
│   │   └── LT-007.md → LT-010.md       (BPE Tokenizer)
│   │
│   ├── sprint-3-utf8-llama-kernels/    (Days 35-40: 4 stories)
│   │   ├── README.md
│   │   └── LT-011.md → LT-014.md       (UTF-8 + Kernels)
│   │
│   ├── sprint-4-gqa-gate1/             (Days 41-53: 7 stories) ← GATE 1
│   │   ├── README.md
│   │   ├── LT-015.md → LT-020.md       (GQA + Gate 1)
│   │   └── GATE_1_CHECKLIST.md
│   │
│   ├── sprint-5-qwen-integration/      (Days 54-66: 6 stories) ← GATE 2
│   │   ├── README.md                   (First Llama model!)
│   │   ├── LT-022.md → LT-027.md       (Qwen + Gate 2)
│   │   └── GATE_2_CHECKLIST.md
│   │
│   ├── sprint-6-phi3-adapter/          (Days 67-77: 6 stories) ← GATE 3
│   │   ├── README.md
│   │   ├── LT-029.md → LT-034.md       (Phi-3 + Adapter + Gate 3)
│   │   └── GATE_3_CHECKLIST.md
│   │
│   └── sprint-7-final-integration/     (Days 78-90: 4 stories)
│       ├── README.md
│       └── LT-035.md → LT-038.md       (Final Testing)
│
├── execution/
│   ├── day-tracker.md                  (Current: Waiting for FFI lock)
│   ├── dependencies.md                 (Blocked by: Foundation-Alpha FT-007)
│   └── milestones.md                   (FFI lock: pending, Gate 2: pending)
│
└── integration-gates/
    ├── gate-1-llama-kernels.md         (Day 53)
    ├── gate-2-qwen-working.md          (Day 66) ← CRITICAL
    └── gate-3-adapter-complete.md      (Day 77)
```

---

## GPT Team (GPT-Gamma)

### Revised Structure

```
gpt-team/
├── README.md                           (Agent profile, 49 stories, 110 days) ← CRITICAL PATH
├── TEAM_PERSONALITY.md                 (GPT-Gamma characteristics)
├── REVISION_SUMMARY.md
│
├── docs/
│   ├── complete-story-list.md          (All 49 stories with day ranges)
│   └── team-charter.md
│
├── sprints/
│   ├── sprint-0-prep-work/             (Days 1-3: 1 story) ← WAITING FOR FFI
│   │   ├── README.md                   (Research MXFP4 spec, HF tokenizers)
│   │   └── GT-000.md                   (MXFP4 Spec Study) ← NEW
│   │
│   ├── sprint-1-hf-tokenizer/          (Days 15-26: 7 stories) ← STARTS DAY 15
│   │   ├── README.md                   (Depends on FFI lock)
│   │   └── GT-001.md → GT-007.md       (HF Tokenizer + GPT Metadata)
│   │
│   ├── sprint-2-gpt-kernels/           (Days 27-41: 9 stories)
│   │   ├── README.md
│   │   └── GT-008.md → GT-016.md       (LayerNorm, GELU, etc.)
│   │
│   ├── sprint-3-mha-gate1/             (Days 42-55: 7 stories) ← GATE 1
│   │   ├── README.md
│   │   ├── GT-017.md → GT-023.md       (MHA + Gate 1)
│   │   └── GATE_1_CHECKLIST.md
│   │
│   ├── sprint-4-gpt-basic/             (Days 56-66: 5 stories) ← GATE 2
│   │   ├── README.md                   (Q4_K_M baseline before MXFP4)
│   │   ├── GT-024.md → GT-028.md       (GPT Basic + Gate 2)
│   │   └── GATE_2_CHECKLIST.md
│   │
│   ├── sprint-5-mxfp4-dequant/         (Days 67-74: 3 stories) ← NOVEL FORMAT
│   │   ├── README.md                   (No reference implementation!)
│   │   ├── GT-029.md                   (MXFP4 Dequantization Kernel)
│   │   ├── GT-030.md                   (MXFP4 Unit Tests)
│   │   └── GT-031.md                   (UTF-8 Streaming Safety)
│   │
│   ├── sprint-6-mxfp4-integration/     (Days 75-89: 6 stories) ← CRITICAL
│   │   ├── README.md                   (Wire into all weight consumers)
│   │   └── GT-033.md → GT-038.md       (MXFP4 Integration + Validation)
│   │
│   ├── sprint-7-adapter-e2e/           (Days 90-96: 2 stories) ← GATE 3
│   │   ├── README.md
│   │   ├── GT-039.md                   (GPTInferenceAdapter)
│   │   ├── GT-040.md                   (GPT-OSS-20B MXFP4 E2E)
│   │   └── GATE_3_CHECKLIST.md
│   │
│   └── sprint-8-final-integration/     (Days 97-110: 8 stories) ← M0 DELIVERY
│       ├── README.md                   (Last agent to finish)
│       ├── GT-041.md → GT-048.md       (Final Testing + Gate 4)
│       └── GATE_4_CHECKLIST.md         (M0 COMPLETE)
│
├── execution/
│   ├── day-tracker.md                  (Current: Waiting for FFI lock)
│   ├── dependencies.md                 (Blocked by: Foundation-Alpha FT-007)
│   ├── milestones.md                   (MXFP4 dequant, MXFP4 integration, M0 delivery)
│   └── mxfp4-validation-framework.md   (Numerical correctness criteria)
│
└── integration-gates/
    ├── gate-1-gpt-kernels.md           (Day 55)
    ├── gate-2-gpt-basic.md             (Day 66)
    ├── gate-3-mxfp4-adapter.md         (Day 96) ← CRITICAL
    └── gate-4-m0-delivery.md           (Day 110) ← M0 COMPLETE
```

---

## Key Differences from Current Structure

### ❌ Removed (Human Team Assumptions)

1. **Week-based folders** (`week-1/`, `week-2/`, etc.)
   - Replaced with: Sprint folders with day ranges

2. **Story workflow folders** (`backlog/`, `in-progress/`, `review/`, `done/`)
   - Replaced with: Stories organized by ID range (sequential execution)
   - Agent doesn't have "backlog" - it has a queue

3. **Team size assumptions** (3-4 people, utilization metrics)
   - Replaced with: Single agent, sequential execution

### ✅ Added (AI Agent Reality)

1. **Sprint-0 prep work** (days 1-14 while waiting for FFI lock)
   - LT-000, GT-000 stories

2. **`execution/` folder** (day-by-day tracking)
   - `day-tracker.md`: Current day, current story
   - `dependencies.md`: What's blocking what
   - `milestones.md`: FFI lock, gates, critical events

3. **Day ranges in sprint names** (`sprint-2-ffi-layer/` → Days 10-22)
   - Makes timeline explicit
   - Shows sequential execution

4. **Critical milestone markers** (← FFI LOCK, ← GATE 1, ← M0 DELIVERY)
   - Highlights coordination points
   - Shows dependencies between agents

---

## Migration Plan

### Phase 1: Restructure Folders (1 day)

**Foundation Team**:
```bash
# Rename sprint folders
mv week-1/ sprint-1-http-foundation/
mv week-2/ sprint-2-ffi-layer/
mv week-3/ sprint-3-shared-kernels/
mv week-4/ sprint-4-integration-gate1/
mv week-5/ sprint-5-support-prep/
mv week-6/ sprint-6-adapter-gate3/
mv week-7/ sprint-7-final-integration/

# Reorganize stories
rm -rf stories/backlog stories/in-progress stories/review stories/done
mkdir -p stories/{FT-001-to-FT-010,FT-011-to-FT-020,FT-021-to-FT-030,FT-031-to-FT-040,FT-041-to-FT-050}

# Create execution folder
mkdir -p execution
touch execution/{day-tracker.md,dependencies.md,milestones.md}
```

**Llama Team & GPT Team**: Similar restructuring

### Phase 2: Create Story Cards (2 days)

For each story (139 total):
- Create `{STORY_ID}.md` in appropriate sprint folder
- Include: Title, Size, Days, Spec Ref, Acceptance Criteria, Dependencies
- Link to spec requirements

### Phase 3: Create Execution Artifacts (1 day)

- `day-tracker.md` templates
- `dependencies.md` with current blocking status
- `milestones.md` with gate criteria
- Gate checklists

### Phase 4: Update Documentation (1 day)

- Update README files in each sprint folder
- Add day ranges and dependencies
- Link to execution tracking

**Total Migration Effort**: 5 days

---

## Benefits of Revised Structure

1. **Clear Sequential Execution**
   - Sprint folders show day ranges
   - Stories organized by execution order
   - No confusion about "backlog" vs "in-progress"

2. **Explicit Dependencies**
   - `execution/dependencies.md` shows blocking relationships
   - Sprint README shows upstream dependencies
   - FFI lock milestone clearly marked

3. **Day-by-Day Tracking**
   - `execution/day-tracker.md` shows current progress
   - Easy to answer "What day are we on?" and "What story is active?"

4. **Milestone Visibility**
   - FFI lock (day 15)
   - Gates (days 52, 66, 77, 96, 110)
   - M0 delivery (day 110)

5. **Agent-Centric**
   - Structure matches how agents work (sequential)
   - No human team artifacts (utilization, capacity, etc.)
   - Focus on execution, not planning

---

## Next Steps

1. ✅ **Approve revised structure** (this document)
2. 🔄 **Execute migration plan** (5 days)
3. ✅ **Create story cards** (139 stories)
4. ✅ **Set up execution tracking**
5. ✅ **Begin agent execution** (Day 1)

---

**Status**: 📋 **PROPOSAL - AWAITING APPROVAL**  
**Estimated Migration Effort**: 5 days  
**Benefit**: Clear, agent-centric structure that matches execution reality

---

*Proposed by Project Manager, reviewed by Narration Core Team 🎀*
