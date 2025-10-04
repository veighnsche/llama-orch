# M0 Execution Guide - For Project Managers

**Date**: 2025-10-04  
**Purpose**: Step-by-step guide for executing M0 worker-orcd development  
**Audience**: Project Manager, Team Leads, Future Self  
**Status**: 📋 **READY TO EXECUTE**

---

## Executive Summary

This guide explains **exactly how to execute** the M0 worker-orcd project with three autonomous AI agents working in parallel across separate monitors.

**Critical Understanding**:
- 3 agents work **in parallel** (different monitors)
- Each agent works **sequentially** (one story at a time)
- Timeline is **110 days** (determined by GPT-Gamma)
- Day 15 FFI lock is **critical coordination point**

---

## Table of Contents

1. [Pre-Execution Setup](#pre-execution-setup)
2. [Day 1: Launch All Three Agents](#day-1-launch-all-three-agents)
3. [Days 1-14: Prep Work Phase](#days-1-14-prep-work-phase)
4. [Day 15: FFI Lock Coordination](#day-15-ffi-lock-coordination)
5. [Days 15-110: Parallel Execution](#days-15-110-parallel-execution)
6. [Daily Execution Routine](#daily-execution-routine)
7. [Gate Validation Process](#gate-validation-process)
8. [Dependency Management](#dependency-management)
9. [Troubleshooting](#troubleshooting)

---

## Pre-Execution Setup

### Step 1: Verify Planning Complete

**Checklist**:
- [ ] All 3 planning gaps addressed (FFI lock, missing stories, validation criteria)
- [ ] 139 story cards created (49 Foundation, 39 Llama, 49 GPT, 2 prep)
- [ ] Sprint folders organized by goal + day range
- [ ] Execution tracking files created for all teams
- [ ] Gate checklists prepared
- [ ] FFI lock template ready

**Verification Command**:
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan

# Check story counts
find foundation-team/stories -name "*.md" | wc -l  # Should be 49
find llama-team/stories -name "*.md" | wc -l       # Should be 39
find gpt-team/stories -name "*.md" | wc -l         # Should be 49

# Check execution folders exist
ls foundation-team/execution/
ls llama-team/execution/
ls gpt-team/execution/
```

---

### Step 2: Set Up Monitors

**Physical Setup**:
- **Monitor 1**: Foundation-Alpha workspace
- **Monitor 2**: Llama-Beta workspace
- **Monitor 3**: GPT-Gamma workspace

**Each Monitor Should Have**:
- Terminal with agent prompt ready
- `execution/day-tracker.md` open
- `execution/dependencies.md` open
- Current sprint folder open
- Story card ready

---

### Step 3: Prepare Coordination Documents

**Create Central Tracking**:
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan

# Create central coordination folder
mkdir -p coordination

# Create master timeline
cat > coordination/master-timeline.md << 'EOF'
# M0 Master Timeline

**Start Date**: [FILL IN]  
**Expected End Date**: [START + 110 days]  
**Current Day**: 0

---

## Agent Status

| Agent | Current Day | Current Story | Sprint | Status |
|-------|-------------|---------------|--------|--------|
| Foundation-Alpha | 0 | None | Not started | ⏳ Ready |
| Llama-Beta | 0 | LT-000 | Prep work | ⏳ Ready |
| GPT-Gamma | 0 | GT-000 | Prep work | ⏳ Ready |

---

## Critical Milestones

- [ ] Day 15: FFI Lock (Foundation-Alpha)
- [ ] Day 52: Gate 1 (All agents)
- [ ] Day 66: Gate 2 (Llama-Beta, GPT-Gamma)
- [ ] Day 71: Gate 3 coordination point
- [ ] Day 96: Gate 3 (GPT-Gamma)
- [ ] Day 110: M0 Delivery (GPT-Gamma)

---

**Last Updated**: [FILL IN]
EOF
```

---

## Day 1: Launch All Three Agents

### Morning (9:00 AM)

**Step 1: Foundation-Alpha (Monitor 1)**

```bash
# Navigate to Foundation workspace
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/foundation-team

# Open day tracker
code execution/day-tracker.md

# Update status
# Current Day: 1
# Current Story: FT-001 (HTTP Server Setup)
# Sprint: Sprint 1 - HTTP Foundation

# Open story card
code sprints/sprint-1-http-foundation/FT-001.md

# Launch agent prompt
# "You are Foundation-Alpha. Begin FT-001: HTTP Server Setup. 
#  Review the story card and acceptance criteria. 
#  Work on this story until complete, then report completion."
```

**Step 2: Llama-Beta (Monitor 2)**

```bash
# Navigate to Llama workspace
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/llama-team

# Open day tracker
code execution/day-tracker.md

# Update status
# Current Day: 1
# Current Story: LT-000 (GGUF Format Research)
# Sprint: Sprint 0 - Prep Work
# Status: Waiting for FFI lock (day 15)

# Open story card
code sprints/sprint-0-prep-work/LT-000.md

# Launch agent prompt
# "You are Llama-Beta. Begin LT-000: GGUF Format Research.
#  Study llama.cpp implementation, GGUF format spec, design test framework.
#  You are blocked from GGUF loader work until day 15 (FFI lock).
#  Focus on research and preparation."
```

**Step 3: GPT-Gamma (Monitor 3)**

```bash
# Navigate to GPT workspace
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/gpt-team

# Open day tracker
code execution/day-tracker.md

# Update status
# Current Day: 1
# Current Story: GT-000 (MXFP4 Spec Study)
# Sprint: Sprint 0 - Prep Work
# Status: Waiting for FFI lock (day 15)

# Open story card
code sprints/sprint-0-prep-work/GT-000.md

# Launch agent prompt
# "You are GPT-Gamma. Begin GT-000: MXFP4 Spec Study.
#  Study MXFP4 format spec, HF tokenizers crate, design validation framework.
#  You are blocked from HF tokenizer work until day 15 (FFI lock).
#  Focus on research and preparation."
```

---

### End of Day 1 (5:00 PM)

**Step 4: Update Tracking Documents**

For each agent:
1. Update `execution/day-tracker.md` with progress
2. Note any blockers in `execution/dependencies.md`
3. Update `coordination/master-timeline.md`

**Example Update (Foundation-Alpha)**:
```markdown
## Today's Work (Day 1)

**Story**: FT-001 (HTTP Server Setup)
**Goal**: Set up Axum HTTP server with basic structure
**Progress**: 
- [x] Created Cargo.toml with Axum dependency
- [x] Set up basic server structure
- [ ] Implemented /health endpoint (in progress)
**Blockers**: None
**Estimated Completion**: Day 2 (on track)
```

---

## Days 1-14: Prep Work Phase

### Foundation-Alpha (Days 1-14)

**Focus**: HTTP Foundation + FFI Layer

**Daily Routine**:
1. Check `execution/day-tracker.md` for current story
2. Work on story until complete
3. Mark story complete in tracker
4. Move to next story in sprint
5. Update dependencies if needed

**Critical Path**:
- Days 1-9: Sprint 1 (HTTP Foundation) - 5 stories
- Days 10-14: Sprint 2 (FFI Layer) - Start FT-006, FT-007

**⚠️ CRITICAL**: Must complete FT-007 by day 15 to unlock other agents!

---

### Llama-Beta (Days 1-3)

**Focus**: Prep Work (LT-000)

**Tasks**:
- Study llama.cpp GGUF implementation
- Read GGUF format specification
- Design test framework for GGUF loader
- Design conformance test vectors for BPE tokenizer
- Document findings

**Deliverable**: Research notes + test framework design

**Days 4-14**: Idle (waiting for FFI lock)
- Monitor Foundation-Alpha progress
- Refine test framework design
- Study RoPE, GQA, RMSNorm algorithms

---

### GPT-Gamma (Days 1-3)

**Focus**: Prep Work (GT-000)

**Tasks**:
- Study MXFP4 format specification
- Research HuggingFace tokenizers crate
- Design MXFP4 validation framework
- Define numerical correctness criteria (±1%)
- Document findings

**Deliverable**: Research notes + validation framework design

**Days 4-14**: Idle (waiting for FFI lock)
- Monitor Foundation-Alpha progress
- Refine validation framework
- Study LayerNorm, GELU, MHA algorithms

---

## Day 15: FFI Lock Coordination

### ⚠️ CRITICAL DAY - FFI LOCK

**Morning (9:00 AM)**

**Step 1: Verify Foundation-Alpha Completed FT-007**

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/foundation-team

# Check day tracker
cat execution/day-tracker.md | grep "Current Day"
# Should show: Current Day: 15

# Check FT-007 status
cat execution/day-tracker.md | grep "FT-007"
# Should show: [x] FT-007 complete
```

**If FT-007 NOT complete**: 
- ⚠️ **CRITICAL BLOCKER**
- Llama-Beta and GPT-Gamma remain blocked
- Foundation-Alpha must prioritize FT-007 completion
- Delay other agents' start

---

**Step 2: Foundation-Alpha Publishes FFI Lock**

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/foundation-team/execution

# Foundation-Alpha fills in FFI_INTERFACE_LOCKED.md
# Copy from template, fill in:
# - C header code
# - Rust bindings code
# - Usage examples
# - Error handling patterns
# - Memory management rules

# Publish (copy to shared location)
cp FFI_INTERFACE_LOCKED.md ../../coordination/

# Update dependencies
echo "✅ FFI Lock published on Day 15" >> dependencies.md
```

---

**Step 3: Notify Llama-Beta and GPT-Gamma**

**Llama-Beta (Monitor 2)**:
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/llama-team

# Update day tracker
# Current Day: 15
# Current Story: LT-001 (GGUF Header Parser)
# Sprint: Sprint 1 - GGUF Foundation
# Status: UNBLOCKED - FFI lock received

# Update dependencies
echo "✅ Unblocked by Foundation-Alpha FT-007 on Day 15" >> execution/dependencies.md

# Open story card
code sprints/sprint-1-gguf-foundation/LT-001.md

# Launch agent prompt
# "You are Llama-Beta. FFI interface is now locked. 
#  Begin LT-001: GGUF Header Parser.
#  Reference FFI_INTERFACE_LOCKED.md for interface details."
```

**GPT-Gamma (Monitor 3)**:
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/gpt-team

# Update day tracker
# Current Day: 15
# Current Story: GT-001 (HF Tokenizers Crate Integration)
# Sprint: Sprint 1 - HF Tokenizer
# Status: UNBLOCKED - FFI lock received

# Update dependencies
echo "✅ Unblocked by Foundation-Alpha FT-007 on Day 15" >> execution/dependencies.md

# Open story card
code sprints/sprint-1-hf-tokenizer/GT-001.md

# Launch agent prompt
# "You are GPT-Gamma. FFI interface is now locked.
#  Begin GT-001: HF Tokenizers Crate Integration.
#  Reference FFI_INTERFACE_LOCKED.md for interface details."
```

---

**Step 4: Update Master Timeline**

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/coordination

# Update master-timeline.md
# - Mark "Day 15: FFI Lock" as ✅ Complete
# - Update all three agents' status to "Active"
# - Note: All agents now working in parallel
```

---

## Days 15-110: Parallel Execution

### Daily Execution Pattern

**All Three Agents Work in Parallel**

Each agent follows this pattern independently:

1. **Morning (9:00 AM)**: Check current story
2. **Work**: Execute story until complete
3. **Completion**: Mark story done, move to next
4. **Evening (5:00 PM)**: Update tracking documents
5. **Repeat**: Next day, next story

---

### Foundation-Alpha (Days 15-89)

**Timeline**:
- Days 15-22: Sprint 2 (FFI Layer) - Complete remaining stories
- Days 23-38: Sprint 3 (Shared Kernels + Logging)
- Days 39-52: Sprint 4 (Integration + Gate 1)
- Days 53-60: Sprint 5 (Support + Prep)
- Days 61-71: Sprint 6 (Adapter + Gate 3)
- Days 72-89: Sprint 7 (Final Integration + Gate 4)

**Key Milestones**:
- Day 52: Gate 1 validation
- Day 71: Gate 3 validation (publish adapter pattern)
- Day 89: Foundation-Alpha complete

---

### Llama-Beta (Days 15-90)

**Timeline**:
- Days 15-25: Sprint 1 (GGUF Foundation)
- Days 26-34: Sprint 2 (GGUF-BPE Tokenizer)
- Days 35-40: Sprint 3 (UTF-8 + Llama Kernels)
- Days 41-53: Sprint 4 (GQA + Gate 1)
- Days 54-66: Sprint 5 (Qwen Integration + Gate 2) ← CRITICAL
- Days 67-77: Sprint 6 (Phi-3 + Adapter + Gate 3)
- Days 78-90: Sprint 7 (Final Integration)

**Key Milestones**:
- Day 53: Gate 1 validation
- Day 66: Gate 2 validation (First Llama model working!)
- Day 77: Gate 3 validation
- Day 90: Llama-Beta complete

---

### GPT-Gamma (Days 15-110) ← M0 CRITICAL PATH

**Timeline**:
- Days 15-26: Sprint 1 (HF Tokenizer)
- Days 27-41: Sprint 2 (GPT Kernels)
- Days 42-55: Sprint 3 (MHA + Gate 1)
- Days 56-66: Sprint 4 (GPT Basic + Gate 2)
- Days 67-74: Sprint 5 (MXFP4 Dequant) ← NOVEL FORMAT
- Days 75-89: Sprint 6 (MXFP4 Integration) ← CRITICAL
- Days 90-96: Sprint 7 (Adapter + E2E + Gate 3)
- Days 97-110: Sprint 8 (Final Integration + M0 DELIVERY)

**Key Milestones**:
- Day 55: Gate 1 validation
- Day 66: Gate 2 validation (GPT basic with Q4_K_M)
- Day 74: MXFP4 dequant complete
- Day 89: MXFP4 integration complete
- Day 96: Gate 3 validation (MXFP4 + Adapter)
- Day 110: M0 DELIVERY ← PROJECT COMPLETE

---

## Daily Execution Routine

### For Each Agent (Every Day)

**Morning Routine (9:00 AM)**:

```bash
# 1. Check current day
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/{team-name}
cat execution/day-tracker.md | head -20

# 2. Check dependencies
cat execution/dependencies.md | grep "Blocked"

# 3. Check current story
# Open story card for current story

# 4. Launch agent with prompt
# "You are {Agent-Name}. Day {N}. 
#  Continue/Begin {STORY-ID}: {Story Title}.
#  Review acceptance criteria and work until complete."
```

---

**During Work**:
- Agent works on story
- Agent can work on multiple files simultaneously
- Agent completes story fully before moving to next
- Agent reports completion

---

**Evening Routine (5:00 PM)**:

```bash
# 1. Update day tracker
code execution/day-tracker.md
# - Increment Current Day
# - Update Current Story (if completed)
# - Log progress

# 2. Update dependencies (if needed)
code execution/dependencies.md
# - Mark blockers resolved
# - Note new blockers

# 3. Update master timeline
cd ../../coordination
code master-timeline.md
# - Update agent status
# - Check milestone progress

# 4. Check for coordination needs
# - Gate validation needed?
# - Interface lock needed?
# - Other agents blocked?
```

---

## Gate Validation Process

### Gate 1 (Day 52-55)

**Participants**: All three agents

**Process**:

1. **Foundation-Alpha** (Day 52):
   ```bash
   cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/foundation-team
   
   # Complete FT-027 (Gate 1 Checkpoint)
   # Run gate validation checklist
   code integration-gates/gate-1-foundation-complete.md
   
   # Checklist:
   # - [ ] HTTP server operational
   # - [ ] FFI layer tested
   # - [ ] CUDA context working
   # - [ ] Shared kernels validated
   # - [ ] KV cache functional
   # - [ ] Integration tests passing
   
   # Publish gate report
   cp integration-gates/gate-1-foundation-complete.md ../../coordination/
   ```

2. **Llama-Beta** (Day 53):
   ```bash
   cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/llama-team
   
   # Complete LT-020 (Gate 1 Participation)
   # Run gate validation checklist
   code integration-gates/gate-1-llama-kernels.md
   
   # Checklist:
   # - [ ] GGUF loader working
   # - [ ] BPE tokenizer working
   # - [ ] Llama kernels implemented
   # - [ ] GQA attention working
   # - [ ] Unit tests passing
   
   # Publish gate report
   cp integration-gates/gate-1-llama-kernels.md ../../coordination/
   ```

3. **GPT-Gamma** (Day 55):
   ```bash
   cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/gpt-team
   
   # Complete GT-022 (Gate 1 Participation)
   # Run gate validation checklist
   code integration-gates/gate-1-gpt-kernels.md
   
   # Checklist:
   # - [ ] HF tokenizer working
   # - [ ] GPT kernels implemented
   # - [ ] MHA attention working
   # - [ ] Unit tests passing
   
   # Publish gate report
   cp integration-gates/gate-1-gpt-kernels.md ../../coordination/
   ```

4. **Project Manager** (Day 55):
   ```bash
   cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/coordination
   
   # Review all gate reports
   # Verify all checklists complete
   # Mark Gate 1 as ✅ in master-timeline.md
   
   # If any failures:
   # - Identify root cause
   # - Create fix stories
   # - Block dependent work
   ```

---

### Gate 2 (Day 66)

**Participants**: Llama-Beta, GPT-Gamma

**Purpose**: First models working (Qwen, GPT-OSS-20B with Q4_K_M)

**Process**: Similar to Gate 1, but focused on end-to-end model execution

---

### Gate 3 (Day 71-96)

**Participants**: All three agents (staggered)

**Foundation-Alpha** (Day 71): Adapter pattern complete
**Llama-Beta** (Day 77): LlamaInferenceAdapter complete
**GPT-Gamma** (Day 96): GPTInferenceAdapter + MXFP4 complete

---

### Gate 4 (Day 110)

**Participant**: GPT-Gamma (last to finish)

**Purpose**: M0 DELIVERY

**Process**:
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/gpt-team

# Complete GT-048 (Performance Baseline)
# Run final gate validation
code integration-gates/gate-4-m0-delivery.md

# Checklist:
# - [ ] All 49 stories complete
# - [ ] All 3 models working (Qwen, Phi-3, GPT-OSS-20B)
# - [ ] MXFP4 validated
# - [ ] All integration tests passing
# - [ ] Documentation complete
# - [ ] Performance baseline established

# Publish final report
cp integration-gates/gate-4-m0-delivery.md ../../coordination/

# 🎉 M0 COMPLETE!
```

---

## Dependency Management

### Tracking Dependencies

**Each agent maintains** `execution/dependencies.md`:

```markdown
## Blocked By (Upstream)

- [x] Foundation-Alpha FT-007 (FFI lock) - Unblocked Day 15
- [ ] Foundation-Alpha FT-023 (Integration framework) - Expected Day 52
- [ ] Foundation-Alpha FT-035 (Adapter pattern) - Expected Day 71

## Blocking Others (Downstream)

- None

## Internal Dependencies

- [ ] LT-006 (GGUF loader) → blocks LT-007 (tokenizer)
- [ ] LT-010 (tokenizer) → blocks LT-022 (Qwen integration)
```

---

### When Dependencies Block

**If agent is blocked**:

1. **Check dependencies.md** for blocking story
2. **Check blocking agent's progress**:
   ```bash
   cd ../{blocking-team}/execution
   cat day-tracker.md | grep "{BLOCKING-STORY}"
   ```
3. **Estimate unblock date**
4. **Update own day-tracker** with wait status
5. **Work on non-blocked stories** (if any)
6. **Notify project manager** if critical path affected

---

### Resolving Blocks

**When blocking story completes**:

1. **Blocking agent** updates their dependencies.md
2. **Blocking agent** publishes deliverable (e.g., FFI_INTERFACE_LOCKED.md)
3. **Blocked agent** receives notification
4. **Blocked agent** updates dependencies.md (mark unblocked)
5. **Blocked agent** begins previously blocked work

---

## Troubleshooting

### Problem: Agent Stuck on Story

**Symptoms**: Story taking longer than estimated

**Actions**:
1. Review story acceptance criteria - are they clear?
2. Check for missing dependencies
3. Review agent's progress notes
4. Consider breaking story into smaller pieces
5. Add clarifying notes to story card
6. Extend estimate if needed

**Update tracking**:
```markdown
## Today's Work (Day X)

**Story**: {STORY-ID}
**Original Estimate**: {N} days
**Actual Progress**: Day {M} of {N+X}
**Blocker**: {Description}
**Action**: {What you're doing to resolve}
```

---

### Problem: FFI Lock Delayed

**Symptoms**: Foundation-Alpha hasn't completed FT-007 by day 15

**Impact**: 🔴 CRITICAL - Blocks Llama-Beta and GPT-Gamma

**Actions**:
1. **Immediate**: Foundation-Alpha prioritizes FT-007
2. **Llama-Beta & GPT-Gamma**: Continue prep work, extend research
3. **Update timeline**: Shift all downstream work by delay days
4. **Notify stakeholders**: M0 delivery date shifts

---

### Problem: Gate Validation Fails

**Symptoms**: Gate checklist has unchecked items

**Actions**:
1. **Identify failures**: Which checklist items failed?
2. **Root cause**: Why did they fail?
3. **Create fix stories**: Add to backlog
4. **Prioritize**: Block dependent work until fixed
5. **Re-validate**: Run gate checklist again after fixes

---

### Problem: MXFP4 Numerical Validation Fails

**Symptoms**: GT-038 fails (±1% tolerance not met)

**Impact**: 🔴 CRITICAL - Blocks M0 delivery

**Actions**:
1. **Analyze**: Which prompts failed? What's the error magnitude?
2. **Debug**: Layer-wise error analysis
3. **Fix**: Adjust MXFP4 dequantization kernel
4. **Re-validate**: Run full validation suite
5. **Fallback**: Use Q4_K_M if MXFP4 cannot be fixed (scope reduction)

---

## Summary Checklist

### Before Starting
- [ ] All planning gaps addressed
- [ ] 139 story cards created
- [ ] Execution tracking set up
- [ ] Monitors configured
- [ ] Coordination documents ready

### Day 1
- [ ] Launch all three agents
- [ ] Foundation-Alpha: Begin FT-001
- [ ] Llama-Beta: Begin LT-000 (prep)
- [ ] GPT-Gamma: Begin GT-000 (prep)

### Day 15 (CRITICAL)
- [ ] Foundation-Alpha: FT-007 complete
- [ ] FFI_INTERFACE_LOCKED.md published
- [ ] Llama-Beta: Unblocked, begin LT-001
- [ ] GPT-Gamma: Unblocked, begin GT-001

### Days 15-110
- [ ] Daily tracking updates
- [ ] Gate validations (Days 52, 66, 71, 77, 96, 110)
- [ ] Dependency coordination
- [ ] Blocker resolution

### Day 110
- [ ] GPT-Gamma: GT-048 complete
- [ ] Gate 4 validation passed
- [ ] M0 DELIVERY COMPLETE 🎉

---

## Quick Reference

### Agent Timelines
- **Foundation-Alpha**: Days 1-89 (89 days)
- **Llama-Beta**: Days 1-90 (90 days, starts work day 15)
- **GPT-Gamma**: Days 1-110 (110 days, starts work day 15) ← CRITICAL PATH

### Critical Coordination Points
- **Day 15**: FFI Lock
- **Day 52**: Integration Framework
- **Day 66**: Gate 2 (First models working)
- **Day 71**: Adapter Pattern
- **Day 110**: M0 Delivery

### Daily Commands
```bash
# Check agent status
cat {team}/execution/day-tracker.md | head -20

# Check dependencies
cat {team}/execution/dependencies.md | grep "Blocked"

# Update master timeline
code coordination/master-timeline.md
```

---

**Status**: 📋 **READY TO EXECUTE**  
**Next Action**: Begin Day 1 execution  
**Owner**: Project Manager

---

*Created by Project Manager, for Project Manager 📋*
