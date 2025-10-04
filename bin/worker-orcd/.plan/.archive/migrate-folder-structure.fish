#!/usr/bin/env fish
# Migrate M0 Planning Folder Structure - AI Agent Reality
# Date: 2025-10-04
# Purpose: Restructure from human team assumptions to AI agent reality

set -l PLAN_ROOT (dirname (status -f))

echo "🔄 Starting M0 Planning Folder Structure Migration..."
echo "📁 Plan root: $PLAN_ROOT"
echo ""

# ============================================================================
# FOUNDATION TEAM (Foundation-Alpha)
# ============================================================================

echo "🏗️  Migrating Foundation Team..."

cd "$PLAN_ROOT/foundation-team"

# Rename sprint folders with day ranges
if test -d sprints/week-1
    echo "  ✓ Renaming sprint folders..."
    mv sprints/week-1 sprints/sprint-1-http-foundation
    mv sprints/week-2 sprints/sprint-2-ffi-layer
    mv sprints/week-3 sprints/sprint-3-shared-kernels
    mv sprints/week-4 sprints/sprint-4-integration-gate1
    mv sprints/week-5 sprints/sprint-5-support-prep
    mv sprints/week-6 sprints/sprint-6-adapter-gate3
    mv sprints/week-7 sprints/sprint-7-final-integration
end

# Remove old story workflow folders
if test -d stories/backlog
    echo "  ✓ Removing old story workflow folders..."
    rm -rf stories/backlog stories/in-progress stories/review stories/done
end

# Create new story organization by ID range
echo "  ✓ Creating story folders by ID range..."
mkdir -p stories/FT-001-to-FT-010
mkdir -p stories/FT-011-to-FT-020
mkdir -p stories/FT-021-to-FT-030
mkdir -p stories/FT-031-to-FT-040
mkdir -p stories/FT-041-to-FT-050

# Create execution tracking folder
echo "  ✓ Creating execution tracking folder..."
mkdir -p execution

# Create execution tracking files
cat > execution/day-tracker.md << 'EOF'
# Foundation-Alpha - Day Tracker

**Agent**: Foundation-Alpha  
**Total Stories**: 49  
**Total Days**: 89  
**Current Status**: Not Started

---

## Current Progress

**Current Day**: 0 (Not started)  
**Current Story**: None  
**Sprint**: None  
**Stories Completed**: 0 / 49  
**Days Elapsed**: 0 / 89

---

## Today's Work

**Story**: N/A  
**Goal**: N/A  
**Blockers**: None  
**Progress**: Waiting to begin

---

## Recent Completions

None yet.

---

**Last Updated**: 2025-10-04
EOF

cat > execution/dependencies.md << 'EOF'
# Foundation-Alpha - Dependencies

**Agent**: Foundation-Alpha  
**Status**: Ready to start (no upstream dependencies)

---

## Blocking Other Agents

### FFI Lock (Day 15 - After FT-007)

**Blocks**:
- Llama-Beta (cannot start until FFI locked)
- GPT-Gamma (cannot start until FFI locked)

**Action Required**:
- Complete FT-006 (FFI Interface Definition)
- Complete FT-007 (Rust FFI Bindings)
- Publish `FFI_INTERFACE_LOCKED.md`
- Notify Llama-Beta and GPT-Gamma

**Status**: Not started

---

### Integration Framework (Day 52 - After FT-023)

**Blocks**:
- Llama-Beta Gate 1 validation
- GPT-Gamma Gate 1 validation

**Status**: Not started

---

### Adapter Pattern (Day 71 - After FT-035)

**Blocks**:
- Llama-Beta Gate 3 validation
- GPT-Gamma Gate 3 validation

**Status**: Not started

---

## Blocked By

**None** - Foundation-Alpha has no upstream dependencies

---

**Last Updated**: 2025-10-04
EOF

cat > execution/milestones.md << 'EOF'
# Foundation-Alpha - Milestones

**Agent**: Foundation-Alpha  
**Total Days**: 89

---

## Critical Milestones

### Day 15: FFI Interface Locked 🔴 CRITICAL

**Stories**: FT-006, FT-007  
**Status**: ⏳ Pending  
**Impact**: Unblocks Llama-Beta and GPT-Gamma  
**Deliverable**: `FFI_INTERFACE_LOCKED.md`

**Checklist**:
- [ ] FT-006 complete (FFI Interface Definition)
- [ ] FT-007 complete (Rust FFI Bindings)
- [ ] C header finalized
- [ ] Rust bindings tested
- [ ] Usage examples documented
- [ ] `FFI_INTERFACE_LOCKED.md` published
- [ ] Llama-Beta notified
- [ ] GPT-Gamma notified

---

### Day 52: Gate 1 - Foundation Complete 🟡

**Stories**: FT-027  
**Status**: ⏳ Pending  
**Impact**: Foundation infrastructure validated  
**Deliverable**: Gate 1 validation report

**Checklist**:
- [ ] HTTP server operational
- [ ] FFI layer tested
- [ ] CUDA context working
- [ ] Shared kernels validated
- [ ] KV cache functional
- [ ] Integration tests passing

---

### Day 71: Gate 3 - Adapter Pattern Complete 🟡

**Stories**: FT-038  
**Status**: ⏳ Pending  
**Impact**: Enables adapter refactoring for other agents  
**Deliverable**: Gate 3 validation report

**Checklist**:
- [ ] InferenceAdapter interface defined
- [ ] Adapter factory implemented
- [ ] Architecture detection working
- [ ] Integration tests updated

---

### Day 89: Foundation-Alpha Complete ✅

**Stories**: FT-047  
**Status**: ⏳ Pending  
**Impact**: Foundation work done (GPT-Gamma still working)  
**Deliverable**: Gate 4 validation report

**Checklist**:
- [ ] All 49 stories complete
- [ ] CI/CD pipeline operational
- [ ] All integration tests passing
- [ ] Documentation complete

---

**Last Updated**: 2025-10-04
EOF

# Create FFI lock template
cat > execution/FFI_INTERFACE_LOCKED.md.template << 'EOF'
# FFI Interface Locked - Foundation-Alpha

**Date**: [TO BE FILLED]  
**Day**: 15  
**Status**: 🔒 **INTERFACE LOCKED - NO CHANGES ALLOWED**

---

## Interface Components

### C Header (`worker_ffi.h`)

```c
// [TO BE FILLED AFTER FT-006]
```

### Rust Bindings (`ffi.rs`)

```rust
// [TO BE FILLED AFTER FT-007]
```

---

## Usage Examples

### Example 1: Initialize Worker

```rust
// [TO BE FILLED]
```

### Example 2: Execute Inference

```rust
// [TO BE FILLED]
```

---

## Error Handling

[TO BE FILLED]

---

## Memory Management

[TO BE FILLED]

---

## Testing

[TO BE FILLED]

---

**Locked By**: Foundation-Alpha  
**Lock Date**: [TO BE FILLED]  
**Notified**: Llama-Beta ✅ | GPT-Gamma ✅
EOF

echo "  ✅ Foundation Team migration complete"
echo ""

# ============================================================================
# LLAMA TEAM (Llama-Beta)
# ============================================================================

echo "🦙 Migrating Llama Team..."

cd "$PLAN_ROOT/llama-team"

# Create sprint folders (they don't exist yet)
echo "  ✓ Creating sprint folders..."
mkdir -p sprints/sprint-0-prep-work
mkdir -p sprints/sprint-1-gguf-foundation
mkdir -p sprints/sprint-2-gguf-bpe-tokenizer
mkdir -p sprints/sprint-3-utf8-llama-kernels
mkdir -p sprints/sprint-4-gqa-gate1
mkdir -p sprints/sprint-5-qwen-integration
mkdir -p sprints/sprint-6-phi3-adapter
mkdir -p sprints/sprint-7-final-integration

# Create story folders
echo "  ✓ Creating story folders by ID range..."
mkdir -p stories/LT-000-prep
mkdir -p stories/LT-001-to-LT-010
mkdir -p stories/LT-011-to-LT-020
mkdir -p stories/LT-021-to-LT-030
mkdir -p stories/LT-031-to-LT-038

# Create execution tracking
echo "  ✓ Creating execution tracking folder..."
mkdir -p execution

cat > execution/day-tracker.md << 'EOF'
# Llama-Beta - Day Tracker

**Agent**: Llama-Beta  
**Total Stories**: 39  
**Total Days**: 90 (starts day 1, finishes day 90)  
**Current Status**: Waiting for FFI Lock

---

## Current Progress

**Current Day**: 0 (Not started)  
**Current Story**: LT-000 (Prep work)  
**Sprint**: Sprint 0 - Prep Work  
**Stories Completed**: 0 / 39  
**Days Elapsed**: 0 / 90

---

## Blocking Status

**Blocked By**: Foundation-Alpha FT-007 (FFI lock)  
**Expected Unblock**: Day 15  
**Current Activity**: Research llama.cpp, GGUF format, design test framework

---

## Today's Work

**Story**: LT-000 (GGUF Format Research)  
**Goal**: Study llama.cpp implementation, GGUF format spec  
**Blockers**: None (prep work)  
**Progress**: Can start immediately

---

**Last Updated**: 2025-10-04
EOF

cat > execution/dependencies.md << 'EOF'
# Llama-Beta - Dependencies

**Agent**: Llama-Beta  
**Status**: Blocked (waiting for FFI lock)

---

## Blocked By (Upstream)

### FFI Lock (Day 15)

**Blocking Story**: Foundation-Alpha FT-007  
**Impact**: Cannot start GGUF loader work  
**Workaround**: LT-000 prep work (research)  
**Status**: ⏳ Waiting

**When Unblocked**:
- Begin LT-001 (GGUF Header Parser)
- Start Sprint 1 (GGUF Foundation)

---

### Integration Framework (Day 52)

**Blocking Story**: Foundation-Alpha FT-023  
**Impact**: Cannot validate Gate 1  
**Status**: ⏳ Waiting

---

### Adapter Pattern (Day 71)

**Blocking Story**: Foundation-Alpha FT-035  
**Impact**: Cannot implement LlamaInferenceAdapter  
**Status**: ⏳ Waiting

---

## Blocking Others (Downstream)

**None** - Llama-Beta does not block other agents

---

## Internal Dependencies

- LT-006 (GGUF loader) → blocks LT-007 (tokenizer)
- LT-010 (tokenizer) → blocks LT-022 (Qwen integration)
- LT-020 (kernels) → blocks LT-022 (Qwen integration)

---

**Last Updated**: 2025-10-04
EOF

cat > execution/milestones.md << 'EOF'
# Llama-Beta - Milestones

**Agent**: Llama-Beta  
**Total Days**: 90 (starts day 1, finishes day 90)

---

## Critical Milestones

### Day 15: FFI Lock (Unblock) 🔴 CRITICAL

**Dependency**: Foundation-Alpha FT-007  
**Status**: ⏳ Waiting  
**Impact**: Can begin GGUF loader work  
**Action**: Start LT-001 immediately when notified

---

### Day 53: Gate 1 - Llama Kernels Complete 🟡

**Stories**: LT-020  
**Status**: ⏳ Pending  
**Impact**: Llama kernels validated  
**Deliverable**: Gate 1 validation report

---

### Day 66: Gate 2 - Qwen Working 🔴 CRITICAL

**Stories**: LT-027  
**Status**: ⏳ Pending  
**Impact**: First Llama model working end-to-end  
**Deliverable**: Gate 2 validation report

**Checklist**:
- [ ] Qwen2.5-0.5B loads successfully
- [ ] Haiku generation works
- [ ] Reproducibility validated (10 runs)
- [ ] VRAM usage within limits

---

### Day 77: Gate 3 - Adapter Complete 🟡

**Stories**: LT-034  
**Status**: ⏳ Pending  
**Impact**: LlamaInferenceAdapter implemented  
**Deliverable**: Gate 3 validation report

---

### Day 90: Llama-Beta Complete ✅

**Stories**: LT-038  
**Status**: ⏳ Pending  
**Impact**: Llama work done (GPT-Gamma still working)

---

**Last Updated**: 2025-10-04
EOF

echo "  ✅ Llama Team migration complete"
echo ""

# ============================================================================
# GPT TEAM (GPT-Gamma)
# ============================================================================

echo "🤖 Migrating GPT Team..."

cd "$PLAN_ROOT/gpt-team"

# Create sprint folders
echo "  ✓ Creating sprint folders..."
mkdir -p sprints/sprint-0-prep-work
mkdir -p sprints/sprint-1-hf-tokenizer
mkdir -p sprints/sprint-2-gpt-kernels
mkdir -p sprints/sprint-3-mha-gate1
mkdir -p sprints/sprint-4-gpt-basic
mkdir -p sprints/sprint-5-mxfp4-dequant
mkdir -p sprints/sprint-6-mxfp4-integration
mkdir -p sprints/sprint-7-adapter-e2e
mkdir -p sprints/sprint-8-final-integration

# Create story folders
echo "  ✓ Creating story folders by ID range..."
mkdir -p stories/GT-000-prep
mkdir -p stories/GT-001-to-GT-010
mkdir -p stories/GT-011-to-GT-020
mkdir -p stories/GT-021-to-GT-030
mkdir -p stories/GT-031-to-GT-040
mkdir -p stories/GT-041-to-GT-048

# Create execution tracking
echo "  ✓ Creating execution tracking folder..."
mkdir -p execution

cat > execution/day-tracker.md << 'EOF'
# GPT-Gamma - Day Tracker

**Agent**: GPT-Gamma  
**Total Stories**: 49  
**Total Days**: 110 (starts day 1, finishes day 110) ← **M0 CRITICAL PATH**  
**Current Status**: Waiting for FFI Lock

---

## Current Progress

**Current Day**: 0 (Not started)  
**Current Story**: GT-000 (Prep work)  
**Sprint**: Sprint 0 - Prep Work  
**Stories Completed**: 0 / 49  
**Days Elapsed**: 0 / 110

---

## Blocking Status

**Blocked By**: Foundation-Alpha FT-007 (FFI lock)  
**Expected Unblock**: Day 15  
**Current Activity**: Research MXFP4 spec, HF tokenizers crate, design validation framework

---

## Today's Work

**Story**: GT-000 (MXFP4 Spec Study)  
**Goal**: Study MXFP4 format, design validation framework  
**Blockers**: None (prep work)  
**Progress**: Can start immediately

---

## Critical Path Status

**GPT-Gamma determines M0 delivery date**  
**M0 Delivery**: Day 110 (when GPT-Gamma finishes)

---

**Last Updated**: 2025-10-04
EOF

cat > execution/dependencies.md << 'EOF'
# GPT-Gamma - Dependencies

**Agent**: GPT-Gamma  
**Status**: Blocked (waiting for FFI lock)  
**Critical Path**: ⚠️ **GPT-Gamma determines M0 delivery**

---

## Blocked By (Upstream)

### FFI Lock (Day 15)

**Blocking Story**: Foundation-Alpha FT-007  
**Impact**: Cannot start HF tokenizer work  
**Workaround**: GT-000 prep work (MXFP4 research)  
**Status**: ⏳ Waiting

**When Unblocked**:
- Begin GT-001 (HF Tokenizers Crate Integration)
- Start Sprint 1 (HF Tokenizer)

---

### Integration Framework (Day 52)

**Blocking Story**: Foundation-Alpha FT-023  
**Impact**: Cannot validate Gate 1  
**Status**: ⏳ Waiting

---

### Adapter Pattern (Day 71)

**Blocking Story**: Foundation-Alpha FT-035  
**Impact**: Cannot implement GPTInferenceAdapter  
**Status**: ⏳ Waiting

---

## Blocking Others (Downstream)

### M0 Delivery (Day 110)

**GPT-Gamma is the last agent to finish**  
**Impact**: M0 cannot be delivered until GPT-Gamma completes  
**Critical Stories**:
- GT-038 (MXFP4 Numerical Validation)
- GT-040 (GPT-OSS-20B MXFP4 E2E)
- GT-048 (Performance Baseline)

---

## Internal Dependencies

- GT-007 (GPT metadata) → blocks GT-024 (GPT basic)
- GT-023 (kernels) → blocks GT-024 (GPT basic)
- GT-028 (GPT basic) → blocks GT-029 (MXFP4 dequant)
- GT-031 (MXFP4 dequant) → blocks GT-033 (MXFP4 integration)
- GT-038 (MXFP4 integration) → blocks GT-039 (adapter)

---

**Last Updated**: 2025-10-04
EOF

cat > execution/milestones.md << 'EOF'
# GPT-Gamma - Milestones

**Agent**: GPT-Gamma  
**Total Days**: 110 (starts day 1, finishes day 110)  
**Critical Path**: ⚠️ **DETERMINES M0 DELIVERY**

---

## Critical Milestones

### Day 15: FFI Lock (Unblock) 🔴 CRITICAL

**Dependency**: Foundation-Alpha FT-007  
**Status**: ⏳ Waiting  
**Impact**: Can begin HF tokenizer work  
**Action**: Start GT-001 immediately when notified

---

### Day 55: Gate 1 - GPT Kernels Complete 🟡

**Stories**: GT-022  
**Status**: ⏳ Pending  
**Impact**: GPT kernels validated

---

### Day 66: Gate 2 - GPT Basic Working 🔴 CRITICAL

**Stories**: GT-028  
**Status**: ⏳ Pending  
**Impact**: GPT pipeline validated with Q4_K_M baseline  
**Deliverable**: Gate 2 validation report

**Checklist**:
- [ ] GPT-OSS-20B loads with Q4_K_M
- [ ] Generation works
- [ ] Provides baseline for MXFP4 comparison

---

### Day 74: MXFP4 Dequant Complete 🔴 CRITICAL

**Stories**: GT-031  
**Status**: ⏳ Pending  
**Impact**: Novel MXFP4 kernel working  
**Note**: No reference implementation - built from spec

---

### Day 89: MXFP4 Integration Complete 🔴 CRITICAL

**Stories**: GT-038  
**Status**: ⏳ Pending  
**Impact**: MXFP4 wired into all weight consumers  
**Deliverable**: Numerical validation report (±1%)

**Checklist**:
- [ ] MXFP4 in embedding lookup
- [ ] MXFP4 in Q/K/V projections
- [ ] MXFP4 in FFN up/down
- [ ] MXFP4 in LM head
- [ ] Numerical correctness ±1%

---

### Day 96: Gate 3 - MXFP4 + Adapter 🔴 CRITICAL

**Stories**: GT-040  
**Status**: ⏳ Pending  
**Impact**: GPT-OSS-20B with MXFP4 working end-to-end  
**Deliverable**: Gate 3 validation report

**Checklist**:
- [ ] GPT-OSS-20B loads with MXFP4
- [ ] Generation works
- [ ] Numerical correctness validated
- [ ] GPTInferenceAdapter implemented

---

### Day 110: M0 DELIVERY 🎉 CRITICAL

**Stories**: GT-048  
**Status**: ⏳ Pending  
**Impact**: M0 COMPLETE  
**Deliverable**: Gate 4 validation report

**Checklist**:
- [ ] All 49 stories complete
- [ ] All 3 models working (Qwen, Phi-3, GPT-OSS-20B)
- [ ] MXFP4 validated
- [ ] All integration tests passing
- [ ] Documentation complete
- [ ] Performance baseline established

---

**Last Updated**: 2025-10-04
EOF

cat > execution/mxfp4-validation-framework.md << 'EOF'
# MXFP4 Validation Framework

**Agent**: GPT-Gamma  
**Purpose**: Define numerical correctness criteria for MXFP4 implementation

---

## Validation Approach

### Baseline: Q4_K_M

MXFP4 correctness is measured against Q4_K_M baseline (not FP16).

**Rationale**:
- Q4_K_M is already quantized (similar precision class)
- FP16 comparison would be too strict
- Q4_K_M is proven to work

---

## Numerical Correctness Criteria

### Per-Token Logit Difference

**Metric**: Mean Absolute Error (MAE) on logits  
**Threshold**: < 1%  
**Test Set**: 10 diverse prompts

```
MAE = mean(|logits_mxfp4 - logits_q4km|) / mean(|logits_q4km|)
```

**Pass Criteria**: MAE < 0.01 (1%)

---

### Final Output Token Match

**Metric**: Token-level agreement  
**Threshold**: 95% match rate  
**Test Set**: 10 diverse prompts

**Pass Criteria**: At least 9 out of 10 prompts produce same output token as Q4_K_M

---

### Layer-Wise Activation Differences

**Metric**: Max absolute error per layer  
**Threshold**: Document only (no hard limit)  
**Purpose**: Debugging and analysis

**Deliverable**: Layer-wise error report

---

## Test Prompts

[TO BE DEFINED - 10 diverse prompts covering:]
- Short prompts (5-10 tokens)
- Medium prompts (20-30 tokens)
- Long prompts (50+ tokens)
- ASCII text
- UTF-8 multibyte
- Code generation
- Reasoning tasks

---

## Validation Procedure

1. **Load GPT-OSS-20B with Q4_K_M** (baseline)
2. **Run 10 test prompts**, record logits and outputs
3. **Load GPT-OSS-20B with MXFP4**
4. **Run same 10 test prompts**, record logits and outputs
5. **Compute metrics**: MAE, token match rate
6. **Generate report**: Pass/fail with detailed analysis

---

## Acceptance Criteria (GT-038)

- [ ] MAE < 1% on all 10 prompts
- [ ] Token match rate ≥ 95% (9/10 prompts)
- [ ] Layer-wise error report generated
- [ ] Validation report documented
- [ ] Results reproducible (same seed)

---

**Status**: Framework defined, awaiting implementation  
**Owner**: GPT-Gamma  
**Story**: GT-038 (MXFP4 Numerical Validation)

---

**Last Updated**: 2025-10-04
EOF

echo "  ✅ GPT Team migration complete"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

cd "$PLAN_ROOT"

echo "✅ Migration Complete!"
echo ""
echo "📊 Summary:"
echo "  • Foundation Team: 7 sprints, 49 stories, 89 days"
echo "  • Llama Team: 8 sprints (incl prep), 39 stories, 90 days"
echo "  • GPT Team: 9 sprints (incl prep), 49 stories, 110 days ← M0 CRITICAL PATH"
echo ""
echo "📁 New Structure:"
echo "  • Sprints: Named by goal + day range (not week-X)"
echo "  • Stories: Organized by ID range (sequential execution)"
echo "  • Execution: Day tracker, dependencies, milestones"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review execution/ folders for each team"
echo "  2. Create story cards (139 total)"
echo "  3. Set up gate checklists"
echo "  4. Begin Day 1 execution"
echo ""
echo "🚀 Ready to execute!"
