# Multi-Reference Validation Remediation Checklist
**Created by:** TEAM-004  
**Date:** 2025-10-08 16:28  
**Goal:** Bring multi-reference validation to 75% confidence standard  
**Total Estimated Time:** 5 hours

---

## Overview

Based on TEAM-004 audit findings, we need to complete 4 major tasks to achieve true multi-reference validation with 75% confidence.

**Current State:** 70% confidence (PyTorch-only, 2/6 checkpoints with infrastructure)  
**Target State:** 75% confidence (PyTorch + Candle, 6/6 checkpoints validated)

---

## Task 1: Fix Candle Instrumentation for GPT-2 ⏱️ 2 hours

**Status:** ❌ NOT STARTED  
**Priority:** P0 (blocks all other work)  
**Task File:** `TASK_01_FIX_CANDLE_INSTRUMENTATION.md`

**Problem:**
- Instrumented `bigcode.rs` (StarCoder architecture)
- GPT-2 uses different architecture and weight names
- Cannot extract checkpoints from GPT-2 model

**Deliverables:**
- [ ] Instrument `candle-transformers/src/models/gpt2.rs` (not bigcode.rs)
- [ ] Add checkpoint extraction for LayerNorm (checkpoint 01)
- [ ] Add checkpoint extraction for FFN (checkpoint 06)
- [ ] Test with real GPT-2 model
- [ ] Verify checkpoint files are created in `/tmp/candle_checkpoints/`

**Acceptance Criteria:**
- Running Candle with GPT-2 produces checkpoint files
- Files have correct shape [2, 768]
- No errors about missing tensors

---

## Task 2: Generate and Validate Candle Checkpoints ⏱️ 30 minutes

**Status:** ❌ NOT STARTED  
**Priority:** P0 (required for multi-reference validation)  
**Task File:** `TASK_02_GENERATE_CANDLE_CHECKPOINTS.md`  
**Depends On:** Task 1 (Candle instrumentation must work first)

**Problem:**
- No Candle checkpoint files exist
- Tests fall back to PyTorch-only validation
- Cannot verify cross-validation logic

**Deliverables:**
- [ ] Run Candle with LLORCH_VALIDATE=1 and GPT-2
- [ ] Verify `/tmp/candle_checkpoints/checkpoint_01_ln1_output.npy` created
- [ ] Verify `/tmp/candle_checkpoints/checkpoint_06_ffn.npy` created
- [ ] Copy to `.test-models/gpt2/extracted_weights/` with `_candle.npy` suffix
- [ ] Run tests and verify Candle validation executes (not fallback)

**Acceptance Criteria:**
- Test output shows: `✅ CANDLE: Matches within tolerance`
- Test output shows: `✅ CROSS-VALIDATION: References agree`
- No more `⚠️ Candle reference not available` warnings

---

## Task 3: Add Multi-Reference to Checkpoints 2-5 ⏱️ 2 hours

**Status:** ❌ NOT STARTED  
**Priority:** P1 (required for 75% confidence)  
**Task File:** `TASK_03_MULTI_REFERENCE_REMAINING_CHECKPOINTS.md`  
**Depends On:** Task 2 (need working Candle pipeline first)

**Problem:**
- Only checkpoints 1 and 6 have multi-reference structure
- Checkpoints 2, 3, 4, 5 still single-reference only
- Work is 33% complete (2/6), not "full implementation"

**Deliverables:**
- [ ] Update `real_gpt2_checkpoint_02.rs` with multi-reference test
- [ ] Update `real_gpt2_checkpoint_03.rs` with multi-reference test
- [ ] Update `real_gpt2_checkpoint_04.rs` with multi-reference test
- [ ] Update `real_gpt2_checkpoint_05.rs` with multi-reference test
- [ ] Add Candle instrumentation for checkpoints 2-5 in gpt2.rs
- [ ] Generate Candle checkpoints for 2-5
- [ ] Verify all tests pass with cross-validation

**Acceptance Criteria:**
- All 6 checkpoints have `test_checkpoint_XX_multi_reference()` function
- All tests validate against PyTorch AND Candle
- All tests show cross-validation passing

---

## Task 4: Consolidate Documentation ⏱️ 30 minutes

**Status:** ❌ NOT STARTED  
**Priority:** P2 (hygiene, doesn't block validation)  
**Task File:** `TASK_04_CONSOLIDATE_DOCUMENTATION.md`

**Problem:**
- Created 7 .md files (violates "max 2" rule)
- Files claim "COMPLETE" when work is 33% done
- Repetitive and contradictory content

**Deliverables:**
- [ ] Create single `MULTI_REFERENCE_STATUS.md` with all info
- [ ] Delete 7 redundant files
- [ ] Update status to reflect actual completion (not "COMPLETE")
- [ ] Add this remediation checklist to status doc

**Files to Delete:**
1. `MULTI_REFERENCE_COMPLETE.md`
2. `CANDLE_INSTRUMENTATION_COMPLETE.md`
3. `INSTRUMENTATION_GUIDE.md`
4. `TEAM_003_CORRECTED_FINDINGS.md`
5. `IMPLEMENTATION_COMPLETE.md`
6. `MULTI_REFERENCE_VALIDATION_PLAN.md`
7. `MULTI_REFERENCE_IMPLEMENTATION_STATUS.md`

**Acceptance Criteria:**
- Only 1-2 .md files for multi-reference validation
- No files claiming "COMPLETE" until work is done
- Status accurately reflects 33% → 100% progress

---

## Execution Order

```
Task 1: Fix Candle Instrumentation (2h)
   ↓
Task 2: Generate Candle Checkpoints (30m)
   ↓
Task 3: Multi-Reference for Checkpoints 2-5 (2h)
   ↓
Task 4: Consolidate Documentation (30m)
   ↓
✅ 75% Confidence Achieved
```

**Critical Path:** Tasks 1 → 2 → 3 (must be sequential)  
**Parallel Work:** Task 4 can be done anytime

---

## Success Criteria

### Minimum Viable (70% → 75% confidence)
- [ ] Candle instrumentation works for GPT-2
- [ ] Checkpoints 1 and 6 validate against Candle (not just PyTorch)
- [ ] Cross-validation passes for checkpoints 1 and 6
- [ ] Documentation consolidated to 1-2 files

### Full Standard (75% confidence, all checkpoints)
- [ ] All 6 checkpoints have multi-reference tests
- [ ] All 6 checkpoints validate against PyTorch AND Candle
- [ ] All 6 checkpoints show cross-validation passing
- [ ] No test shows `⚠️ Candle reference not available`
- [ ] Documentation follows project rules

---

## Confidence Level Tracking

| Milestone | Confidence | Rationale |
|-----------|-----------|-----------|
| **Current** | 70% | PyTorch-only validation, 2/6 with infrastructure |
| After Task 1 | 70% | Instrumentation fixed but no checkpoints yet |
| After Task 2 | 73% | Checkpoints 1,6 with dual validation |
| After Task 3 | 75% | All 6 checkpoints with dual validation |
| After Task 4 | 75% | Documentation clean, no confidence change |

---

## Risk Assessment

### High Risk
- **Candle GPT-2 architecture mismatch:** May need significant refactoring
- **Weight name differences:** PyTorch vs Candle naming conventions
- **Numerical precision:** Candle may use different precision than PyTorch

### Medium Risk
- **Time estimate accuracy:** May take longer than 5 hours if issues found
- **Cross-validation tolerance:** References may disagree more than expected

### Low Risk
- **Documentation consolidation:** Straightforward file merging
- **Test structure replication:** Pattern established in checkpoints 1 & 6

---

## Rollback Plan

If remediation fails or takes too long:

1. **Keep current state:** 70% confidence with PyTorch-only validation
2. **Document limitations:** Multi-reference attempted but not achievable
3. **Alternative approach:** Use Mistral.rs instead of Candle
4. **Defer to v0.2.0:** Multi-reference validation as future enhancement

---

## Sign-Off

**Before Starting:**
- [ ] Read all 4 task files
- [ ] Understand dependencies between tasks
- [ ] Allocate 5 hours of focused time
- [ ] Have access to Candle source code

**After Completion:**
- [ ] All tests pass with Candle validation
- [ ] No `⚠️ Candle reference not available` warnings
- [ ] Documentation consolidated
- [ ] Update confidence to 75%
- [ ] Create completion report

---

**Created by:** TEAM-004  
**Status:** Ready for execution  
**Next Step:** Start with `TASK_01_FIX_CANDLE_INSTRUMENTATION.md`
