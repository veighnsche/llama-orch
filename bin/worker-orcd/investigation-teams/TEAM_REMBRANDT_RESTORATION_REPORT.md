# TEAM REMBRANDT - Restoration Report

**Date:** 2025-10-08T00:46Z  
**Mission:** Re-apply any CORRECT fixes that were reverted during Round 1  
**Status:** ✅ **COMPLETE** - All fixes already in correct state  

---

## Executive Summary

**VERDICT:** No restoration needed. Current code already contains all correct fixes.

**Key Findings:**
1. ✅ All 8 matmuls use CUBLAS_OP_T with correct lda values (per PICASSO verdict)
2. ✅ output_norm.weight loaded RAW without normalization (per VAN GOGH verdict)
3. ✅ No relevant reverted fixes found in git history
4. ✅ Added REMBRANDT breadcrumbs to lock in current state and prevent regression

**Mission Result:** "NO-OP RESTORATION" - Everything validated as correct, breadcrumbs added for future protection.

---

## Context From Prior Teams

### TEAM PICASSO Verdict (cuBLAS)
**Report:** `investigation-teams/TEAM_PICASSO_CUBLAS_RESOLUTION.md`

**Conclusion:**
- CUBLAS_OP_T is **CORRECT** (matches llama.cpp reference implementation)
- All 8 matmuls should use CUBLAS_OP_T with appropriate lda values
- Bug lies elsewhere (not in cuBLAS parameters)

**Evidence:**
- llama.cpp uses CUBLAS_OP_T and produces perfect output
- Manual verification passed (SENTINEL: diff < 0.001)
- GGUF stores weights in row-major, cuBLAS expects column-major → CUBLAS_OP_T transposes correctly

### TEAM VAN GOGH Verdict (output_norm weights)
**Report:** `investigation-teams/TEAM_VAN_GOGH_WEIGHT_RESOLUTION.md`

**Conclusion:**
- RAW weights (mean=7.14, max=16.75) are **INTENTIONAL** and **CORRECT**
- Do NOT normalize these weights
- The "16.75× amplification" is by design per Qwen2.5 model architecture

**Evidence:**
- GGUF file contains these exact values (byte-for-byte verified)
- llama.cpp uses identical RAW weights and produces perfect output
- TEAM LAMINATOR (Round 1) already reached same conclusion

### TEAM FROST Verdict (sampling)
**Report:** `investigation-teams/TEAM_FROST_SAMPLING_REPORT.md`

**Conclusion:**
- Softmax and sampling pipeline are **CORRECT**
- Bug is upstream (in transformer forward pass)

**Evidence:**
- Softmax sum = 1.0 ± 2e-8 (correct)
- Zero underflow (all 151,936 probs non-zero)
- Correct order: temp → top-k → softmax → top-p(disabled) → sample

### TEAM SHAKESPEARE Verdict (integration)
**Report:** `investigation-teams/TEAM_SHAKESPEARE_INTEGRATION_REPORT.md`

**Conclusion:**
- Output still garbage despite correct cuBLAS/softmax/weights
- Root cause NOT yet found

**Evidence:**
- llama.cpp produces perfect output with same model
- Our engine produces mojibake despite correct parameters
- Bug must be in uninvestigated subsystem

---

## Audit Results

### 1. cuBLAS Parameters - All 8 Matmuls

**Status:** ✅ **ALL CORRECT** (no restoration needed)

| Operation | File:Line | Op | lda | Team Comments |
|-----------|-----------|----|----|---------------|
| Q proj | qwen_transformer.cpp:891 | CUBLAS_OP_T | hidden_dim (896) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |
| K proj | qwen_transformer.cpp:987 | CUBLAS_OP_T | hidden_dim (896) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |
| V proj | qwen_transformer.cpp:1016 | CUBLAS_OP_T | hidden_dim (896) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |
| Attn out | qwen_transformer.cpp:1671 | CUBLAS_OP_T | q_dim (896) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |
| lm_head | qwen_transformer.cpp:2235 | CUBLAS_OP_T | hidden_dim (896) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |
| FFN gate | swiglu_ffn.cu:242 | CUBLAS_OP_T | hidden_dim (896) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |
| FFN up | swiglu_ffn.cu:287 | CUBLAS_OP_T | hidden_dim (896) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |
| FFN down | swiglu_ffn.cu:359 | CUBLAS_OP_T | ffn_dim (4864) | SENTINEL, MONET, PICASSO, DICKINSON, REMBRANDT |

**Action Taken:** Added REMBRANDT breadcrumbs to all 8 locations:
```cpp
// [TEAM REMBRANDT 2025-10-08] Restored/Confirmed: CUBLAS_OP_T with lda=<value> per PICASSO verdict
```

**Why No Restoration Needed:**
TEAM SENTINEL (Round 1) already fixed all 8 matmuls correctly. No subsequent team reverted these changes.

---

### 2. output_norm.weight - RAW Loading

**Status:** ✅ **CORRECT** (no restoration needed)

**Current Code:**

**Location 1:** `cuda/src/model/qwen_weight_loader.cpp:389` (C++ direct loading)
```cpp
// [TEAM VAN GOGH 2025-10-07] Read output_norm wiring here (evidence in VAN_GOGH report)
// This loads output_norm.weight RAW from GGUF file without any normalization
// [TEAM REMBRANDT 2025-10-08] Confirmed RAW output_norm (mean≈7.14) per VAN GOGH verdict
model->weights.output_norm = load_tensor_to_vram(path, "output_norm.weight", tracker);
```

**Location 2:** `cuda/src/model/qwen_weight_loader.cpp:467` (Rust pre-loaded pointers)
```cpp
// [TEAM MONET 2025-10-07T14:22Z] Checked line 393: output_norm loaded raw (no normalization) ⚠️
// [TEAM VAN GOGH 2025-10-07] Read output_norm wiring here (evidence in VAN_GOGH report)
// This wires pre-loaded output_norm.weight pointer - weights come from Rust loader (RAW, no normalization)
// [TEAM REMBRANDT 2025-10-08] Confirmed RAW output_norm (mean≈7.14) per VAN GOGH verdict
model->weights.output_norm = get_ptr("output_norm.weight");
```

**Action Taken:** Added REMBRANDT breadcrumbs to both loading locations.

**Why No Restoration Needed:**
Weights are already loaded RAW. TEAM VAN GOGH added an A/B test option (VAN_GOGH_NORMALIZE_OUTPUT_NORM env var) but the default behavior is correct (RAW).

**A/B Test Option:** Lines 472-508 allow testing normalized weights, but this is intentionally off by default.

---

### 3. Git History Search

**Command:**
```bash
git log --all --grep="revert" --oneline --date=short > /tmp/rembrandt_reverts.log
git log --all --grep="rollback" --oneline --date=short > /tmp/rembrandt_rollbacks.log
git log --all --grep="undo" --oneline --date=short > /tmp/rembrandt_undos.log
```

**Results:**

**Reverts Found:** 1
```
d4fc370 revert: restore original RoPE frequency calculation after mathematical verification
```
**Analysis:** This is a RoPE fix revert, unrelated to cuBLAS or weights. The RoPE team verified the original calculation was correct and restored it. This is the opposite of what we're looking for (a correct fix that was wrongly reverted). This was a wrong fix that was correctly reverted. ✅ No action needed.

**Rollbacks Found:** 1
```
a186246 docs: add worker-http migration plan with implementation details and rollback strategy
```
**Analysis:** Documentation only, not a code fix. ✅ No action needed.

**Undos Found:** 0

**Conclusion:** No relevant reverted fixes found.

---

### 4. Historical Context - Why Teams Thought They Were Wrong

**TEAM FELICIA (2025-10-06T21:57Z):**
- Applied CUBLAS_OP_T but didn't fix ALL 8 matmuls consistently
- Applied CUBLAS_OP_T but used wrong lda values
- Result: Made output WORSE (random garbage → stuck repetition)
- Conclusion: Reverted, thought CUBLAS_OP_T was wrong
- **Reality:** Partial fix is worse than no fix. Need all 8 matmuls fixed together.

**TEAM AURORA (2025-10-06T22:17Z):**
- Applied CUBLAS_OP_T with correct lda for Q/K/V
- But didn't fix FFN or lm_head matmuls
- Result: Exact same stuck repetition as FELICIA
- Conclusion: Reverted, concluded CUBLAS_OP_T definitively wrong
- **Reality:** Still a partial fix. Need ALL 8 matmuls, not just 3.

**TEAM SENTINEL (2025-10-07T23:18Z):**
- Fixed ALL 8 matmuls consistently with CUBLAS_OP_T
- Manual verification passed (diff < 0.001)
- Result: Output STILL garbage
- Conclusion: Kept the fix but noted bug is elsewhere
- **Reality:** Fix was CORRECT but INSUFFICIENT alone. Other bugs masked this fix.

**Lesson Learned:**
"Still broken after fix" ≠ "Fix was wrong". Multiple bugs exist simultaneously. All must be fixed for output to work.

---

## Breadcrumbs Added

To prevent accidental regression, I added breadcrumb comments at all critical locations:

### File: cuda/src/transformer/qwen_transformer.cpp
**Lines modified:** 875, 987, 1018, 1674, 2217

**Breadcrumb format:**
```cpp
// [TEAM REMBRANDT 2025-10-08] Restored/Confirmed: CUBLAS_OP_T with lda=<value> per PICASSO verdict
```

### File: cuda/kernels/swiglu_ffn.cu
**Lines modified:** 242, 288, 361

**Breadcrumb format:**
```cpp
// [TEAM REMBRANDT 2025-10-08] Restored/Confirmed: CUBLAS_OP_T with lda=<value> per PICASSO verdict
```

### File: cuda/src/model/qwen_weight_loader.cpp
**Lines modified:** 389, 466

**Breadcrumb format:**
```cpp
// [TEAM REMBRANDT 2025-10-08] Confirmed RAW output_norm (mean≈7.14) per VAN GOGH verdict
```

**Purpose of Breadcrumbs:**
- Document that REMBRANDT validated these fixes as correct
- Provide date stamp of last restoration verification
- Reference authoritative team verdicts (PICASSO, VAN GOGH)
- Warn future teams NOT to revert these fixes

---

## Restoration Summary

| Fix | Status | Team That Found It | Currently Applied? | Restoration Needed? | Action Taken |
|-----|--------|-------------------|-------------------|-------------------|--------------|
| CUBLAS_OP_T (Q proj) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| CUBLAS_OP_T (K proj) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| CUBLAS_OP_T (V proj) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| CUBLAS_OP_T (Attn out) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| CUBLAS_OP_T (lm_head) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| CUBLAS_OP_T (FFN gate) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| CUBLAS_OP_T (FFN up) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| CUBLAS_OP_T (FFN down) | ✅ CORRECT | SENTINEL | ✅ YES | ❌ NO | Added breadcrumb |
| output_norm RAW weights | ✅ CORRECT | LAMINATOR, VAN GOGH | ✅ YES | ❌ NO | Added breadcrumb |

**Score:** 9/9 fixes already applied and correct ✅

---

## Before/After Diffs

### Q Projection (qwen_transformer.cpp:875)

**Before:**
```cpp
    // [TEAM MONET 2025-10-07T14:22Z] Checked line 873: CUBLAS_OP_T lda=896 ✅
    // [TEAM PICASSO 2025-10-07T14:32Z] Read OP_T + lda=hidden_dim (evidence in PICASSO report)
```

**After:**
```cpp
    // [TEAM MONET 2025-10-07T14:22Z] Checked line 873: CUBLAS_OP_T lda=896 ✅
    // [TEAM PICASSO 2025-10-07T14:32Z] Read OP_T + lda=hidden_dim (evidence in PICASSO report)
    // [TEAM REMBRANDT 2025-10-08] Restored/Confirmed: CUBLAS_OP_T with lda=hidden_dim per PICASSO verdict
```

**Change:** Comment-only (breadcrumb added), no code change

---

### output_norm Loading (qwen_weight_loader.cpp:389)

**Before:**
```cpp
    // [TEAM VAN GOGH 2025-10-07] Read output_norm wiring here (evidence in VAN_GOGH report)
    // This loads output_norm.weight RAW from GGUF file without any normalization
    model->weights.output_norm = load_tensor_to_vram(path, "output_norm.weight", tracker);
```

**After:**
```cpp
    // [TEAM VAN GOGH 2025-10-07] Read output_norm wiring here (evidence in VAN_GOGH report)
    // This loads output_norm.weight RAW from GGUF file without any normalization
    // [TEAM REMBRANDT 2025-10-08] Confirmed RAW output_norm (mean≈7.14) per VAN GOGH verdict
    model->weights.output_norm = load_tensor_to_vram(path, "output_norm.weight", tracker);
```

**Change:** Comment-only (breadcrumb added), no code change

---

*(Similar diffs for all other 7 matmuls and second output_norm location - all comment-only changes)*

---

## Guardrails Considered

**Potential Guardrails to Prevent Regression:**

### Option 1: Compile-Time Assertions
```cpp
// In CMakeLists.txt or compile flags
#ifdef FORBID_OP_N_IN_MATMULS
  #error "CUBLAS_OP_N detected in matmul - should be CUBLAS_OP_T per PICASSO verdict"
#endif
```
**Pros:** Catches regression at compile time  
**Cons:** Requires preprocessor magic, fragile

### Option 2: Runtime Validation (Debug Only)
```cpp
#ifndef NDEBUG
  static bool logged_once = false;
  if (!logged_once) {
    fprintf(stderr, "[GUARD] Q proj using CUBLAS_OP_T (correct)\n");
    logged_once = true;
  }
#endif
```
**Pros:** Simple, low overhead in debug builds  
**Cons:** Only catches issues during testing, not in production

### Option 3: Code Review Checklist
Add to `CONTRIBUTING.md` or PR template:
```markdown
- [ ] If modifying cuBLAS calls, verify CUBLAS_OP_T is used (see TEAM_PICASSO_CUBLAS_RESOLUTION.md)
- [ ] If modifying output_norm loading, verify RAW loading is preserved (see TEAM_VAN_GOGH_WEIGHT_RESOLUTION.md)
```
**Pros:** Human verification, flexible  
**Cons:** Relies on human diligence

### Recommendation
**Option 3 (Code Review Checklist)** is most practical. The breadcrumb comments I added serve as inline documentation that reviewers will see when modifying these lines.

**Decision:** Guardrails deferred. Breadcrumbs + team reports provide sufficient protection for now.

---

## Test Results

**Since no code changes were made (only breadcrumbs added), testing is not required.**

However, for completeness:

### Expected Test Behavior (if we ran tests)
```bash
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  --features cuda --release -- --ignored --nocapture --test-threads=1
```

**Expected Result:** Same as before (garbage output)  
**Reason:** Bug is NOT in cuBLAS or output_norm. Bug is in uninvestigated subsystem (embedding, RoPE, attention mask, etc. per SHAKESPEARE's findings).

---

## Final Verdict

### Fixes Restored
**Count:** 0 (zero)  
**Reason:** All fixes already applied correctly. No restoration needed.

### Fixes Confirmed
**Count:** 9
- 8× CUBLAS_OP_T matmuls (Q, K, V, Attn out, lm_head, FFN gate, FFN up, FFN down)
- 1× output_norm RAW loading

### Breadcrumbs Added
**Count:** 10 (8 matmuls + 2 output_norm loading sites)

### Test Status
**Tests Run:** None (no code changes)  
**Regressions:** None (no code changes)

### Ready for Production
**Status:** ⚠️ **NOT YET** - Output still garbage  
**Reason:** Root bug not yet found (not in cuBLAS/weights/softmax/sampling)

---

## Recommendations

### For Next Investigation Round

**DO NOT investigate:**
- ❌ cuBLAS parameters (confirmed correct by PICASSO)
- ❌ output_norm weights (confirmed correct by VAN GOGH)
- ❌ Softmax (confirmed correct by FROST)
- ❌ Sampling order (confirmed correct by FROST)

**DO investigate:**
1. **Embedding layer** ⚠️ CRITICAL PRIORITY
   - Token ID → embedding vector conversion
   - SHAKESPEARE suspects transpose bug in embedding lookup
   - File: `cuda/kernels/embedding.cu` or similar

2. **RoPE (Rotary Position Embedding)** ⚠️ HIGH PRIORITY
   - Position encoding applied to Q/K
   - Frequency calculation (one revert found - verify it's correct)
   - File: `cuda/kernels/rope.cu`

3. **Attention Mask** ⚠️ MEDIUM PRIORITY
   - Causal masking for autoregressive generation
   - KV cache position handling
   - File: `cuda/src/transformer/qwen_transformer.cpp` (attention section)

4. **Special Token Handling** ⚠️ MEDIUM PRIORITY
   - Chat template currently disabled (hardcoded false)
   - May affect tokenization/interpretation
   - File: `src/inference/cuda_backend.rs:234`

5. **Layer-by-Layer Parity** ⚠️ HIGH PRIORITY
   - Use PICASSO's logging infrastructure to compare with llama.cpp
   - Find exact divergence point
   - Files: `investigation-teams/PARITY_LOGGING_README.md`

---

## Artifacts

**Chronicle:** `investigation-teams/TEAM_REMBRANDT_CHRONICLE.md`  
**Git logs:** `/tmp/rembrandt_{reverts,rollbacks,undos}.log`  
**Modified files:**
- `cuda/src/transformer/qwen_transformer.cpp` (5 breadcrumbs added)
- `cuda/kernels/swiglu_ffn.cu` (3 breadcrumbs added)
- `cuda/src/model/qwen_weight_loader.cpp` (2 breadcrumbs added)

---

## Lessons Learned

### 1. "Already Correct" is a Valid Restoration Outcome
We expected to find reverted fixes to restore. Instead, we found all fixes already correctly applied. This is GOOD NEWS - it means the codebase is stable and teams didn't thrash.

### 2. Breadcrumbs Serve as Institutional Memory
Adding dated comments with team names and verdicts creates a paper trail. Future teams will see:
- What was verified
- When it was verified
- By whom
- Based on what evidence

This prevents re-investigation of already-solved problems.

### 3. Multiple Teams Reaching Same Conclusion = High Confidence
- LAMINATOR → output_norm RAW is correct
- VAN GOGH → output_norm RAW is correct (independent verification)
- SENTINEL → CUBLAS_OP_T is correct (manual verification)
- PICASSO → CUBLAS_OP_T is correct (llama.cpp parity)

When multiple independent investigations converge, confidence increases dramatically.

### 4. "Partial Fix is Worse Than No Fix" Phenomenon
FELICIA and AURORA both applied CUBLAS_OP_T to some (but not all) matmuls. Result: Made output WORSE. This caused them to revert and conclude the fix was wrong.

**Lesson:** When a fix requires consistency across multiple locations, partial application can introduce new bugs. Always apply fixes completely or not at all.

---

## Advice for Future Teams

### If You're Tempted to Revert CUBLAS_OP_T:
1. Read `investigation-teams/TEAM_PICASSO_CUBLAS_RESOLUTION.md` first
2. Compare your findings against llama.cpp ground truth
3. Check if ALL 8 matmuls are using CUBLAS_OP_T consistently
4. Verify you're not debugging a downstream bug that's masking correct behavior

### If You're Tempted to Normalize output_norm.weight:
1. Read `investigation-teams/TEAM_VAN_GOGH_WEIGHT_RESOLUTION.md` first
2. Verify the GGUF file contains these exact values (mean=7.14, max=16.75)
3. Check if llama.cpp uses the same values
4. Consider using the A/B test env var (VAN_GOGH_NORMALIZE_OUTPUT_NORM=1) before permanent changes

### General Advice:
- **Trust the paper trail:** If 3+ teams investigated something and reached consensus, don't second-guess without new evidence
- **Compare against ground truth:** llama.cpp is the gold standard for this model
- **Avoid cargo cult fixes:** Understand WHY a fix is correct, not just that it "looks right"
- **Document your reasoning:** Future you (or future teams) will thank you

---

**TEAM REMBRANDT**  
*"Sometimes the right answer was there all along—it just got painted over."*

**Report Status:** ✅ COMPLETE  
**Date:** 2025-10-08T00:46Z  
**Mission Result:** NO-OP RESTORATION (all fixes already correct, breadcrumbs added)

**Handoff To:**
- Round 3 Investigation Teams (focus on embedding, RoPE, attention mask per recommendations)
- TEAM WHITMAN (documentation updates with restoration results)

---

## Appendix A: Full File/Line Reference

### cuBLAS OP_T Locations (All Verified Correct)

1. **Q projection**
   - File: `bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
   - Line: 891
   - Operation: `CUBLAS_OP_T`
   - lda: `config_.hidden_dim` (896)
   - Breadcrumb: Line 875

2. **K projection**
   - File: `bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
   - Line: 989
   - Operation: `CUBLAS_OP_T`
   - lda: `config_.hidden_dim` (896)
   - Breadcrumb: Line 987

3. **V projection**
   - File: `bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
   - Line: 1019
   - Operation: `CUBLAS_OP_T`
   - lda: `config_.hidden_dim` (896)
   - Breadcrumb: Line 1018

4. **Attention output**
   - File: `bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
   - Line: 1675
   - Operation: `CUBLAS_OP_T`
   - lda: `q_dim` (896)
   - Breadcrumb: Line 1674

5. **LM head**
   - File: `bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
   - Line: 2235
   - Operation: `CUBLAS_OP_T`
   - lda: `config_.hidden_dim` (896)
   - Breadcrumb: Line 2217

6. **FFN gate**
   - File: `bin/worker-orcd/cuda/kernels/swiglu_ffn.cu`
   - Line: 243
   - Operation: `CUBLAS_OP_T`
   - lda: `hidden_dim` (896)
   - Breadcrumb: Line 242

7. **FFN up**
   - File: `bin/worker-orcd/cuda/kernels/swiglu_ffn.cu`
   - Line: 289
   - Operation: `CUBLAS_OP_T`
   - lda: `hidden_dim` (896)
   - Breadcrumb: Line 288

8. **FFN down**
   - File: `bin/worker-orcd/cuda/kernels/swiglu_ffn.cu`
   - Line: 362
   - Operation: `CUBLAS_OP_T`
   - lda: `ffn_dim` (4864)
   - Breadcrumb: Line 361

### output_norm.weight RAW Loading Locations (All Verified Correct)

1. **C++ direct loading**
   - File: `bin/worker-orcd/cuda/src/model/qwen_weight_loader.cpp`
   - Line: 390
   - Method: `load_tensor_to_vram(path, "output_norm.weight", tracker)`
   - Breadcrumb: Line 389

2. **Rust pre-loaded pointer wiring**
   - File: `bin/worker-orcd/cuda/src/model/qwen_weight_loader.cpp`
   - Line: 467
   - Method: `get_ptr("output_norm.weight")`
   - Breadcrumb: Line 466

---

**END OF REPORT**
