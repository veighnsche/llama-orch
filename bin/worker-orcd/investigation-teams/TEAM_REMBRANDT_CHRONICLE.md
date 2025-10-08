# 🎨 TEAM REMBRANDT - Fix Restoration Chronicle

**Round:** 2  
**Specialization:** Reverted Fix Re-Application  
**Mission:** Re-apply fixes that were incorrectly reverted  
**Status:** ✅ COMPLETE - All fixes already correct, breadcrumbs added

---

## 👥 Team Introduction

**Team Name:** REMBRANDT (after Rembrandt van Rijn, master of restoration and light)

**Why This Name:**
Rembrandt's paintings have been restored many times, bringing back their original brilliance. TEAM REMBRANDT restores code fixes that were lost, bringing back their original correctness.

**Team Philosophy:**
*"Sometimes the right answer was there all along—it just got painted over."*

**Specialization:**
We are the fix restorers. Round 1 had teams that found CORRECT fixes but reverted them because output was still broken due to OTHER bugs. Now that those other bugs are fixed, we restore the correct fixes that were lost.

We don't investigate new bugs. We restore old solutions.

---

## 📋 Mission Briefing

**Objective:** Identify and re-apply any fixes that were reverted but are actually correct

**Why This Matters:**
Multiple teams in Round 1 found correct fixes but concluded they were wrong because output was still broken:
- FELICIA: Reverted CUBLAS_OP_T
- AURORA: Reverted CUBLAS_OP_T with correct lda
- Possibly others

Now that we know the bug was a CONSTELLATION (multiple bugs needed fixing), we need to restore these correct fixes.

**Dependencies:**
- TEAM PICASSO (verdict on cuBLAS)
- TEAM VAN GOGH (verdict on weights)

**Teams Depending On Us:**
- TEAM SHAKESPEARE (needs all fixes applied for integration test)

---

## 📝 Investigation Log

### Session 1: 2025-10-08T00:46Z

**Investigator:** TEAM REMBRANDT (Cascade AI)

**Verdicts from other teams:**
```
[TEAM PICASSO's report - TEAM_PICASSO_CUBLAS_RESOLUTION.md]
- cuBLAS verdict: CUBLAS_OP_T is CORRECT (matches llama.cpp)
- Recommendation: KEEP CUBLAS_OP_T for all 8 matmuls
- Evidence: llama.cpp uses CUBLAS_OP_T and produces perfect output

[TEAM VAN GOGH's report - TEAM_VAN_GOGH_WEIGHT_RESOLUTION.md]
- Weight verdict: RAW weights (mean=7.14, max=16.75) are INTENTIONAL
- Recommendation: DO NOT MODIFY these weights
- Evidence: llama.cpp uses identical weights and works perfectly
```

**What I'm restoring:**
Nothing! After audit, current code already has:
- ✅ All 8 matmuls using CUBLAS_OP_T with correct lda
- ✅ output_norm.weight loaded RAW (no normalization by default)

**Findings:**
1. **cuBLAS Parameters:** All 8 matmuls already use CUBLAS_OP_T
   - Q proj: qwen_transformer.cpp:891 (CUBLAS_OP_T, lda=hidden_dim)
   - K proj: qwen_transformer.cpp:987 (CUBLAS_OP_T, lda=hidden_dim)
   - V proj: qwen_transformer.cpp:1016 (CUBLAS_OP_T, lda=hidden_dim)
   - Attn out: qwen_transformer.cpp:1671 (CUBLAS_OP_T, lda=q_dim)
   - lm_head: qwen_transformer.cpp:2235 (CUBLAS_OP_T, lda=hidden_dim)
   - FFN gate: swiglu_ffn.cu:242 (CUBLAS_OP_T, lda=hidden_dim)
   - FFN up: swiglu_ffn.cu:287 (CUBLAS_OP_T, lda=hidden_dim)
   - FFN down: swiglu_ffn.cu:359 (CUBLAS_OP_T, lda=ffn_dim)

2. **output_norm.weight:** Already loaded RAW
   - Direct load: qwen_weight_loader.cpp:389
   - Pre-loaded wire: qwen_weight_loader.cpp:465
   - A/B test option available via VAN_GOGH_NORMALIZE_OUTPUT_NORM env var

3. **Git History:** No relevant reverts found
   - revert: Only RoPE frequency calculation (unrelated)
   - rollback: Only migration plan doc (unrelated)
   - undo: None found

**Questions/Blockers:**
None! Everything is already in the correct state.

**Next Steps:**
1. ✅ Add REMBRANDT breadcrumbs to lock in current state
2. ✅ Create restoration report documenting no-op restoration
3. Consider adding guardrails (compile-time checks) to prevent regression

---

### Session 2: [Date/Time]

**Investigator:** [Your name/handle]

**What I'm working on:**

**Findings:**

**Questions/Blockers:**

**Next Steps:**

---

## 🔍 Detailed Findings

### 1. cuBLAS Fix Restoration (if needed)

**TEAM PICASSO's verdict:** CUBLAS_OP_T / CUBLAS_OP_N

**Current code state:** CUBLAS_OP_T / CUBLAS_OP_N

**Action needed:**
- ✅ No action (already correct)
- ⚠️ Need to apply CUBLAS_OP_T
- ⚠️ Need to apply CUBLAS_OP_N

**If restoration needed:**

**Changes made:**
```
File: cuda/src/transformer/qwen_transformer.cpp
- Line 327 (Q proj): Changed to CUBLAS_OP_? with lda=???
- Line 361 (K proj): Changed to CUBLAS_OP_? with lda=???
- Line 386 (V proj): Changed to CUBLAS_OP_? with lda=???
- Line 574 (Attn out): Changed to CUBLAS_OP_? with lda=???
- Line 926 (lm_head): Changed to CUBLAS_OP_? with lda=???

File: cuda/kernels/swiglu_ffn.cu
- Line 132 (FFN gate): Changed to CUBLAS_OP_? with lda=???
- Line 151 (FFN up): Changed to CUBLAS_OP_? with lda=???
- Line 181 (FFN down): Changed to CUBLAS_OP_? with lda=???
```

**Test after changes:**
```bash
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  --features cuda --release -- --ignored --nocapture
```

**Result:**
- Test: ✅ PASS / ❌ FAIL
- Output: [first 100 chars]
- Quality: ✅ Coherent / ❌ Garbage / ⚠️ Repetitive

### 2. Weight Fix Restoration (if needed)

**TEAM VAN GOGH's verdict:** Normalized / Raw

**Current code state:** Normalized / Raw

**Action needed:**
- ✅ No action (already correct)
- ⚠️ Need to normalize weights
- ⚠️ Need to use raw weights

**If restoration needed:**

**Changes made:**
```
File: cuda/src/model/qwen_weight_loader.cpp
- Line ???: [describe change]
```

**Test after changes:**
```bash
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  --features cuda --release -- --ignored --nocapture
```

**Result:**
- Test: ✅ PASS / ❌ FAIL
- Output: [first 100 chars]
- Quality: ✅ Coherent / ❌ Garbage / ⚠️ Repetitive

### 3. Other Reverted Fixes Search

**Git history search:**
```bash
cd bin/worker-orcd
git log --all --grep="revert" --oneline
git log --all --grep="undo" --oneline
git log --all --grep="rollback" --oneline
```

**Reverted commits found:**

| Commit | Date | Author | Reason for Revert | Should Restore? |
|--------|------|--------|-------------------|-----------------|
| [hash] | [date] | [name] | [reason] | ✅ / ❌ |
| [hash] | [date] | [name] | [reason] | ✅ / ❌ |

**Commits to restore:**
```
[List commits that should be restored]
```

**Restoration process:**
```bash
# For each commit to restore:
git cherry-pick [hash]
# Or manually re-apply changes
```

### 4. Full Test Suite Verification

**After all restorations:**
```bash
cargo test --features cuda --release
```

**Results:**
- All tests pass: ✅ / ❌
- Failed tests: [list if any]
- Regressions: [list if any]

---

## 🎯 Final Verdict

**Fixes Restored:**
- cuBLAS: ⚠️ Not needed - Already correct (CUBLAS_OP_T for all 8 matmuls)
- Weights: ⚠️ Not needed - Already correct (RAW output_norm)
- Other: None found in git history

**Test Status:**
- Haiku test: ⏸️ Not run (no code changes, only breadcrumbs)
- Full suite: ⏸️ Not run (no code changes)
- Regressions: ✅ None (no code changes)

**Ready for Production:**
- ❌ No - Output still garbage (but not due to cuBLAS/weights)
- Issues: Root bug remains in uninvestigated subsystem (embedding/RoPE/attention)

**Recommendation:**
```
Next investigation round should focus on:
1. Embedding layer (SHAKESPEARE suspects transpose bug)
2. RoPE implementation
3. Attention mask / KV cache
4. Use PICASSO's parity logging to find exact divergence point
```

---

## 📊 Restoration Summary

| Fix | Team That Found It | Was Reverted? | Restored? | Test Result |
|-----|-------------------|---------------|-----------|-------------|
| CUBLAS_OP_T | FELICIA/AURORA/SENTINEL | ✅ / ❌ | ✅ / ❌ | ✅ / ❌ |
| Weight normalization | Output Norm Team | ✅ / ❌ | ✅ / ❌ | ✅ / ❌ |
| [Other] | [Team] | ✅ / ❌ | ✅ / ❌ | ✅ / ❌ |

---

## 📦 Deliverable

**Status:** 🚧 IN PROGRESS / ✅ COMPLETE

**File:** `investigation-teams/TEAM_REMBRANDT_RESTORATION_REPORT.md`

**Handoff To:**
- TEAM SHAKESPEARE (all fixes now applied)
- TEAM WHITMAN (for documentation)

---

## 💭 Reflections

**What Went Well:**
- Efficient audit - found all fixes already correct within minutes
- Clear verdicts from PICASSO and VAN GOGH made verification straightforward
- Breadcrumbs added provide institutional memory for future teams
- No thrashing - codebase stable, teams didn't revert correct fixes

**What Was Challenging:**
- Initially expected to find reverted code to restore
- Had to shift mindset from "restoration" to "validation and lock-in"
- Git history had few relevant commits (good news, but unexpected)

**Lessons Learned:**
1. "No restoration needed" is a valid and positive outcome
2. Breadcrumbs serve as paper trail to prevent re-investigation
3. Multiple teams reaching same conclusion = high confidence
4. "Partial fix is worse than no fix" explains FELICIA/AURORA failures
5. Trust the evidence: If 3+ teams verified something, it's likely correct

**Advice for Future Teams:**
- Read team reports BEFORE attempting to "fix" something already validated
- Compare against llama.cpp ground truth, not just internal consistency
- Add breadcrumbs when you verify something is correct
- Don't cargo cult fixes - understand the WHY
- "Still broken after fix" ≠ "Fix was wrong" (multiple bugs can coexist)

---

**TEAM REMBRANDT**  
*"Sometimes the right answer was there all along—it just got painted over."*

**Chronicle Status:** ✅ COMPLETE  
**Last Updated:** 2025-10-08T00:46Z
