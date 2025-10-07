# 🔍 Investigation Chronicle: Garbage Output Bug

**Purpose:** Comprehensive chronological record of all debugging attempts for the "garbage output" bug  
**Generated:** 2025-10-06T22:22Z  
**Status:** Active Investigation - Multiple Teams

---

## 📅 1. Chronological Timeline

### Team Blue (2025-10-06 ~20:56 UTC)

**Mission:** Hunt down garbage output bug (mojibake/repetitive tokens)

**Hypotheses Tested:**
1. BOS token missing from tokenization
2. Special tokens (`<|im_start|>`, `<|im_end|>`) being split by BPE tokenizer

**Changes Made:**
- Added debug logging to `cuda_backend.rs` to trace token IDs
- Modified `src/inference/cuda_backend.rs` lines 148-184 to manually insert special token IDs (151644, 151645)

**Observations:**
- Special tokens were being split into 6 separate tokens: `<` + `|` + `im` + `_start` + `|` + `>`
- After fix: Token[0] = 151644 (single token) ✅
- But model still generates garbage: `ĠsupplementationãĤ¸ãĥ¥_handlesĠLumpà¸Ħà¸²...`

**Conclusions:**
- ✅ **FIXED:** Tokenization bug - special tokens now correctly inserted as single tokens
- ❌ **REMAINING:** Another bug exists in forward pass/transformer logic
- Handoff: Bug must be in embedding lookup, transformer, or final projection

---

### Team Purple (2025-10-06 ~21:12 UTC)

**Mission:** Verify Team Blue's fix and investigate embeddings

**Hypotheses Tested:**
1. Token IDs 151644/151645 are out of bounds (vocab size = 151643)
2. Special token embeddings are zeros or garbage
3. Tokenization approach matters (all-together vs separate parts)
4. Chat template format needs different newlines

**Changes Made:**
- Added logging to verify special token embeddings
- Tested removing newline after "assistant" in chat template

**Observations:**
- Vocab size is actually 151936 (not 151643) - tokens 151644/151645 are VALID ✅
- Special token embeddings have normal values (~0.01 range) ✅
- Token sequence matches llama.cpp format exactly ✅
- Embedding lookup returns correct values ✅

**Conclusions:**
- ✅ **VERIFIED:** Team Blue's token IDs are correct
- ✅ **VERIFIED:** Special token embeddings exist and are valid
- ✅ **VERIFIED:** Token sequence format is correct
- ❌ **REMAINING:** Bug is NOT in tokenization or embeddings - must be deeper in inference pipeline
- Handoff: Focus on forward pass (attention, FFN, KV cache, position encoding)

---

### Team Green (2025-10-06 ~20:38 UTC)

**Mission:** Comprehensive review of all previous findings

**Hypotheses Tested:**
1. Q/K/V biases were missing (model has biases but we weren't loading them)
2. Embedding scaling might be missing

**Changes Made:**
- Added Q/K/V bias loading in `qwen_weight_loader.cpp` lines 356-360
- Added bias addition in `qwen_transformer.cpp` lines 298-354

**Observations:**
- Biases exist in model file ✅
- BUT all bias values are zeros! ✅
- Adding zero biases has no effect on output
- Model still generates mojibake: `è®«æŁ¥æī¾ĠindReactĠScouts...`

**Conclusions:**
- ✅ **FIXED:** Real bug - biases weren't being loaded (but they're all zeros anyway)
- ❌ **FALSE_LEAD:** Adding biases didn't fix garbage output
- ✅ **VERIFIED:** All infrastructure working (cuBLAS, sampling, cache, RoPE, RMSNorm)
- Handoff: Need systematic comparison with llama.cpp to find divergence point

**Files Modified:**
- `cuda/src/model/qwen_weight_loader.cpp` (lines 350-360)
- `cuda/src/transformer/qwen_transformer.cpp` (lines 294-354)

---

### Team Water (2025-10-06 ~17:45 UTC)

**Mission:** Verify KV cache infrastructure

**Hypotheses Tested:**
1. cache_len parameter not being passed correctly
2. Cache writes going to wrong positions
3. Position tracking broken

**Changes Made:**
- Added extensive debug output to `gqa_attention.cu` (wrapper and kernel)
- Added position tracking verification

**Observations:**
- cache_len passes correctly: 0, 1, 2, 3... ✅
- Cache writes to correct positions: 0, 1, 2... ✅
- Position increments correctly ✅
- RoPE applies different rotations per position ✅

**Conclusions:**
- ✅ **VERIFIED:** KV cache infrastructure is correct
- ✅ **VERIFIED:** Position tracking is correct
- ✅ **VERIFIED:** RoPE is working
- ❌ **FALSE_LEAD:** Team Charlie Gamma's clue about cache_len=0 was based on old debug output
- Handoff: Bug is NOT in cache - focus on model computation logic

**Files Modified:**
- `cuda/kernels/gqa_attention.cu` (debug output lines 623-637, 387-391)
- Added [TEAM_WATER] comments in 5 locations

---

### Team Charlie (2025-10-06 ~16:08 UTC)

**Mission:** Mathematical verification of cuBLAS operations

**Hypotheses Tested:**
1. cuBLAS reading wrong memory layout (row-major vs column-major)
2. Leading dimension parameter (lda) is incorrect
3. Matrix multiplication producing wrong results

**Changes Made:**
- Implemented manual dot product computation for verification
- Tested 9 positions including problematic ones (8850, 44394, 137131)

**Observations:**
- Manual computation matches cuBLAS within 0.00002 tolerance ✅
- Position 0: manual=3.197784, cuBLAS=3.197778, diff=0.000006 ✅
- Position 8850: manual=14.264349, cuBLAS=14.264330, diff=0.000019 ✅
- Position 44394: manual=12.341835, cuBLAS=12.341816, diff=0.000019 ✅
- Position 137131: manual=14.712263, cuBLAS=14.712248, diff=0.000015 ✅

**Conclusions:**
- ✅ **VERIFIED:** cuBLAS is computing correctly - the high logits (14+) are mathematically correct!
- ✅ **VERIFIED:** Memory layout is correct
- ❌ **NOT A BUG:** The "garbage" values are actually correct given the inputs
- 🔍 **ROOT CAUSE:** Hidden state is outside normal range [-32.8, 31.2] (should be ±20)
- Handoff: Bug is in hidden state accumulation, likely RMSNorm or residual connections

**Files Modified:**
- Added verification code in `qwen_transformer.cpp` (lines 245-360)

---

### Team Charlie Beta (2025-10-06 ~17:07 UTC)

**Mission:** Find missing weight loading

**Hypotheses Tested:**
1. FFN down projection weight not being loaded

**Changes Made:**
- Added missing line in `qwen_weight_loader.cpp` line 327: `layer.ffn_down = get_ptr(prefix + "ffn_down.weight");`

**Observations:**
- `load()` function loads 4 FFN weights ✅
- `load_from_gpu_pointers()` was only loading 3 FFN weights ❌
- ffn_down was completely missing!

**Conclusions:**
- 🔥 **CRITICAL BUG FOUND:** FFN down projection weight was never loaded
- ⚠️ **NOT TESTED:** Fix applied but not verified due to compilation errors
- Theory: Uninitialized ffn_down pointer caused FFN to use garbage memory
- Handoff: MUST TEST this fix before proceeding

**Files Modified:**
- `cuda/src/model/qwen_weight_loader.cpp` (line 327 - CRITICAL FIX)

---

### Root Cause Investigation (2025-10-06 ~16:15 UTC, Updated 16:32 UTC)

**Mission:** Investigate output_norm.weight corruption

**Hypotheses Tested:**
1. output_norm.weight contains corrupted values (up to 16.75 instead of ~1.0)

**Changes Made:**
- Normalized output_norm.weight to mean=1.0 (scaled by 0.1401)

**Observations:**
- Before fix: output_norm weights range [-0.0114, 16.7500], mean=7.14
- After fix: Hidden state ±4.6 (was ±32.8) ✅
- After fix: Max logit 2.17 (was 14+) ✅
- BUT: Still generates same token repeatedly ❌

**Conclusions:**
- ✅ **PARTIAL FIX:** Corrupted weights made problem worse, but not root cause
- ⚠️ **DEEPER ISSUE:** Even with reasonable logits, model still broken
- Note: Team Charlie later verified mean=7.14 is actually CORRECT for this model
- Handoff: Check if other norm weights (attn_norm, ffn_norm) are also "abnormal"

---

### Peer Review Team (2025-10-06 ~15:36 UTC)

**Mission:** Independent verification of Team Alpha/Charlie findings

**Hypotheses Tested:**
1. cuBLAS verification claims
2. Hidden state range claims
3. Softmax correctness
4. Argmax correctness

**Changes Made:**
- Implemented automated verification tests

**Observations:**
- cuBLAS verification: ✅ VERIFIED (all differences < 0.0001)
- Hidden state range: ⚠️ PARTIALLY VERIFIED ([-32.8, 31.2] vs reported [-13.8, 23.9])
- Softmax: ✅ VERIFIED (weights sum to 1.0 after normalization)
- Argmax: ✅ VERIFIED (correctly finds maximum logit)

**Conclusions:**
- ✅ **CONFIRMED:** All computational components working correctly
- ✅ **CONFIRMED:** This is NOT a code bug in cuBLAS, attention, or argmax
- 🔍 **IMPLICATION:** Issue is model quality, configuration, or tokenizer mismatch
- Recommendation: DO NOT modify cuBLAS, attention, or argmax - they're correct

---

### Team Bygone (2025-10-06 ~21:39 UTC)

**Mission:** Verify causal masking and prefill logic

**Hypotheses Tested:**
1. Causal masking missing in attention kernel
2. Prefill processing tokens one-at-a-time corrupts context
3. Hidden state range slightly outside bounds

**Changes Made:**
- Added FALSE_LEAD markers in code
- Updated FALSE_LEADS_SUMMARY.md with 3 new false leads

**Observations:**
- Decode kernel only attends to positions 0..cache_len ✅ (this IS causal masking)
- Processing tokens one-at-a-time is CORRECT for autoregressive prefill ✅
- Hidden state range [-20.4531, 20.7188] is acceptable ✅
- First generated token is ALREADY wrong (token 14271 "cn" - a code token)

**Conclusions:**
- ✅ **VERIFIED:** Causal masking is implemented correctly
- ✅ **VERIFIED:** Prefill logic is correct
- ✅ **VERIFIED:** Hidden state range is normal
- 🔍 **CRITICAL:** Bug manifests during/immediately after prefill, not during generation
- Handoff: Need systematic llama.cpp comparison to find integration bug

**Files Modified:**
- `cuda/kernels/gqa_attention.cu` (FALSE_LEAD comments lines 253-263)
- `src/inference/cuda_backend.rs` (FALSE_LEAD comments lines 469-479)
- `investigation-teams/FALSE_LEADS_SUMMARY.md` (added leads #5, #6, #7)

---

### Team Felicia (2025-10-06 ~21:50 UTC)

**Mission:** Fix cuBLAS transpose operations

**Hypotheses Tested:**
1. All matrix multiplications using wrong cuBLAS parameters (CUBLAS_OP_N should be CUBLAS_OP_T)
2. GGUF stores weights in row-major, cuBLAS expects column-major

**Changes Made:**
- Changed 8 matrix multiplications from CUBLAS_OP_N to CUBLAS_OP_T
- Adjusted lda parameters
- Files: `qwen_transformer.cpp` lines 279-740, `swiglu_ffn.cu` lines 131-177

**Observations:**
- Before: Random garbage `é¹ŀĠinsultsannersĠLumpæĤĴ...`
- After: Repetitive tokens `macrosmacrosncyĳľĳľĳľĳľĳľ...`
- Output changed from random → repetitive (token 71443 "ĳľ" repeated 20+ times)

**Conclusions:**
- ❌ **FALSE_FIX:** CUBLAS_OP_T made output WORSE (stuck repetition)
- ✅ **REVERTED:** All changes reverted
- 🎯 **PROGRESS:** Output changed (random → repetitive) suggests we're affecting weights
- Handoff: Current CUBLAS_OP_N is correct; focus on KV cache or attention weights

**Files Modified (then reverted):**
- `cuda/src/transformer/qwen_transformer.cpp` (lines 279, 311, 334, 429, 740)
- `cuda/kernels/swiglu_ffn.cu` (lines 131, 149, 177)

---

### Team Aurora (2025-10-06 ~22:17 UTC)

**Mission:** Test CUBLAS_OP_T with correct lda parameters

**Hypotheses Tested:**
1. Team Felicia's CUBLAS_OP_T failed because they used wrong lda values
2. With correct lda (lda=hidden_dim for Q/K/V, etc.), CUBLAS_OP_T should work

**Changes Made:**
- Changed to CUBLAS_OP_T with theoretically correct lda parameters:
  - Q/K/V: lda=hidden_dim (was lda=q_dim/kv_dim)
  - attn_output: lda=q_dim (was lda=hidden_dim)
  - FFN: lda=hidden_dim for gate/up, lda=ffn_dim for down
  - lm_head: lda=hidden_dim (was lda=padded_vocab_size)

**Observations:**
- Before: Random garbage `ĠmotifsĠ×Ĳ×¡×ķ×¨ãĥĲãĤ¹...`
- After: EXACT SAME stuck repetition as Team Felicia! Token 71443 "ĳľ" repeated 5+ times
- cuBLAS verification test FAILED: manual=-0.021, cuBLAS=-2.234, diff=2.21 ❌

**Conclusions:**
- ❌ **FALSE_FIX:** CUBLAS_OP_T approach is definitively WRONG, even with correct lda
- ✅ **CONFIRMED:** Team Felicia was RIGHT to revert
- ✅ **VERIFIED:** Current CUBLAS_OP_N approach is CORRECT
- 🔍 **RECOMMENDATION:** Bug is NOT in cuBLAS - investigate RoPE, RMSNorm, or SwiGLU
- Handoff: Stop investigating cuBLAS transpose - it's a dead end

**Files Modified (then reverted):**
- `cuda/src/transformer/qwen_transformer.cpp` (lines 275-291)

---

## 🧪 2. Key Experiments Table

| Team | Experiment | Hypothesis | Change | Result | Conclusion |
|------|-----------|------------|--------|--------|------------|
| **Blue** | Special Token Fix | Special tokens split by BPE | Manually insert token IDs 151644/151645 | Tokens now single IDs but output still garbage | ✅ Fixed tokenization, ❌ bug remains |
| **Purple** | Token ID Verification | IDs 151644/151645 out of bounds | Verified vocab size = 151936 | Token IDs are valid | ✅ Verified correct |
| **Purple** | Embedding Check | Special token embeddings are zeros | Read embeddings from VRAM | Values ~0.01 (normal) | ✅ Embeddings valid |
| **Green** | Bias Loading | Q/K/V biases missing | Load biases from model | Biases all zeros, no effect | ✅ Fixed bug, ❌ didn't fix output |
| **Water** | Cache Verification | cache_len always 0 | Added debug logging | cache_len = 0,1,2,3... correctly | ✅ Cache working |
| **Charlie** | cuBLAS Verification | Matrix mult wrong | Manual dot product | Matches within 0.00002 | ✅ cuBLAS correct |
| **Charlie Beta** | FFN Weight Loading | ffn_down not loaded | Added missing load line | NOT TESTED YET | ⚠️ Needs testing |
| **Root Cause** | Norm Weight Fix | output_norm corrupted | Normalize to mean=1.0 | Logits better but still broken | ⚠️ Partial fix |
| **Peer Review** | Verification Suite | Validate all claims | Automated tests | All verified | ✅ Confirmed findings |
| **Bygone** | Causal Mask Check | Mask missing | Verified kernel logic | Already implemented | ✅ Verified correct |
| **Bygone** | Prefill Logic | One-at-a-time wrong | Verified approach | Correct for autoregressive | ✅ Verified correct |
| **Felicia** | Matrix Transpose | CUBLAS_OP_N → OP_T | Changed 8 matmuls | Repetitive token output | ❌ Made worse, reverted |
| **Aurora** | Transpose + LDA | Wrong lda with OP_T | OP_T with correct lda | Same repetition as Felicia | ❌ Definitively wrong |

---

## 🚫 3. False Leads Index

### From Code Comments

| File | Line | Team | Description |
|------|------|------|-------------|
| `cuda/src/transformer/qwen_transformer.cpp` | 281 | Felicia | CUBLAS_OP_T made repetition worse |
| `cuda/src/transformer/qwen_transformer.cpp` | 288 | Aurora | CUBLAS_OP_T with correct lda still wrong |
| `cuda/src/transformer/qwen_transformer.cpp` | 296 | Green | Adding Q/K/V biases (all zeros anyway) |
| `cuda/src/transformer/qwen_transformer.cpp` | 328 | Green | K bias addition (biases all zeros) |
| `cuda/src/transformer/qwen_transformer.cpp` | 351 | Green | V bias addition (biases all zeros) |
| `cuda/src/transformer/qwen_transformer.cpp` | 745 | Felicia | Final projection CUBLAS_OP_T |
| `cuda/src/transformer/qwen_transformer.cpp` | 1099 | Purple | Special token embeddings being zeros/garbage |
| `cuda/kernels/gqa_attention.cu` | 253-263 | Bygone | Missing causal mask (already implemented) |
| `src/inference/cuda_backend.rs` | 469-479 | Bygone | Prefill one-at-a-time (correct approach) |

### From FALSE_LEADS_SUMMARY.md

1. **Token IDs Out of Bounds** - Vocab size is 151936, not 151643; tokens 151644/151645 are valid
2. **Special Token Embeddings Are Zeros** - All special tokens have valid FP16 embeddings (~0.01)
3. **Tokenization Approach Matters** - Both approaches produce identical token sequences
4. **Chat Template Format** - Current format matches llama.cpp exactly
5. **Missing Causal Mask** - Decode kernel only processes 0..cache_len (IS causal masking)
6. **Prefill One Token at a Time** - This is CORRECT for autoregressive prefill
7. **Hidden State Range Outside Bounds** - Deviation of 0.4531 is minimal, normal variation
8. **CUBLAS_OP_T with Corrected lda** - Team Aurora tested, got same stuck repetition

---

## 🧠 4. Patterns & Gaps

### Patterns Emerging

1. **Multiple teams suspected cuBLAS but all reverted changes after identical failure modes**
   - Team Felicia: CUBLAS_OP_T → stuck on token 71443 "ĳľ"
   - Team Aurora: CUBLAS_OP_T with correct lda → EXACT SAME stuck repetition
   - Both teams independently confirmed current CUBLAS_OP_N is correct

2. **All infrastructure verified working, yet system fails**
   - Classic integration bug: all parts work individually but system produces garbage
   - Suggests missing operation or subtle parameter mismatch

3. **Output evolution shows progress**
   - Initial: Random garbage (foreign languages, code tokens)
   - After Team Blue: Still garbage but with correct tokenization
   - After Team Felicia experiments: Changed to repetitive patterns (shows we're affecting computation)
   - Suggests we're getting closer but haven't found root cause

4. **First generated token is already wrong**
   - Token 14271 "cn" (code token) instead of haiku-related word
   - Bug manifests during/immediately after prefill, not during generation
   - Indicates problem in forward pass, not in generation loop

5. **Hidden state accumulation pattern**
   - Embedding: ±0.04
   - Layer 10: ±6.8
   - Layer 20: ±18
   - Layer 23: ±23.4
   - After final norm: ±32.8 (with corrupted weights) or ±4.6 (after fix)
   - Exponential growth suggests residual accumulation without proper constraint

### What Hasn't Been Seriously Investigated

1. **RoPE Implementation** - Formula verified correct, but ACTUAL COMPUTATION not compared with llama.cpp
   - Need to dump Q/K values before/after RoPE for first 3 tokens
   - Compare with llama.cpp RoPE output

2. **RMSNorm Implementation** - Formula verified, but epsilon value and exact computation not confirmed
   - Need to verify epsilon matches llama.cpp (should be ~1e-6)
   - Dump intermediate RMS values and compare

3. **SwiGLU Activation** - Less scrutinized than matrix multiplications
   - Need to verify silu(x) = x * sigmoid(x) matches llama.cpp
   - Dump gate, up, and swiglu intermediate values

4. **Weight Tensor Byte Order** - llama.cpp works with same file, we don't
   - Possible endianness problem
   - Possible alignment issues
   - Need to dump raw bytes and compare with llama.cpp

5. **Embedding Scaling** - Some models scale by sqrt(hidden_dim)
   - Our code does direct lookup with NO scaling
   - Need to check if llama.cpp applies scaling factor

6. **Model Configuration Mismatch** - Maybe using wrong architecture parameters
   - Need to verify num_heads, num_kv_heads, head_dim, hidden_dim, ffn_dim
   - Compare with llama.cpp's detected config

### Teams That Kept Repeating Same Mistakes

- **None identified** - Teams generally learned from previous attempts and documented false leads well
- Team Aurora specifically tested Team Felicia's hypothesis with corrections, confirming it was wrong
- FALSE_LEADS_SUMMARY.md effectively prevented redundant investigation

### Critical Untested Fix

- **Team Charlie Beta's ffn_down loading fix** - This could be THE bug, but wasn't tested due to compilation errors
- HIGH PRIORITY: Test this fix immediately
- If ffn_down was never loaded, FFN would use uninitialized memory → explains garbage output

### Recommended Next Steps

1. **URGENT:** Test Team Charlie Beta's ffn_down fix
   - This is the most promising lead
   - Missing weight loading would cause exactly these symptoms

2. **If ffn_down fix doesn't work:** Systematic llama.cpp comparison
   - Add logging at each stage (embedding, layer 0, layer 5, layer 10, etc.)
   - Run llama.cpp with same prompt and logging
   - Find FIRST point where values diverge
   - That's where the bug is

3. **Focus areas for comparison:**
   - RoPE actual computation (not just formula)
   - RMSNorm epsilon and intermediate values
   - SwiGLU intermediate values
   - Embedding scaling factor (if any)

4. **Stop investigating:**
   - ❌ cuBLAS transpose parameters (definitively proven correct)
   - ❌ Tokenization (fixed by Team Blue, verified by Team Purple)
   - ❌ KV cache infrastructure (verified by Team Water)
   - ❌ Causal masking (verified by Team Bygone)
   - ❌ Prefill logic (verified by Team Bygone)

---

## 📊 5. Investigation Statistics

- **Total Teams:** 10+ (Blue, Purple, Green, Water, Charlie, Charlie Beta, Root Cause, Peer Review, Bygone, Felicia, Aurora)
- **Total Experiments:** 13 major experiments
- **Verified Correct:** 11 components (tokenization, embeddings, cuBLAS, cache, RoPE, sampling, etc.)
- **False Leads Documented:** 8+ in code + 8 in FALSE_LEADS_SUMMARY.md
- **Critical Bugs Found:** 2 (special token splitting - FIXED, ffn_down loading - UNTESTED)
- **Partial Fixes:** 1 (output_norm normalization - helped but didn't solve)
- **Time Span:** ~6 hours (20:56 - 22:17 UTC on 2025-10-06)

---

## 🎯 6. Current Status

### What We Know For Certain

✅ **Working Correctly:**
- Tokenization (special tokens as single IDs)
- Token embeddings (valid values, correct lookup)
- cuBLAS matrix multiplication (verified mathematically)
- KV cache infrastructure (positions, reads, writes)
- Position tracking (increments correctly)
- RoPE formula (conceptually correct)
- Causal masking (implemented in kernel)
- Prefill logic (one-at-a-time is correct)
- Attention softmax (weights sum to 1.0)
- Argmax sampling (finds true maximum)
- Residual connections (simple addition)

❌ **Still Broken:**
- Model generates garbage output (mojibake, code tokens)
- Repetitive token generation (same token 10+ times)
- First generated token already wrong (code token instead of haiku word)

### Most Promising Lead

🔥 **Team Charlie Beta's ffn_down Fix** - Missing weight loading for FFN down projection
- Status: Code changed but NOT TESTED
- Impact: If correct, this would fix the entire bug
- Priority: URGENT - test immediately

### If ffn_down Doesn't Fix It

🔍 **Systematic llama.cpp Comparison** - Find divergence point
- All infrastructure verified, so bug must be in subtle implementation detail
- Need to compare intermediate values at each stage
- Focus on RoPE computation, RMSNorm epsilon, SwiGLU activation

---

### Team HYPERION (2025-10-06 ~22:35 UTC)

**Mission:** Deep investigation of RoPE, RMSNorm, SwiGLU, and KV cache

**Hypotheses Tested:**
1. RoPE implementation has runtime bugs
2. RMSNorm epsilon value is wrong
3. SwiGLU activation has implementation errors
4. KV cache has subtle bugs
5. Attention output projection buffer usage causes corruption

**Changes Made:**
- Added investigation comments in `qwen_transformer.cpp` (lines 457-466)
- No code changes - all suspects verified correct

**Observations:**
- RoPE formula: CORRECT (re-confirmed Team Polaris findings) ✅
- RMSNorm epsilon: `1e-6f` matches llama.cpp ✅
- SwiGLU activation: CORRECT implementation ✅
- KV cache: CORRECT (re-confirmed Team Water findings) ✅
- Attention output projection: Inefficient but not buggy ✅
- Model still generates garbage: `_STRUCTUREQSëĨįannersĠgeniÅŁCollector...` ❌
- Logits DO vary across tokens (computation working) ✅
- Hidden state range: `[-20.4531, 20.7188]` (slightly outside bounds) ⚠️

**Conclusions:**
- ✅ **VERIFIED:** All four suspect areas (RoPE, RMSNorm, SwiGLU, KV cache) are CORRECT
- ✅ **VERIFIED:** All formulas match llama.cpp exactly
- ❌ **REMAINING:** Bug is NOT in algorithms or infrastructure
- 🔍 **CRITICAL INSIGHT:** Bug must be in DATA, not LOGIC
- 🎯 **MOST LIKELY:** Weight loading/dequantization differs from llama.cpp
- Handoff: Focus on weight tensor verification (compare our loaded weights with llama.cpp)

**Files Modified:**
- `cuda/src/transformer/qwen_transformer.cpp` (investigation comments only)
- `investigation-teams/TEAM_HYPERION_HANDOFF.md` (comprehensive handoff document)

---

## 📚 7. Key Documents

- `investigation-teams/FALSE_LEADS_SUMMARY.md` - Comprehensive false leads list
- `investigation-teams/TEAM_*_HANDOFF.md` - Individual team handoff documents
- `investigation-teams/PEER_REVIEW_FINAL_REPORT.md` - Independent verification
- `cuda/src/transformer/qwen_transformer.cpp` - Main transformer with extensive comments
- `cuda/kernels/gqa_attention.cu` - Attention kernel with verification comments
- `src/inference/cuda_backend.rs` - Token flow and prefill logic

---

---

### Team ORION (2025-10-06 ~23:53 UTC to 2025-10-07 ~00:06 UTC)

**Mission:** Find first activation divergence between FP16 forward pass and llama.cpp

**Hypotheses Tested:**
1. Divergence occurs in RMSNorm or embedding layers
2. Q/K/V projection outputs are within normal range
3. Q bias contains outliers causing extreme values
4. Q weight matrix has corrupted values

**Changes Made:**
- `cuda/src/transformer/qwen_transformer.cpp` lines 366-826: Added comprehensive activation logging
- Logged min/max/mean for: input, attn_norm, Q/K/V projections (tokens 0 & 1, layer 0)
- Added Q bias verification (lines 786-808)
- Added Q weight first-16 dump (lines 810-825)

**Observations:**
- Token 0: Input normal (±0.056), attn_norm normal (±1.038) ✅
- **Token 0 Q projection: min=-16.047 max=14.336** ❌ EXTREME VALUES!
- **Extremes at Q[95] (head 1, dim 31) and Q[126] (head 1, dim 62)**
- Token 0 K projection: min=-4.645 max=3.166 ⚠️ LARGE
- Token 0 V projection: min=-0.281 max=0.094 ✅ NORMAL
- Token 1 Q projection: min=-3.912 max=3.695 ⚠️ LARGE (same indices!)
- Q bias: ALL ZEROS (min=0.0, max=0.0, mean=0.0) ✅
- Q weight first 16: normal range (±0.01) ✅
- Manual Q[0] verification: manual=-0.043, cuBLAS=-0.043, diff=0.000015 ✅

**Conclusions:**
- ✅ **FOUND DIVERGENCE POINT:** Q projection has extreme values (±16) at specific indices
- ✅ **VERIFIED:** Bias is not the cause (all zeros)
- ✅ **VERIFIED:** Q weight looks normal (first 16 values in ±0.01 range)
- ✅ **VERIFIED:** Q[0] cuBLAS calculation is correct (matches manual)
- 🔍 **CRITICAL PATTERN:** Extremes always at same indices (95, 126) across tokens
- 🔍 **HYPOTHESIS:** cuBLAS reads wrong memory for elements beyond position 0 (stride issue)
- Handoff: Investigate why Q[95] and Q[126] have extremes while Q[0] is correct

**Files Modified:**
- `cuda/src/transformer/qwen_transformer.cpp` (lines 366-826 - ORION logging)

---

### Team THIMBLE (2025-10-07 ~00:18 UTC to ~00:25 UTC)

**Mission:** Test stride hypothesis - does CUBLAS_OP_T stride interpretation cause Q[95]/Q[126] extremes?

**Hypotheses Tested:**
1. CUBLAS_OP_T with lda=hidden_dim causes wrong memory walks past row 0
2. Explicit CPU transpose + CUBLAS_OP_N will fix the extremes
3. Manual dot product will match cuBLAS if stride is correct

**Changes Made:**
- `cuda/src/transformer/qwen_transformer.cpp` lines 6-17: Experiment banner
- Lines 139-151: CPU transpose helper function `cpu_transpose_fp16()`
- Lines 420-614: Pre-transpose experiment code (guarded by `THIMBLE_PRETRANSPOSE_EXPERIMENT`)
- Lines 668-779: Q-projection outlier diagnosis with manual parity checks
- Lines 829-936: Input stats logging and index provenance documentation

**Observations:**
- **Task 1 - Reproducible extremes:** Token 0: Q[95]=-16.047, Q[126]=14.336; Token 1: Q[95]=-3.912, Q[126]=3.695 ✅
- **Task 2 - Manual parity FAILED:**
  - Token 0 Q[95]: manual=-0.058, cuBLAS=-16.047, diff=15.99 ❌
  - Token 0 Q[126]: manual=0.055, cuBLAS=14.336, diff=14.28 ❌
  - Token 1 Q[95]: manual=0.079, cuBLAS=-3.912, diff=3.99 ❌
  - Token 1 Q[126]: manual=0.020, cuBLAS=3.695, diff=3.68 ❌
- **Task 3 - Pre-transpose experiment FAILED:**
  - Explicitly transposed Q weight [896,896] on CPU
  - Used CUBLAS_OP_N with lda=q_dim
  - **Result: IDENTICAL extremes!** Q[95]=-16.047, Q[126]=14.336 (NO CHANGE!)
- Input (normed) stats: Token 0 min=-0.576@741, max=1.038@75, mean=0.003 ✅ NORMAL

**Conclusions:**
- ❌ **STRIDE HYPOTHESIS DISPROVEN:** Extremes persist with both CUBLAS_OP_T and CUBLAS_OP_N
- ✅ **VERIFIED:** Manual FP32 calculation gives correct small values (±0.08)
- ✅ **VERIFIED:** cuBLAS (both OP_T and OP_N) gives wrong extreme values at same indices
- 🔍 **CRITICAL INSIGHT:** Bug is NOT about stride semantics - it's deeper
- 🔍 **NEW HYPOTHESES:** Compute type (tensor cores), weight column corruption, or FP16 overflow
- Handoff: Test CUBLAS_COMPUTE_32F, dump weight columns 95/126, try full FP32 GEMM

**Files Modified:**
- `cuda/src/transformer/qwen_transformer.cpp` (lines 6-17, 139-151, 420-614, 668-936)
- `investigation-teams/TEAM_THIMBLE_SUMMARY.md` (created)

---

### Team TOP HAT (2025-10-07 ~00:30 UTC to ~00:34 UTC)

**Mission:** Eliminate Q[95]/Q[126] extremes by testing 3 remaining hypotheses

**Hypotheses Tested:**
1. Compute type (CUBLAS_COMPUTE_32F_FAST_16F vs CUBLAS_COMPUTE_32F) causes extremes
2. Weight columns 95 and 126 are corrupted in GPU memory
3. Input (normed) has spikes that couple into those columns

**Changes Made:**
- `cuda/src/transformer/qwen_transformer.cpp` lines 420-422: Token counter declaration
- Lines 424-487: Weight column verification (Step 2)
- Lines 466-484: Input hot-spot check (Step 3)
- Lines 546-554: Compute type A/B test macro (Step 1)
- Lines 566-577: Pre-bias Q output logging
- Lines 880-903: Post-projection Q logging with bias checks

**Observations:**
- **H1 Test - Compute type:**
  - CUBLAS_COMPUTE_32F_FAST_16F: Q[95]=-16.047, Q[126]=14.336 ❌
  - CUBLAS_COMPUTE_32F (full FP32): Q[95]=-16.047, Q[126]=14.336 ❌
  - **Result: IDENTICAL extremes with full precision!**
- **H2 Test - Weight columns:**
  - Column 95: min=-0.217, max=0.174, mean=-0.000443 ✅ NORMAL
  - Column 126: min=-0.194, max=0.180, mean=-0.000864 ✅ NORMAL
  - First 16 values all in normal range (|max| < 0.22)
- **H3 Test - Input spikes:**
  - Token 0: normed min=-0.576@741, max=1.038@75, mean=0.003 ✅ NORMAL
  - Token 1: normed min=-0.542@190, max=0.425@75, mean=0.001 ✅ NORMAL
  - No spikes >2 in input
- **Additional finding:** Q before bias already has extremes (not introduced by bias addition)

**Conclusions:**
- ❌ **H1 ELIMINATED:** Tensor-core fast-math is NOT the issue (32F shows same extremes)
- ❌ **H2 ELIMINATED:** Weight columns are NOT corrupted (normal values)
- ❌ **H3 ELIMINATED:** Input does NOT have spikes (normal range)
- ✅ **VERIFIED:** cuBLAS GEMM itself produces extremes, not bias addition
- 🔍 **CORE MYSTERY:** Manual FP32 gives ±0.08, cuBLAS (both FAST_16F and 32F) gives ±16
- 🔍 **ALL STANDARD HYPOTHESES ELIMINATED:** Stride, transpose, compute type, weight corruption, input spikes, bias corruption
- Handoff: Deep cuBLAS audit, memory inspection, or implement workaround while investigating root cause

**Files Modified:**
- `cuda/src/transformer/qwen_transformer.cpp` (lines 420-487, 546-577, 880-903)
- `investigation-teams/TEAM_TOP_HAT_HANDOFF.md` (created)

---

### Team HELIOS (2025-10-08 ~date unknown)

**Mission:** Investigate sampling & generation logic (post-logits phase)

**Hypotheses Tested:**
1. Top-p operates on logits instead of probabilities (wrong order)
2. Top-p softmax normalization only uses first 1000 tokens (broken normalization)
3. Sampling parameters (temperature, seed) not applied correctly

**Changes Made:**
- `cuda/kernels/sampling_wrapper.cu` lines 251-277: Added critical fix banner
- Lines 289-303: Moved softmax BEFORE top-p (architectural fix)
- Lines 305-337: Disabled broken top-p with detailed TODO
- Lines 347-389: Added generation-phase logging (first 20 tokens)

**Observations:**
- **BEFORE:** temperature → top-k → top-p → softmax → sample (WRONG!)
- **AFTER:** temperature → top-k → **softmax → top-p** → sample (CORRECT!)
- Evidence from llama.cpp: `llama-sampling.cpp` line 783 does softmax BEFORE top-p
- Test logs show: temp=0.70 ✅, seeds incrementing ✅, tokens varying ✅
- Probability distribution peaked (expected for temp<1.0) ✅
- **BUT:** Model still generates mojibake: `macros-closeĳľĠminimumĳľ(libraryĳľĳľularitytees...` ❌

**Conclusions:**
- ✅ **FIXED:** Sampling pipeline order (architectural bug)
- ✅ **VERIFIED:** Temperature=0.7 applied correctly
- ✅ **VERIFIED:** Seeds increment correctly (1759794426, 1759794427, ...)
- ✅ **VERIFIED:** Tokens vary (not stuck in loops)
- ✅ **VERIFIED:** Sampling works probabilistically
- ❌ **REMAINING:** Model generates semantically wrong tokens (mojibake)
- 🔍 **CRITICAL INSIGHT:** Sampling is NOT the root cause - bug is in transformer forward pass
- 🔍 **CONCLUSION:** Model computes WRONG logits → sampling correctly picks from wrong distribution
- Handoff: Focus on transformer (attention, RMSNorm, FFN, residual connections, weight application)

**Files Modified:**
- `cuda/kernels/sampling_wrapper.cu` (lines 251-389)
- `investigation-teams/TEAM_HELIOS_FINDINGS.md` (created)
- `investigation-teams/TEAM_HELIOS_SUMMARY.md` (created)
- `investigation-teams/TEAM_HELIOS_HANDOFF.md` (created)

---

## 🧪 2. Key Experiments Table (UPDATED)

| Team | Experiment | Hypothesis | Change | Result | Conclusion |
|------|-----------|------------|--------|--------|------------|
| **Blue** | Special Token Fix | Special tokens split by BPE | Manually insert token IDs 151644/151645 | Tokens now single IDs but output still garbage | ✅ Fixed tokenization, ❌ bug remains |
| **Purple** | Token ID Verification | IDs 151644/151645 out of bounds | Verified vocab size = 151936 | Token IDs are valid | ✅ Verified correct |
| **Purple** | Embedding Check | Special token embeddings are zeros | Read embeddings from VRAM | Values ~0.01 (normal) | ✅ Embeddings valid |
| **Green** | Bias Loading | Q/K/V biases missing | Load biases from model | Biases all zeros, no effect | ✅ Fixed bug, ❌ didn't fix output |
| **Water** | Cache Verification | cache_len always 0 | Added debug logging | cache_len = 0,1,2,3... correctly | ✅ Cache working |
| **Charlie** | cuBLAS Verification | Matrix mult wrong | Manual dot product | Matches within 0.00002 | ✅ cuBLAS correct |
| **Charlie Beta** | FFN Weight Loading | ffn_down not loaded | Added missing load line | NOT TESTED YET | ⚠️ Needs testing |
| **Root Cause** | Norm Weight Fix | output_norm corrupted | Normalize to mean=1.0 | Logits better but still broken | ⚠️ Partial fix |
| **Peer Review** | Verification Suite | Validate all claims | Automated tests | All verified | ✅ Confirmed findings |
| **Bygone** | Causal Mask Check | Mask missing | Verified kernel logic | Already implemented | ✅ Verified correct |
| **Bygone** | Prefill Logic | One-at-a-time wrong | Verified approach | Correct for autoregressive | ✅ Verified correct |
| **Felicia** | Matrix Transpose | CUBLAS_OP_N → OP_T | Changed 8 matmuls | Repetitive token output | ❌ Made worse, reverted |
| **Aurora** | Transpose + LDA | Wrong lda with OP_T | OP_T with correct lda | Same repetition as Felicia | ❌ Definitively wrong |
| **ORION** | Activation Logging | Find divergence point | Log layer 0 activations | Q proj has ±16 extremes at [95,126] | 🔍 Found divergence |
| **ORION** | Q Bias Check | Q bias has outliers | Dump Q bias stats | All zeros (no outliers) | ✅ Bias not the cause |
| **ORION** | Q[0] Manual Verify | cuBLAS params wrong | Manual dot product | Matches cuBLAS (diff=0.000015) | ✅ Q[0] is correct |
| **THIMBLE** | Pre-transpose Experiment | CUBLAS_OP_T stride issue | CPU transpose + OP_N | IDENTICAL extremes | ❌ Stride hypothesis disproven |
| **THIMBLE** | Manual Parity Q[95] | cuBLAS reads wrong memory | Manual dot product | manual=±0.08, cuBLAS=±16 | ❌ cuBLAS gives wrong values |
| **TOP HAT** | Compute Type Test | Tensor-core fast-math | 32F_FAST_16F vs 32F | IDENTICAL extremes | ❌ Compute type not the issue |
| **TOP HAT** | Weight Column Dump | Columns 95/126 corrupted | Dump column stats | Both normal (|max|<0.22) | ❌ Weights not corrupted |
| **TOP HAT** | Input Hot-Spot Check | Normed has spikes | Log normed min/max/mean | Normal range (±1) | ❌ Input not the cause |
| **HELIOS** | Sampling Order Fix | Top-p before softmax | Move softmax before top-p | Sampling works, output still mojibake | ✅ Fixed sampling, ❌ bug upstream |
| **HELIOS** | Top-P Disable | Broken normalization | Disable top-p filtering | No change (test uses top_p=1.0) | ✅ Safe workaround |

---

## 🚫 3. False Leads Index (UPDATED)

### From Code Comments

| File | Line | Team | Description |
|------|------|------|-------------|
| `cuda/src/transformer/qwen_transformer.cpp` | 281 | Felicia | CUBLAS_OP_T made repetition worse |
| `cuda/src/transformer/qwen_transformer.cpp` | 288 | Aurora | CUBLAS_OP_T with correct lda still wrong |
| `cuda/src/transformer/qwen_transformer.cpp` | 296 | Green | Adding Q/K/V biases (all zeros anyway) |
| `cuda/src/transformer/qwen_transformer.cpp` | 328 | Green | K bias addition (biases all zeros) |
| `cuda/src/transformer/qwen_transformer.cpp` | 351 | Green | V bias addition (biases all zeros) |
| `cuda/src/transformer/qwen_transformer.cpp` | 745 | Felicia | Final projection CUBLAS_OP_T |
| `cuda/src/transformer/qwen_transformer.cpp` | 1099 | Purple | Special token embeddings being zeros/garbage |
| `cuda/src/transformer/qwen_transformer.cpp` | 6-17 | THIMBLE | Pre-transpose experiment (stride hypothesis) |
| `cuda/src/transformer/qwen_transformer.cpp` | 20-30 | TOP HAT | All 3 hypotheses (compute type, weight corruption, input spikes) |
| `cuda/kernels/gqa_attention.cu` | 253-263 | Bygone | Missing causal mask (already implemented) |
| `cuda/kernels/sampling_wrapper.cu` | 251-277 | HELIOS | Top-p before softmax (architectural bug - FIXED) |
| `src/inference/cuda_backend.rs` | 469-479 | Bygone | Prefill one-at-a-time (correct approach) |

### From FALSE_LEADS_SUMMARY.md

1. **Token IDs Out of Bounds** - Vocab size is 151936, not 151643; tokens 151644/151645 are valid
2. **Special Token Embeddings Are Zeros** - All special tokens have valid FP16 embeddings (~0.01)
3. **Tokenization Approach Matters** - Both approaches produce identical token sequences
4. **Chat Template Format** - Current format matches llama.cpp exactly
5. **Missing Causal Mask** - Decode kernel only processes 0..cache_len (IS causal masking)
6. **Prefill One Token at a Time** - This is CORRECT for autoregressive prefill
7. **Hidden State Range Outside Bounds** - Deviation of 0.4531 is minimal, normal variation
8. **CUBLAS_OP_T with Corrected lda** - Team Aurora tested, got same stuck repetition
9. **CUBLAS_OP_T Stride Interpretation** - Team THIMBLE tested with explicit transpose, same extremes
10. **Tensor-Core Fast-Math** - Team TOP HAT tested CUBLAS_COMPUTE_32F, same extremes
11. **Weight Column Corruption** - Team TOP HAT dumped columns 95/126, both normal
12. **Input Spikes in Normed** - Team TOP HAT verified normed is normal (±1 range)
13. **Sampling Pipeline Order** - Team HELIOS fixed (softmax before top-p), but output still broken

---

## 🧠 4. Patterns & Gaps (UPDATED)

### Patterns Emerging

1. **Multiple teams suspected cuBLAS but all reverted changes after identical failure modes**
   - Team Felicia: CUBLAS_OP_T → stuck on token 71443 "ĳľ"
   - Team Aurora: CUBLAS_OP_T with correct lda → EXACT SAME stuck repetition
   - Both teams independently confirmed current CUBLAS_OP_N is correct

2. **All infrastructure verified working, yet system fails**
   - Classic integration bug: all parts work individually but system produces garbage
   - Suggests missing operation or subtle parameter mismatch

3. **Output evolution shows progress**
   - Initial: Random garbage (foreign languages, code tokens)
   - After Team Blue: Still garbage but with correct tokenization
   - After Team Felicia experiments: Changed to repetitive patterns (shows we're affecting computation)
   - Suggests we're getting closer but haven't found root cause

4. **First generated token is already wrong**
   - Token 14271 "cn" (code token) instead of haiku-related word
   - Bug manifests during/immediately after prefill, not during generation
   - Indicates problem in forward pass, not in generation loop

5. **Hidden state accumulation pattern**
   - Embedding: ±0.04
   - Layer 10: ±6.8
   - Layer 20: ±18
   - Layer 23: ±23.4
   - After final norm: ±32.8 (with corrupted weights) or ±4.6 (after fix)
   - Exponential growth suggests residual accumulation without proper constraint

6. **Q-projection extremes are highly localized** (NEW)
   - Extremes always at Q[95] (head 1, dim 31) and Q[126] (head 1, dim 62)
   - Same indices across all tokens (token 0, token 1, etc.)
   - Q[0] is correct (manual verification passes)
   - Suggests bug affects specific output positions, not all positions

7. **All standard debugging eliminated the obvious causes** (NEW)
   - Stride/transpose: THIMBLE tested explicit transpose, same result
   - Compute type: TOP HAT tested full FP32, same result
   - Weight corruption: TOP HAT dumped columns, both normal
   - Input spikes: TOP HAT verified normed is normal
   - Manual calculation works (±0.08) but cuBLAS fails (±16)

8. **Sampling architecture was wrong but not the root cause** (NEW)
   - HELIOS fixed sampling order (softmax before top-p)
   - Temperature, seeds, token variety all correct
   - But output still mojibake (semantically wrong tokens)
   - Proves bug is upstream in transformer forward pass

### What Hasn't Been Seriously Investigated

1. **RoPE Implementation** - Formula verified correct, but ACTUAL COMPUTATION not compared with llama.cpp
   - Need to dump Q/K values before/after RoPE for first 3 tokens
   - Compare with llama.cpp RoPE output

2. **RMSNorm Implementation** - Formula verified, but epsilon value and exact computation not confirmed
   - Need to verify epsilon matches llama.cpp (should be ~1e-6)
   - Dump intermediate RMS values and compare

3. **SwiGLU Activation** - Less scrutinized than matrix multiplications
   - Need to verify silu(x) = x * sigmoid(x) matches llama.cpp
   - Dump gate, up, and swiglu intermediate values

4. **Weight Tensor Byte Order** - llama.cpp works with same file, we don't
   - Possible endianness problem
   - Possible alignment issues
   - Need to dump raw bytes and compare with llama.cpp

5. **Embedding Scaling** - Some models scale by sqrt(hidden_dim)
   - Our code does direct lookup with NO scaling
   - Need to check if llama.cpp applies scaling factor

6. **Model Configuration Mismatch** - Maybe using wrong architecture parameters
   - Need to verify num_heads, num_kv_heads, head_dim, hidden_dim, ffn_dim
   - Compare with llama.cpp's detected config

7. **cuBLAS Internal Bug or Misuse** (NEW - HIGH PRIORITY)
   - All parameters verified correct (lda, ldb, ldc, transpose flags)
   - Manual calculation works, cuBLAS doesn't
   - Extremes at specific indices (95, 126) suggest memory alignment or indexing issue
   - May need to test alternative GEMM implementations or custom kernel

8. **Attention Output Projection** (NEW)
   - Q-projection is broken, but what about attention output projection?
   - Does it have similar localized extremes?
   - Could RoPE be amplifying the Q spikes?

### Teams That Kept Repeating Same Mistakes

- **None identified** - Teams generally learned from previous attempts and documented false leads well
- Team Aurora specifically tested Team Felicia's hypothesis with corrections, confirming it was wrong
- Team THIMBLE specifically tested ORION's stride hypothesis with explicit transpose
- Team TOP HAT systematically eliminated all remaining standard hypotheses
- FALSE_LEADS_SUMMARY.md effectively prevented redundant investigation

### Critical Untested Fix

- **Team Charlie Beta's ffn_down loading fix** - This could be THE bug, but wasn't tested due to compilation errors
- HIGH PRIORITY: Test this fix immediately
- If ffn_down was never loaded, FFN would use uninitialized memory → explains garbage output

### Recommended Next Steps

1. **URGENT:** Test Team Charlie Beta's ffn_down fix
   - This is the most promising lead
   - Missing weight loading would cause exactly these symptoms

2. **HIGH PRIORITY:** Deep cuBLAS investigation for Q-projection bug
   - Test with `cublasSgemm` (FP32 weights/inputs) instead of `cublasGemmEx`
   - Try different cuBLAS algorithms (CUBLAS_GEMM_ALGO_*)
   - Write custom FP16 GEMM kernel for columns 95/126 to verify
   - Dump raw memory at weight buffer + offsets 95 and 126 in hex
   - Check for NaN/inf bits or alignment issues

3. **WORKAROUND:** Zero out Q[95] and Q[126] after GEMM
   - Measure impact on haiku generation quality
   - If output improves → confirms Q spikes are breaking downstream
   - Allows testing to continue while investigating root cause

4. **If Q-projection fix doesn't work:** Systematic llama.cpp comparison
   - Add logging at each stage (embedding, layer 0, layer 5, layer 10, etc.)
   - Run llama.cpp with same prompt and logging
   - Find FIRST point where values diverge
   - That's where the bug is

5. **Focus areas for comparison:**
   - RoPE actual computation (not just formula)
   - RMSNorm epsilon and intermediate values
   - SwiGLU intermediate values
   - Embedding scaling factor (if any)
   - Attention output projection (check for similar extremes)

6. **Stop investigating:**
   - ❌ cuBLAS transpose parameters (definitively proven correct by THIMBLE)
   - ❌ Tokenization (fixed by Team Blue, verified by Team Purple)
   - ❌ KV cache infrastructure (verified by Team Water)
   - ❌ Causal masking (verified by Team Bygone)
   - ❌ Prefill logic (verified by Team Bygone)
   - ❌ Sampling pipeline (fixed by Team HELIOS, verified working)
   - ❌ Tensor-core fast-math (eliminated by Team TOP HAT)
   - ❌ Weight corruption at columns 95/126 (eliminated by Team TOP HAT)
   - ❌ Input spikes (eliminated by Team TOP HAT)

---

## 📊 5. Investigation Statistics (UPDATED)

- **Total Teams:** 15+ (Blue, Purple, Green, Water, Charlie, Charlie Beta, Root Cause, Peer Review, Bygone, Felicia, Aurora, ORION, THIMBLE, TOP HAT, HELIOS, BATTLESHIP)
- **Total Experiments:** 26 major experiments
- **Verified Correct:** 17+ components (tokenization, embeddings, cuBLAS Q[0], cache, RoPE, sampling architecture, attention filtering, buffer management, etc.)
- **False Leads Documented:** 14+ in code + 14 in FALSE_LEADS_SUMMARY.md (Q spikes added)
- **Critical Bugs Found:** 4 (special token splitting - FIXED, ffn_down loading - UNTESTED, sampling order - FIXED, double-free crash - FIXED)
- **Partial Fixes:** 1 (output_norm normalization - helped but didn't solve)
- **Time Span:** ~33 hours (2025-10-06 20:56 UTC - 2025-10-07 00:56 UTC)
- **Compilation Status:** ✅ Fixed (double-free removed)

---

## 🎯 6. Current Status (UPDATED)

### What We Know For Certain

✅ **Working Correctly:**
- Tokenization (special tokens as single IDs)
- Token embeddings (valid values, correct lookup)
- cuBLAS matrix multiplication for Q[0] (verified mathematically)
- KV cache infrastructure (positions, reads, writes)
- Position tracking (increments correctly)
- RoPE formula (conceptually correct)
- Causal masking (implemented in kernel)
- Prefill logic (one-at-a-time is correct)
- Attention softmax (weights sum to 1.0)
- Argmax sampling (finds true maximum)
- Residual connections (simple addition)
- Sampling pipeline order (fixed by HELIOS - softmax before top-p)
- Temperature application (verified 0.7 applied correctly)
- Seed incrementing (verified correct)
- **Attention filtering (BATTLESHIP - Q spikes washed out by softmax)**
- **Buffer management (BATTLESHIP - no aliasing detected)**
- **Attention output projection (BATTLESHIP - clean, no spikes introduced)**

❌ **Still Broken:**
- Model generates garbage output (mojibake, code tokens)
- Q projection has extreme values (±16) at indices 95 and 126
- Repetitive token generation (same token 10+ times in some tests)
- First generated token already wrong (code token instead of haiku word)

### Most Promising Leads

🔥 **#1: Team Charlie Beta's ffn_down Fix** - Missing weight loading for FFN down projection
- Status: Code changed but NOT TESTED (compilation errors)
- Impact: If correct, this would fix the entire bug
- Priority: **URGENT** - test immediately
- Evidence: BATTLESHIP proved Q spikes are filtered by attention, so bug must be elsewhere

~~🔥 **#2: Q-Projection cuBLAS Bug** - Extreme values at specific indices (95, 126)~~
- Status: **ELIMINATED by BATTLESHIP** ✅
- Evidence: Q spikes exist (±16) but are completely filtered by attention softmax
- After attention: values return to normal (±0.03)
- Impact: **HARMLESS** - Q spikes don't propagate downstream
- Priority: **CLOSED** - documented as red herring, no further action needed

### If These Don't Fix It

🔍 **Systematic llama.cpp Comparison** - Find divergence point
- All infrastructure verified, so bug must be in subtle implementation detail
- Need to compare intermediate values at each stage
- Focus on RoPE computation, RMSNorm epsilon, SwiGLU activation, attention output projection

### ~~Compilation Blocker~~

~~⚠️ **Build Error:** `layer_call_count` undeclared at line 318~~
- Status: **FIXED by BATTLESHIP** ✅
- Issue: Double-free of `h_q_full` causing "double free or corruption" crash
- Root cause: Duplicate `delete[]` at lines 942 and 993
- Fix: Removed duplicate delete at line 942
- Tests now run to completion without crashes

---

## 📚 7. Key Documents (UPDATED)

- `investigation-teams/FALSE_LEADS_SUMMARY.md` - Comprehensive false leads list
- `investigation-teams/TEAM_*_HANDOFF.md` - Individual team handoff documents
- `investigation-teams/PEER_REVIEW_FINAL_REPORT.md` - Independent verification
- `investigation-teams/TEAM_ORION_*.md` - Q-projection divergence findings (NEW)
- `investigation-teams/TEAM_THIMBLE_SUMMARY.md` - Stride hypothesis disproven
- `investigation-teams/TEAM_TOP_HAT_HANDOFF.md` - All standard hypotheses eliminated
- `investigation-teams/TEAM_HELIOS_FINDINGS.md` - Sampling architecture fix
- `investigation-teams/TEAM_HELIOS_HANDOFF.md` - Sampling verified, bug upstream
- `investigation-teams/TEAM_BATTLESHIP_HANDOFF.md` - Downstream wiring investigation (NEW)
- `investigation-teams/TEAM_BATTLESHIP_FINDINGS.md` - Q spikes proven harmless (NEW)
- `investigation-teams/TEAM_BATTLESHIP_QUICKSTART.md` - Quick start guide (NEW)
- `cuda/src/transformer/qwen_transformer.cpp` - Main transformer with extensive comments
- `cuda/kernels/gqa_attention.cu` - Attention kernel with verification comments
- `cuda/kernels/sampling_wrapper.cu` - Sampling with HELIOS fix
- `src/inference/cuda_backend.rs` - Token flow and prefill logic

---

### Team PRINTER (2025-10-07 ~01:24 UTC to ~01:33 UTC)

**Mission:** Parity Data Sweep - Collect side-by-side checkpoint data from our engine and llama.cpp (utility team, no bug fixing)

**Hypotheses Tested:**
1. Infrastructure needed for systematic parity comparison
2. Existing logging sufficient for initial comparison
3. Full checkpoint logging needed for precise divergence identification

**Changes Made:**
- Created `cuda/src/utils/checkpoint_logger.h` - Checkpoint logging header
- Created `cuda/src/utils/checkpoint_logger.cpp` - Implementation with NPZ export
- Modified `cuda/CMakeLists.txt` line 58 - Added checkpoint_logger.cpp to build
- Modified `cuda/src/transformer/qwen_transformer.cpp` lines 5, 302, 311 - Integrated init/finalize
- Created `investigation-teams/TEAM_PRINTER_PARITY/` directory with 11 files:
  - `README.md` - Comprehensive guide with practical strategy
  - `GO_NO_GO_CHECKLIST.md` - Pre-flight verification checklist
  - `HANDOFF.md` - Complete handoff document
  - `EXECUTION_SUMMARY.md` - Infrastructure summary
  - `printer_meta.json` - Environment and test metadata
  - `run_our_engine.sh` - Runner script for our engine
  - `run_llamacpp.sh` - Runner script for llama.cpp
  - `convert_to_npz.py` - Binary to numpy .npz converter
  - `collect_parity_data.py` - Automated diff report generator
  - `vocab_and_tokenizer_snapshot/` - Directory for vocab comparison

**Observations:**
- Existing logging from SENTINEL, ORION, RACE CAR covers many checkpoints ✅
- Checkpoint logger integrated cleanly into build system ✅
- Non-invasive design preserves all previous investigation code ✅
- FP16 → FP32 conversion ensures precision in comparisons ✅
- Token-based filtering (default: tokens 0 & 1 only) reduces data volume ✅
- Binary + manifest format allows C++ → Python workflow ✅

**Conclusions:**
- ✅ **INFRASTRUCTURE COMPLETE:** Full checkpoint logging system ready to use
- ✅ **TWO-PHASE APPROACH:** Can use existing logs first, add full logging if needed
- ✅ **KEY QUESTION:** Does llama.cpp also see Q[95]/Q[126] spikes?
  - If YES → Bug is in model file or expected behavior
  - If NO → Bug is in our cuBLAS usage or weight loading
- 🎯 **READY TO EXECUTE:** Build, run, compare, document
- 📊 **PRAGMATIC STRATEGY:** Start with existing logs (10 min), escalate to full logging if needed (2 hours)

**Files Modified:**
- `cuda/CMakeLists.txt` (line 58 - added checkpoint_logger.cpp)
- `cuda/src/transformer/qwen_transformer.cpp` (lines 5, 302, 311 - integrated logger)

**Files Created:**
- `cuda/src/utils/checkpoint_logger.h` (header)
- `cuda/src/utils/checkpoint_logger.cpp` (implementation)
- `investigation-teams/TEAM_PRINTER_PARITY/*` (11 files total)

**Status:** Infrastructure complete, ready for execution

**Next Steps:**
1. Build: `cargo clean && cargo build --release --features cuda`
2. Run our engine: `./investigation-teams/TEAM_PRINTER_PARITY/run_our_engine.sh`
3. Run llama.cpp: `./investigation-teams/TEAM_PRINTER_PARITY/run_llamacpp.sh`
4. Compare logs and identify first divergence
5. Document findings in `diff_report.md`
6. Hand off to appropriate team based on divergence location

---

### Team LAMINATOR (2025-10-07 ~08:48 UTC to ~08:52 UTC)

**Mission:** Output RMSNorm Investigation - Prove or falsify: "The output RMSNorm (final normalization before LM head) is numerically wrong (epsilon/formula/scale, dtype, stride), producing out-of-range hidden states that cause garbage output."

**Hypotheses Tested:**
1. RMSNorm epsilon value is wrong
2. RMSNorm formula implementation has bugs
3. Gamma weights have wrong shape/stride/dtype
4. Post-norm values should contract to O(1) range but don't
5. Accumulation dtype is wrong (FP16 instead of FP32)

**Changes Made:**
- Added diagnostic markers in `qwen_transformer.cpp` lines 2541-2672
- Instrumented pre-RMSNorm input stats (min/max/mean/first8)
- Instrumented post-RMSNorm output stats (min/max/mean/first8)
- Logged gamma weight stats (len/mean/min/max)
- Verified epsilon, hidden_dim, dtypes
- Manual formula verification: computed expected y[0] vs actual y[0]

**Observations:**
- **Pre-RMS input:** min=-11.85, max=25.02, mean=0.082, range ~37
- **Post-RMS output:** min=-34.91, max=23.80, mean=0.126, range ~59
- **Gamma weights:** len=896, mean=7.139, min=-0.011, max=16.750
- **Config:** eps=1e-6, hidden_dim=896, dtype_in=FP16, dtype_accum=FP32
- **Formula check:** manual_y[0]=0.965462, actual_y[0]=0.965332, diff=0.000130 ✅
- **Key finding:** Post-norm values EXPAND instead of contract (59 > 37) due to gamma_mean=7.14

**Conclusions:**
- ✅ **FALSIFIED:** The hypothesis "output RMSNorm numerics wrong" is FALSIFIED
- ✅ **VERIFIED:** Epsilon 1e-6 matches llama.cpp (llamacpp.run.log line 68)
- ✅ **VERIFIED:** Formula implementation correct (diff=0.00013, within FP16 precision)
- ✅ **VERIFIED:** Gamma weights correct (mean=7.14 matches Team Charlie's findings)
- ✅ **VERIFIED:** Shape/stride correct (len=896 matches hidden_dim)
- ✅ **VERIFIED:** Dtype correct (FP16 input, FP32 accumulation)
- 🔍 **KEY INSIGHT:** Post-norm "amplification" is INTENTIONAL per model design
  - llama.cpp uses identical gamma weights (mean=7.14) and generates perfect haiku
  - The weights are stored correctly in GGUF file (Team Charlie verified)
- ❌ **REMAINING:** Bug is NOT in output RMSNorm - must be elsewhere

**Handoff:** The RMSNorm implementation is correct and matches llama.cpp exactly. Recommend investigating:
1. Layer 23 FFN output (what feeds into this RMSNorm) - why range [-11.85, 25.02]?
2. LM head projection deep dive - compare post-RMSNorm → logits with llama.cpp
3. Systematic parity comparison (Team Printer infrastructure) to find first divergence

**Files Modified:**
- `cuda/src/transformer/qwen_transformer.cpp` (lines 2541-2672 - investigation markers)

**Files Created:**
- `investigation-teams/TEAM_LAMINATOR_HANDOFF.md` (complete handoff document)
- `investigation-teams/FALSE_LEADS_SUMMARY.md` (added FALSE LEAD #9)

**Status:** Investigation complete, hypothesis falsified, handoff document created

---

### Team HOLE_PUNCH (2025-10-07 ~09:10 UTC)

**Mission:** RoPE Numeric Parity Investigation - Prove or falsify: "Our RoPE application produces numerically wrong Q/K values (angles/base/scale/indexing/stride/dtype), which later corrupt attention and the final output."

**Hypotheses Tested:**
1. RoPE config values (head_dim, rope_freq_base) don't match model spec
2. Angle generation uses wrong formula or truncated arithmetic
3. Q/K indexing/stride/layout causes head mixing or off-by-one errors
4. Numeric transformation at pos=0 doesn't produce identity (Q_PRE != Q_POST)
5. Rotation at pos=1 uses wrong cos/sin values

**Changes Made:**
- Added diagnostic markers in `qwen_transformer.cpp` lines 1177-1319
- Logged config parameters: head_dim, num_heads, num_kv_heads, rope_freq_base, pos
- Dumped Q/K first8 pre-RoPE and post-RoPE for layer 0 & 1, tokens 0-1
- Added angle logging in `rope.cu` lines 213-221: theta, cos, sin, dim, inv_freq
- Checked head 0 and last head (num_heads-1, num_kv_heads-1) for consistency

**Observations:**
- **Config:** head_dim=64, num_heads=14, num_kv_heads=2, rope_freq_base=1000000.0, pos=0 ✅
- **Token 0 (pos=0):** theta=0.0, cos=1.0, sin=0.0 for all dim_pairs ✅
  - Q_PRE == Q_POST (diff=0.0 for all 8 values) - perfect identity ✅
  - K_PRE == K_POST (diff=0.0 for all 8 values) - perfect identity ✅
- **Token 1 (pos=1):** 
  - dim_pair=0: theta=1.000000, cos=0.540302, sin=0.841471 ✅
    - Manual verify: cos(1.0)=0.5403, sin(1.0)=0.8415 ✅ MATCH
  - dim_pair=1: theta=0.649382, cos=0.796458, sin=0.604694 ✅
  - dim_pair=2: theta=0.421697, cos=0.912396, sin=0.409309 ✅
  - dim_pair=3: theta=0.273842, cos=0.962739, sin=0.270432 ✅
- **Formula verification:**
  - inv_freq = 1 / (rope_freq_base ^ (dim/head_dim))
  - For dim=0: 1 / (1000000^(0/64)) = 1.0 ✅
  - For dim=2: 1 / (1000000^(2/64)) = 0.6494 ✅
  - theta = pos * inv_freq (matches observed) ✅
- **Last head check:** All heads use consistent angles, correct strides ✅
- **Layer consistency:** All 24 layers use identical angle calculations ✅

**Conclusions:**
- ✅ **FALSIFIED:** The hypothesis "RoPE numeric mismatch" is FALSIFIED
- ✅ **VERIFIED:** Config matches model spec exactly (head_dim=64, freq_base=1000000.0)
- ✅ **VERIFIED:** Angle generation correct (all cos/sin match closed-form math)
- ✅ **VERIFIED:** Identity transformation at pos=0 works perfectly
- ✅ **VERIFIED:** Non-zero rotations at pos=1 use correct trigonometric values
- ✅ **VERIFIED:** Formula matches llama.cpp and RoPE paper
- ✅ **VERIFIED:** Indexing/layout correct (contiguous strides, correct head offsets)
- ❌ **REMAINING:** Bug is NOT in RoPE - must be elsewhere

**Handoff:** The RoPE implementation is mathematically and numerically correct. All config, angles, and transformations match expected values. Recommend investigating:
1. Attention mechanism (Q·K scoring, softmax, V aggregation, GQA grouping)
2. KV cache usage or indexing correctness
3. Attention output projection
4. LM head projection numeric parity

**Files Modified:**
- `cuda/src/transformer/qwen_transformer.cpp` (lines 1177-1319 - investigation markers)
- `cuda/kernels/rope.cu` (lines 213-221 - angle logging)

**Files Created:**
- `investigation-teams/TEAM_HOLE_PUNCH_SUMMARY.md` (complete summary document)
- `investigation-teams/FALSE_LEADS_SUMMARY.md` (added FALSE LEAD #10)

**Status:** Investigation complete, hypothesis falsified, summary document created

---

**Chronicle Complete**  
**Last Updated:** 2025-10-07T09:10Z  
**Status:** Active investigation - TEAM HOLE_PUNCH complete, RoPE numeric parity falsified as root cause
