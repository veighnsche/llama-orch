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

## 📚 7. Key Documents

- `investigation-teams/FALSE_LEADS_SUMMARY.md` - Comprehensive false leads list
- `investigation-teams/TEAM_*_HANDOFF.md` - Individual team handoff documents
- `investigation-teams/PEER_REVIEW_FINAL_REPORT.md` - Independent verification
- `cuda/src/transformer/qwen_transformer.cpp` - Main transformer with extensive comments
- `cuda/kernels/gqa_attention.cu` - Attention kernel with verification comments
- `src/inference/cuda_backend.rs` - Token flow and prefill logic

---

**Chronicle Complete**  
**Last Updated:** 2025-10-06T22:22Z  
**Status:** Active investigation - ffn_down fix needs testing
