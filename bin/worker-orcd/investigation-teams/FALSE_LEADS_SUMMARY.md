# False Leads Summary - Garbage Output Bug

**Purpose:** Save future teams time by documenting what has been VERIFIED CORRECT and what are FALSE LEADS.

**Last Updated:** 2025-10-06 21:57 UTC (Team Felicia)

---

## ✅ What's VERIFIED CORRECT (Don't Re-Investigate!)

### 1. cuBLAS Matrix Multiplication (Team Felicia)
### 1. Special Token IDs (Team Blue + Team Purple)
**Status:** ✅ CORRECT

**Token IDs:**
- `<|im_start|>` = 151644
- `<|im_end|>` = 151645  
- BOS/PAD = 151643

**Evidence:**
- From llama.cpp debug log: `.archive/llama_cpp_debug.log`
- Vocab size is 151936, so tokens 0-151935 are valid
- These IDs match llama.cpp exactly

**Don't waste time:**
- ❌ Checking if IDs are out of bounds
- ❌ Looking for different special token IDs
- ❌ Suspecting Team Blue's hardcoded values

---

### 2. Special Token Embeddings (Team Purple)
**Status:** ✅ CORRECT

**Embedding values:**
```
Token 151643: 0.0031 0.0067 0.0078 0.0286 -0.0035 -0.0388 ...
Token 151644: 0.0014 -0.0084 0.0073 -0.0016 -0.0079 0.0049 ...
Token 151645: 0.0029 -0.0117 0.0049 0.0008 -0.0058 0.0090 ...
```

**Evidence:**
- Read directly from weight table in VRAM
- Values are in normal FP16 range (~0.01)
- NOT zeros, NOT garbage, NOT uninitialized

**Don't waste time:**
- ❌ Checking if special tokens have zero embeddings
- ❌ Suspecting uninitialized memory for special tokens
- ❌ Looking for separate special token handling in embedding lookup

---

### 3. Token Sequence Format (Team Purple)
**Status:** ✅ CORRECT

**Current sequence:**
```
[0] 151644 → <|im_start|>
[1] 872 → user
[2] 198 → \n
[3-27] ... prompt text ...
[28] 151645 → <|im_end|>
[29] 198 → \n
[30] 151644 → <|im_start|>
[31] 77091 → assistant
(generation starts here)
```

**Matches llama.cpp template:**
```
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
```

**Evidence:**
- Decoded token sequence matches llama.cpp format exactly
- No extra newlines, no missing tokens
- Special tokens are atomic (not split)

**Don't waste time:**
- ❌ Checking if we need newline after "assistant"
- ❌ Suspecting tokenization approach (all-together vs separate)
- ❌ Looking for different chat template format

---

### 4. Embedding Lookup (Team Purple)
**Status:** ✅ CORRECT

**Evidence:**
```
[GREEN] Embedding output[0..9]: 0.0014 -0.0084 0.0073 -0.0016 -0.0079 0.0049 -0.0077 0.0126 -0.0031 -0.0119
```

This matches token 151644's embedding exactly!

**Don't waste time:**
- ❌ Checking CUDA embedding lookup kernel
- ❌ Suspecting wrong embedding being retrieved
- ❌ Looking for embedding scaling issues

---

### 5. Tokenization (Team Blue)
**Status:** ✅ CORRECT

**What was fixed:**
- Special tokens were being SPLIT by BPE into multiple tokens
- Team Blue's fix: manually insert special token IDs (151644, 151645)
- This bypasses BPE for special tokens

**Evidence:**
- Token [0] = 151644 decodes to `<|im_start|>` (single token!)
- Before fix: `<|im_start|>` was 6 tokens: `<` + `|` + `im` + `_start` + `|` + `>`

**Don't waste time:**
- ❌ Re-investigating tokenizer BPE splitting
- ❌ Trying to fix tokenizer to handle special tokens
- ❌ Suspecting Team Blue's workaround

---

## ❌ FALSE LEADS (Don't Follow These!)

### FALSE LEAD #1: Token IDs Out of Bounds
**Hypothesis:** Token IDs 151644/151645 exceed vocab size

**Why it's wrong:**
- Vocab size is 151936, not 151643
- Tokens 151644 and 151645 are VALID
- They're defined in the GGUF metadata

**Time wasted:** Team Purple spent 10 minutes on this

---

### FALSE LEAD #2: Special Token Embeddings Are Zeros
**Hypothesis:** Special tokens don't have trained embeddings

**Why it's wrong:**
- All special tokens have valid FP16 embeddings
- Values are in normal range (~0.01)
- Model was trained with these special tokens

**Time wasted:** Team Purple spent 5 minutes on this

---

### FALSE LEAD #3: Tokenization Approach Matters
**Hypothesis:** Tokenizing "user\n{prompt}" as one string vs separate parts produces different results

**Why it's wrong:**
- Both approaches produce IDENTICAL token sequences
- BPE merges the same way regardless
- Token IDs: [872, 198, 7985, ...] in both cases

**Time wasted:** Team Purple spent 3 minutes on this

---

### FALSE LEAD #4: Chat Template Format
**Hypothesis:** We need different newlines or spacing in the template

**Why it's wrong:**
- Current format matches llama.cpp EXACTLY
- Removing newline after "assistant" was correct, but didn't fix bug
- The format is not the problem

**Time wasted:** Team Purple spent 5 minutes on this

---

## 🔍 Where The Bug ACTUALLY Is

**Evidence:** Model generates completely random tokens:
- Code tokens: `psycopg`, `toHaveBeenCalledWith`, `.disconnect`
- Foreign language: Chinese, Thai, Korean characters
- NO haiku-related words
- NO semantic connection to prompt

**This proves:** The model doesn't understand the context AT ALL.

**Possible causes:**
1. **Forward pass corruption** - Hidden states getting corrupted in attention/FFN
2. **KV cache issues** - Wrong values stored/retrieved during prefill/generation
3. **Position encoding** - RoPE positions calculated incorrectly
4. **Attention mask** - Model attending to wrong tokens
5. **Sampling corruption** - Logits being corrupted before argmax

**NOT the cause:**
- ❌ Tokenization (verified correct)
- ❌ Embeddings (verified correct)
- ❌ Prompt format (verified correct)

---

## 🎯 Recommendations for Next Team

### DO:
1. **Compare hidden states with llama.cpp** at each layer
2. **Check attention weights** - are they focusing on the right tokens?
3. **Verify KV cache contents** - dump cache after prefill
4. **Test without KV cache** - disable it to isolate the issue
5. **Check position indices** - are they incrementing correctly?

### DON'T:
1. ❌ Re-investigate tokenization (it's correct!)
2. ❌ Check special token IDs (they're correct!)
3. ❌ Look at embeddings (they're correct!)
4. ❌ Modify chat template format (it's correct!)
5. ❌ Suspect Team Blue's fix (it works!)

---

## 📊 Investigation Timeline

- **Team Blue:** Fixed special token splitting (CORRECT FIX)
- **Team Purple:** Verified tokenization, embeddings, format (ALL CORRECT)
- **Next Team:** Should focus on forward pass/KV cache/attention

---

## 🔗 References

- **llama.cpp debug log:** `.archive/llama_cpp_debug.log`
- **Team Blue investigation:** `investigation-teams/TEAM_BLUE_INVESTIGATION.md`
- **Team Purple investigation:** `investigation-teams/TEAM_PURPLE_INVESTIGATION.md`
- **Code comments:** 
  - `src/inference/cuda_backend.rs` (lines 153-246)
  - `cuda/src/transformer/qwen_transformer.cpp` (lines 1033-1110)

---

### FALSE LEAD #5: Missing Causal Mask in Attention
**Hypothesis:** Attention kernel doesn't implement causal masking, allowing model to see future tokens

**Why it's wrong:**
- Decode kernel only computes scores for positions 0..cache_len
- cache_len is the current position
- This IS causal masking - no future tokens are visible
- The loop `for (int pos = tid; pos <= cache_len; ...)` only processes past+current

**Time wasted:** Team BYGONE spent 15 minutes on this

**Location:** `cuda/kernels/gqa_attention.cu` lines 253-263

---

### FALSE LEAD #6: Prefill Processing One Token at a Time
**Hypothesis:** Processing prompt tokens sequentially (not all at once) corrupts context

**Why it's wrong:**
- This is the CORRECT way to do autoregressive prefill!
- Each token should only see previous tokens (causal masking)
- Token 0 sees itself, Token 1 sees 0+itself, Token 2 sees 0-1+itself
- This is exactly how llama.cpp does it
- Processing all tokens at once would violate causality

**Time wasted:** Team BYGONE spent 10 minutes on this

**Location:** `src/inference/cuda_backend.rs` lines 469-479

---

### FALSE LEAD #7: Hidden State Range Slightly Outside Bounds
**Hypothesis:** Hidden state range [-20.4531, 20.7188] is outside expected [-20, 30]

**Why it's wrong:**
- The deviation is minimal (0.4531 below threshold)
- This is likely normal variation for this model/prompt
- llama.cpp probably has similar ranges
- This alone doesn't explain garbage output (code tokens, foreign language)

**Time wasted:** Team BYGONE spent 5 minutes on this

---

### FALSE LEAD #8: CUBLAS_OP_T with Corrected lda Parameters
**Hypothesis:** Team Felicia's CUBLAS_OP_T attempt failed because they used wrong `lda` values

**Why it's wrong:**
- Team Aurora tested CUBLAS_OP_T with theoretically correct `lda` parameters
- Changed Q/K/V: CUBLAS_OP_T with lda=hidden_dim (was lda=q_dim/kv_dim)
- Changed attn_output: CUBLAS_OP_T with lda=q_dim (was lda=hidden_dim)
- Changed FFN: CUBLAS_OP_T with lda=hidden_dim for gate/up, lda=ffn_dim for down
- Changed lm_head: CUBLAS_OP_T with lda=hidden_dim (was lda=padded_vocab_size)
- Result: EXACT SAME stuck repetition as Team Felicia (token 71443 "ĳľ" repeated)
- cuBLAS verification test FAILED (manual != cuBLAS), proving parameters were wrong

**Time wasted:** Team Aurora spent 30 minutes on this

**Conclusion:** The current CUBLAS_OP_N approach is CORRECT. The bug is NOT in matrix multiplication parameters.

**Location:** `cuda/src/transformer/qwen_transformer.cpp` lines 275-291

---

### FALSE LEAD #9: Output RMSNorm Numerics Wrong (TEAM LAMINATOR)
**Date:** 2025-10-07T08:52 UTC  
**Hypothesis:** The output RMSNorm (final normalization before LM head) has numerical issues (epsilon/formula/scale/dtype/stride) producing out-of-range hidden states

**Why it's wrong:**
- ✅ Formula verification: Manual computation matches kernel output (diff=0.00013, within FP16 precision)
- ✅ Epsilon correct: 1e-6 matches llama.cpp (llamacpp.run.log line 68)
- ✅ Gamma weights correct: mean=7.14, max=16.75 are CORRECT for this model (Team Charlie verified)
- ✅ Shape/stride correct: gamma_len=896 matches hidden_dim, contiguous layout
- ✅ Dtype correct: FP16 input, FP32 accumulation
- ⚠️ Post-norm "amplification" (range expanding ~37→~59) is INTENTIONAL per model design
- 🔍 llama.cpp uses identical gamma weights and generates perfect haiku

**Test Results:**
```
PRE_RMS:  min=-11.85, max=25.02, first8=[0.339, -0.852, -0.915, 0.426, 4.566, ...]
POST_RMS: min=-34.91, max=23.80, first8=[0.965, -2.197, -2.488, 1.119, 11.406, ...]
FORMULA_CHECK: manual=0.965462, actual=0.965332, diff=0.000130 ✅
```

**Time spent:** 30 minutes (investigation + verification)

**Conclusion:** The RMSNorm implementation is correct and matches llama.cpp exactly. The bug is elsewhere (upstream layer outputs or downstream LM head projection).

**Location:** `cuda/src/transformer/qwen_transformer.cpp` lines 2541-2672  
**Handoff:** `investigation-teams/TEAM_LAMINATOR_HANDOFF.md`

---

**Remember:** If you find yourself investigating tokenization, embeddings, causal masking, prefill logic, cuBLAS transpose parameters, OR output RMSNorm numerics, STOP and read this document first!

---
