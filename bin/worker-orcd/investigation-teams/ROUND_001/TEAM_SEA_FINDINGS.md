# Team SEA Investigation Report
**Date:** 2025-10-06 20:26 UTC  
**Mission:** Hunt down the **garbage output** bug (mojibake/unreadable tokens)  
**Status:** 🔍 **INVESTIGATION IN PROGRESS**

---

## 🎯 Mission Objective

Fix the garbage output bug where the model generates unreadable tokens instead of coherent haikus.

**Handoff from Team HOTEL:**
- Team HOTEL fixed cuBLAS dimension bugs (vocab_size vs padded_vocab_size)
- cuBLAS verification now passes at all positions
- But model still generates garbage: mojibake, repetitive tokens, Unicode soup

---

## 🔍 Test Results Analysis

### Current Output (2025-10-06 20:25 UTC)
```
Prompt: "Write a haiku about GPU computing that includes the word "twenty-five" (nonce: l66ax3Av)"

Output: _loadĠstretchedà¹Ģà¸Ńà¸ĩçĥŃçĤ¹à¹Ĥà¸Īêª®çĥŃçĤ¹ĠfrÃ¦kkeçĥŃçĤ¹Ġinadvertently...
```

### Token Analysis (First 10 tokens)
```
[0] ID= 12411 → "_load"
[1] ID= 40298 → "Ġstretched"      ← Repeated 10+ times!
[2] ID=124862 → "à¹Ģà¸Ńà¸ĩ"          ← Thai Unicode (mojibake)
[3] ID=104763 → "çĥŃçĤ¹"          ← Chinese Unicode (repeated 4x in first 10)
[4] ID=136426 → "à¹Ĥà¸Ī"            ← Thai Unicode
[5] ID=150843 → "êª®"              ← Korean Unicode
[6] ID=104763 → "çĥŃçĤ¹"          ← Same as [3] - REPETITIVE!
[7] ID= 98403 → "ĠfrÃ¦kke"        ← Norwegian/Danish
[8] ID=104763 → "çĥŃçĤ¹"          ← Same as [3] again!
[9] ID= 69085 → "Ġinadvertently"
```

### 🚨 Critical Observations

1. **REPETITIVE TOKENS**: Token 104763 appears 3 times in first 10 tokens
2. **REPETITIVE TOKENS**: Token 40298 ("Ġstretched") appears 10+ times in full output
3. **HIGH TOKEN IDs**: 124862, 136426, 150843 are near vocab limit (151643)
4. **UNICODE MOJIBAKE**: Thai, Chinese, Korean characters in English haiku prompt
5. **WRONG LANGUAGE**: Model selecting tokens from non-English vocab regions

---

## 🕵️ Investigation Trail

### Step 1: Verified Code Paths ✅

**Sampling Implementation** (`cuda/kernels/sampling_wrapper.cu:149-226`)
- ✅ Argmax correctly scans `vocab_size` positions (not padded)
- ✅ Temperature scaling applied correctly
- ✅ Softmax normalization correct
- ✅ Random sampling logic correct

**FFI Layer** (`cuda/src/ffi_inference.cpp:152-241`)
- ✅ Token ID passed correctly to transformer
- ✅ `forward()` called before sampling
- ✅ Logits buffer updated each token
- ✅ Vocab size passed correctly to sampler

**Rust Backend** (`src/inference/cuda_backend.rs:370-485`)
- ✅ Temperature from config used (not hardcoded)
- ✅ Token flow correct: `current_token` → `generate_token()` → `next_token_id`
- ✅ Decoding uses tokenizer correctly

### Step 2: Ruled Out ❌

- ❌ NOT a sampling bug (argmax/temperature/softmax all correct)
- ❌ NOT a token flow bug (Rust correctly feeds tokens back)
- ❌ NOT a decoding bug (tokenizer.decode works)
- ❌ NOT a vocab_size/padding bug (Team HOTEL fixed this)

---

## 🔥 SUSPECT: Logits Are Corrupted BEFORE Sampling

### Hypothesis

The sampling code is working correctly, but it's sampling from **corrupted logits**. The logits coming out of the transformer are wrong, causing:

1. **High logits for wrong tokens** → Argmax picks garbage tokens
2. **Repetitive logits** → Same token selected multiple times
3. **Wrong vocab region** → High-ID tokens (near padding) have high logits

### Evidence

From test output:
```
First 10 logits: 0.83 0.79 -0.95 2.55 6.87 0.86 -1.98 -1.76 2.26 3.05
```

These logits look reasonable in range, BUT:
- Token 104763 (Chinese mojibake) has highest logit → gets selected
- Token 40298 ("stretched") has high logit repeatedly → repetitive output
- Tokens in 100k-150k range (near padding) have abnormally high logits

### Root Cause Candidates

1. **Final projection matrix corruption**
   - `output.weight` tensor might be loaded incorrectly
   - Matrix transpose issue (row-major vs column-major)
   - Stride mismatch in cuBLAS call

2. **Output norm corruption**
   - Test shows: "⚠️ WARNING: output_norm weights are abnormal!"
   - Norm weights range: [-0.0114, 16.7500] with mean 7.1393
   - After norm: Range [-28.1094, 39.4375] with Std 7.2609
   - This is OUTSIDE expected range!

3. **Hidden state explosion**
   - Hidden state grows through layers: Std 0.38 → 3.94
   - Final layer: Range [-14.7031, 12.5781]
   - This might be normal, but could indicate accumulating error

---

## 🎯 PLAN: Next Investigation Steps

### Priority 1: Check Output Norm Weights
**File:** `cuda/src/model/qwen_weight_loader.cpp`
**Action:** Verify `output_norm` weights are loaded correctly
**Why:** Test explicitly warns "output_norm weights are abnormal!"

### Priority 2: Verify Final Projection Matrix
**File:** `cuda/src/transformer/qwen_transformer.cpp:612-652`
**Action:** 
- Check `output.weight` dimensions and layout
- Verify cuBLAS parameters (lda, ldb, ldc)
- Compare first few logits with llama.cpp

### Priority 3: Check for Bias Addition
**File:** `cuda/src/transformer/qwen_transformer.cpp`
**Action:** Verify no bias is added to final projection (Qwen2.5 doesn't use bias)

### Priority 4: Trace Logits Computation
**Action:** Add logging to dump:
- Hidden state before final norm
- Hidden state after final norm  
- First 20 logits after projection
- Compare with llama.cpp at same position

### Priority 5: Compare with llama.cpp Forward Pass
**File:** `reference/llama.cpp/src/llama.cpp`
**Action:** 
- Trace llama.cpp's forward pass step-by-step
- Look for differences in:
  - How embeddings are processed
  - How attention mask is applied
  - How position encoding is done
  - Any normalization or scaling we're missing

---

## 🚨 CRITICAL OBSERVATION

**Prefill/Generation Logic is CORRECT** (2025-10-06T20:30Z)

The user asked if we should process ALL prompt tokens in prefill. Answer: **NO**.

Current behavior is correct:
- Prefill: Process tokens 0..N-2, build KV cache
- Generation: Use token N-1 as input, predict token N

This is standard autoregressive generation. The bug is NOT here.

**The Real Question:** Why does the model generate garbage when llama.cpp generates perfect haikus with the SAME model file?

This means we're doing something different in the forward pass that llama.cpp does correctly.

---

## 📝 Comments to Add (Append-Only)

### In `cuda/src/transformer/qwen_transformer.cpp` (after line 652)

```cpp
// [TEAM SEA] 2025-10-06T20:26Z
// SUSPECT: Final projection produces corrupted logits → garbage tokens
// OBSERVED: Model generates mojibake (Thai/Chinese Unicode) and repetitive tokens
//   Token 104763 ("çĥŃçĤ¹") appears 3x in first 10 tokens
//   Token 40298 ("Ġstretched") appears 10+ times in output
//   High token IDs (124862, 136426, 150843) near vocab limit selected
// PLAN: Verify output_norm weights and final projection cuBLAS call
// CONTRADICTION: Test shows "output_norm weights are abnormal!" with range [-0.01, 16.75]
//   Expected: RMSNorm weights should be close to 1.0, not 7.14 mean
```

### In `cuda/src/model/qwen_weight_loader.cpp` (near output_norm loading)

```cpp
// [TEAM SEA] 2025-10-06T20:26Z
// SUSPECT: output_norm weights might be loaded incorrectly
// OBSERVED: Test reports "output_norm weights are abnormal!"
//   Range: [-0.0114, 16.7500], Mean: 7.1393
//   Expected: RMSNorm weights typically close to 1.0
// PLAN: Verify tensor name, dimensions, and data type
// PLAN: Compare first 10 values with llama.cpp's loaded weights
```

---

## 🔑 Key Insights

### Why This Bug is Hard

1. **Symptoms are downstream** - Garbage tokens are the END result, not the cause
2. **Multiple possible causes** - Could be weights, projection, norm, or accumulation
3. **Looks like sampling bug** - Repetitive tokens suggest sampling, but sampling code is correct
4. **No crashes** - Code runs successfully, just produces wrong output

### What We Know For Sure

- ✅ cuBLAS matrix multiplication works (Team HOTEL verified)
- ✅ Sampling/argmax works correctly
- ✅ Token flow Rust→C++→Rust works
- ✅ Tokenizer encode/decode works
- ❌ Logits are corrupted somewhere in transformer
- ❌ Output norm weights are "abnormal"

---

## 📊 Test Command

```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

**Current Result:** ❌ FAIL - Generates garbage tokens, not haiku

---

## 🚀 Next Team Handoff

**For the next investigator:**

1. Start with `output_norm` weights - test explicitly warns about this
2. Check final projection cuBLAS call parameters
3. Add logging to compare logits with llama.cpp
4. Don't re-investigate sampling - it's verified correct
5. Focus on WHY logits are corrupted, not HOW they're sampled

**Files to investigate:**
- `cuda/src/model/qwen_weight_loader.cpp` (output_norm loading)
- `cuda/src/transformer/qwen_transformer.cpp:612-652` (final projection)
- Compare with `reference/llama.cpp/src/llama.cpp` (reference implementation)

---

**Team SEA 🌊**  
*"We surf the code waves to find the bugs"*
