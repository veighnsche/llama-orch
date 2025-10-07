# 📧 Team Handoff: SEA → Next Team

**To:** Next Investigation Team  
**From:** Team SEA 🌊  
**Date:** 2025-10-06 20:30 UTC  
**Subject:** 🔍 Garbage Output Bug - Rust Code Verified, Bug is in C++ Forward Pass  
**Priority:** HIGH

---

## TL;DR

**What I Verified:**
- ✅ Prefill/generation logic is CORRECT (standard autoregressive)
- ✅ Sampling (argmax/temperature) is CORRECT
- ✅ Token flow Rust→C++→Rust is CORRECT
- ✅ Tokenizer encode/decode is CORRECT

**What's Wrong:**
- ❌ Model generates garbage tokens immediately after correct prompt
- ❌ Mojibake (Thai/Chinese Unicode) in English haiku
- ❌ Repetitive tokens (same token 3-10 times)
- ❌ High token IDs (near vocab limit) selected

**Root Cause:**
The bug is in the **C++ transformer forward pass**. llama.cpp works perfectly with the same model file, so we're doing something differently.

---

## 🎯 Current Status

### Test Output (2025-10-06 20:25 UTC)

**Prompt:**
```
<|im_start|>user
Write a haiku about GPU computing that includes the word "twenty-five" (nonce: l66ax3Av)<|im_end|>
<|im_start|>assistant
```

**Expected:** A coherent haiku with "twenty-five"

**Actual:**
```
_loadĠstretchedà¹Ģà¸Ńà¸ĩçĥŃçĤ¹à¹Ĥà¸Īêª®çĥŃçĤ¹ĠfrÃ¦kkeçĥŃçĤ¹Ġinadvertently...
```

### Token Analysis

First 10 generated tokens:
```
[0] ID= 12411 → "_load"
[1] ID= 40298 → "Ġstretched"      ← Repeated 10+ times!
[2] ID=124862 → "à¹Ģà¸Ńà¸ĩ"          ← Thai Unicode
[3] ID=104763 → "çĥŃçĤ¹"          ← Chinese Unicode (repeated 4x)
[4] ID=136426 → "à¹Ĥà¸Ī"            ← Thai Unicode
[5] ID=150843 → "êª®"              ← Korean Unicode
[6] ID=104763 → "çĥŃçĤ¹"          ← REPETITIVE!
[7] ID= 98403 → "ĠfrÃ¦kke"        ← Norwegian
[8] ID=104763 → "çĥŃçĤ¹"          ← REPETITIVE!
[9] ID= 69085 → "Ġinadvertently"
```

**Symptoms:**
1. Mojibake (wrong language tokens)
2. Repetitive tokens (same ID multiple times)
3. High token IDs (124862, 136426, 150843 near vocab limit 151643)

---

## ✅ What I Ruled Out

### 1. Prefill/Generation Logic ✅

**User Question:** "Should we process ALL prompt tokens in prefill?"

**Answer:** NO! Current behavior is CORRECT.

```rust
// Prefill: tokens 0..N-2 → build KV cache
for i in 0..token_ids.len()-1 {
    generate_token(token_ids[i], ...); // Ignore output
}

// Generation: token N-1 → predict token N
current_token = token_ids[N-1];
next_token = generate_token(current_token, ...); // Use output
```

This is standard autoregressive generation. If we processed ALL tokens in prefill, we'd have nothing to generate from.

**File:** `src/inference/cuda_backend.rs:342-385`  
**Status:** ✅ VERIFIED CORRECT

### 2. Sampling Implementation ✅

**Argmax:** Correctly scans `vocab_size` (151643) positions, not padded  
**Temperature:** Applied correctly  
**Softmax:** Normalization correct  
**Random sampling:** Logic correct

**File:** `cuda/kernels/sampling_wrapper.cu:149-296`  
**Status:** ✅ VERIFIED CORRECT (Team Alpha + Team Love)

### 3. Token Flow ✅

**Rust → C++:**
- Token ID passed correctly via FFI
- `forward()` called before sampling
- Logits buffer updated each token

**C++ → Rust:**
- Sampled token returned correctly
- Decoding uses tokenizer correctly

**Files:** 
- `cuda/src/ffi_inference.cpp:152-241`
- `src/inference/cuda_backend.rs:403-473`

**Status:** ✅ VERIFIED CORRECT

### 4. cuBLAS & Vocab Sizes ✅

Team HOTEL fixed the vocab_size/padded_vocab_size bug. cuBLAS now computes all 151936 logits correctly.

**File:** `cuda/src/transformer/qwen_transformer.cpp:639-652`  
**Status:** ✅ FIXED by Team HOTEL

---

## ❌ Where the Bug Is

### The Logits Are Corrupted

The sampling code is working correctly, but it's sampling from **corrupted logits**.

**Evidence:**
```
First 10 logits: 0.83 0.79 -0.95 2.55 6.87 0.86 -1.98 -1.76 2.26 3.05
```

These values look reasonable, BUT:
- Token 104763 (Chinese mojibake) has the highest logit → gets selected
- Token 40298 ("stretched") has high logit repeatedly → repetitive output
- Tokens near vocab limit (100k-150k) have abnormally high logits

### Why llama.cpp Works But We Don't

**Critical Fact:** llama.cpp generates perfect haikus with the SAME model file.

This means:
- ✅ Model file is correct
- ✅ Weights are correct
- ❌ **Our forward pass does something different**

---

## 🔍 Investigation Leads for Next Team

### Priority 1: Compare Forward Pass with llama.cpp

**Action:** Trace llama.cpp's forward pass step-by-step and find differences

**Key areas to check:**
1. **Embedding lookup** - Do we scale embeddings?
2. **Attention mask** - Are we applying it correctly?
3. **Position encoding** - RoPE parameters correct?
4. **Layer normalization** - Any scaling factors we're missing?
5. **Residual connections** - Are we adding correctly?
6. **Final projection** - Any post-processing llama.cpp does?

**Files to compare:**
- Our code: `cuda/src/transformer/qwen_transformer.cpp:210-1145`
- Reference: `reference/llama.cpp/src/llama.cpp`

### Priority 2: Check Hidden State Evolution

The test shows hidden state grows through layers:
- Layer 0: Std 0.38
- Layer 23: Std 3.94
- After final norm: Std 7.26

**Question:** Is this growth normal or is it accumulating error?

**Action:** Compare hidden state statistics with llama.cpp at same positions

### Priority 3: Verify Output Norm

Test warns: "⚠️ WARNING: output_norm weights are abnormal!"
- Range: [-0.0114, 16.7500]
- Mean: 7.1393

**BUT:** Team Charlie verified these weights are CORRECT (llama.cpp uses same values).

**Action:** Verify we're USING these weights correctly in RMSNorm

**File:** `cuda/src/transformer/qwen_transformer.cpp:1026-1034`

### Priority 4: Add Comparative Logging

**Action:** Add logging to dump intermediate values and compare with llama.cpp:

```cpp
// After embedding
fprintf(stderr, "Embedding[0]: %.4f\n", embedding[0]);

// After each layer
fprintf(stderr, "Layer %d output[0]: %.4f\n", layer_idx, hidden[0]);

// After final norm
fprintf(stderr, "Final norm[0]: %.4f\n", normed[0]);

// After projection
fprintf(stderr, "Logit[0]: %.4f, Logit[100]: %.4f\n", logits[0], logits[100]);
```

Then run llama.cpp with same prompt and compare values.

---

## 🚨 Common Traps (Don't Waste Time)

1. **Don't re-investigate prefill/generation** - It's verified correct
2. **Don't re-investigate sampling** - Teams Alpha and Love verified it
3. **Don't re-investigate vocab sizes** - Team HOTEL fixed it
4. **Don't assume weights are wrong** - llama.cpp uses same weights successfully
5. **Don't modify output_norm weights** - They're correct (see Team Charlie's findings)

---

## 📝 Files I Modified

### 1. `src/inference/cuda_backend.rs` (lines 347-358, 380-385)
Added Team SEA investigation comments:
- Documented prefill/generation logic
- Marked as FALSE_LEAD (it's correct)
- Explained why this is NOT the bug

### 2. `investigation-teams/TEAM_SEA_FINDINGS.md`
Complete investigation report with:
- Test output analysis
- Token analysis
- What I verified
- What I ruled out
- Investigation leads for next team

---

## 🎯 Success Criteria

Your mission is complete when:

1. **Model generates coherent text** (not mojibake)
2. **No repetitive tokens** (varied output)
3. **Haiku test passes** with minute word in output
4. **Output matches llama.cpp quality**

**Expected Output:**
```
Haiku:
Circuits hum and glow,
Twenty-five cores compute fast,
Silicon dreams flow.

✅ QUALITY CHECK PASSED: Minute word 'twenty-five' found exactly once
```

---

## 🔑 Key Insight

**The bug is NOT in the Rust code.** All Rust logic is correct:
- Tokenization ✅
- Prefill/generation split ✅
- Token flow ✅
- Decoding ✅

**The bug IS in the C++ forward pass.** Something we do differently from llama.cpp causes corrupted logits.

**Your mission:** Find what llama.cpp does that we don't (or vice versa) in the transformer forward pass.

---

## 📞 Quick Reference

**Test Command:**
```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

**Current Result:** ❌ FAIL - Generates mojibake and repetitive tokens

**llama.cpp Test (for comparison):**
```bash
cd reference/llama.cpp
./llama-cli -m /path/to/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku about GPU computing that includes the word \"twenty-five\"" \
  -n 50 --temp 0.7
```

**llama.cpp Result:** ✅ PASS - Generates perfect haiku

---

## 📚 Related Documents

- `TEAM_SEA_FINDINGS.md` - My complete investigation report
- `TEAM_HOTEL_FINDINGS.md` - cuBLAS dimension fix
- `TEAM_CHARLIE_I_WAS_WRONG.md` - output_norm weights are correct
- `VICTORY_BUG_FIXED.md` - ffn_down weight loading fix (historical)

---

**Good luck! The bug is close - it's just a difference between our forward pass and llama.cpp's. Find it! 🚀**

---

*Signed,*  
**Team SEA 🌊**  
*"We surf the code waves to find the bugs"*
