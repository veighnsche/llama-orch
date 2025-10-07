# Team GREEN Investigation Report

**Date:** 2025-10-06 20:35 UTC  
**Mission:** Fix garbage output bug (mojibake/repetitive tokens)  
**Status:** 🔍 **ACTIVE INVESTIGATION**

---

## 🎯 Current Symptoms

**Test Output (2025-10-06 20:35 UTC):**
```
Prompt: "Write a haiku about GPU computing that includes the word "thirty-five" (nonce: wvJOlSHl)"

Output: è®«æŁ¥æī¾ĠindReactĠScoutsĠconciseè®«çĥŃçĤ¹èįĥçĥŃçĤ¹ĠÐ»ÐµÑĩĠstretched...
```

**First 10 Generated Tokens:**
```
[0] ID=119578 → "è®«"          (Chinese)
[1] ID=109547 → "æŁ¥æī¾"        (Chinese)
[2] ID=  1257 → "Ġind"
[3] ID= 14799 → "React"
[4] ID= 69307 → "ĠScouts"
[5] ID= 63594 → "Ġconcise"
[6] ID=119578 → "è®«"          (REPETITIVE!)
[7] ID=104763 → "çĥŃçĤ¹"        (Chinese - appears 10+ times)
[8] ID=120042 → "èįĥ"          (Chinese)
[9] ID=104763 → "çĥŃçĤ¹"        (REPETITIVE!)
```

**Critical Observations:**
1. **Mojibake:** Chinese/Thai/Korean tokens instead of English
2. **Repetitive:** Token 104763 appears multiple times, "stretched" appears 10+ times
3. **High Token IDs:** 119578, 109547, 120042 near vocab limit (151643)
4. **Wrong Context:** "React", "Scouts", "llvm" - code tokens, not haiku

---

## ✅ What's Been Verified (DO NOT RE-INVESTIGATE)

### Team HOTEL ✅
- cuBLAS dimensions fixed: `[hidden_dim=896, padded_vocab=151936]`
- All 151936 logits computed correctly
- Verification passes at positions 0, 8850, 44394, 137131

### Team SEA ✅
- Sampling (argmax/temperature/softmax) is CORRECT
- Token flow Rust→C++→Rust is CORRECT
- Prefill/generation logic is CORRECT
- Tokenizer encode/decode is CORRECT

### Team WATER ✅
- KV cache parameter passing is CORRECT
- Cache read/write positions are CORRECT
- Position tracking (pos increments) is CORRECT
- RoPE applies different rotations per position

### Team PROMPT ✅
- Chat template format is CORRECT for Qwen2.5
- Prompt rendering matches llama.cpp (user-only mode)

### Team CHARLIE ✅
- output_norm weights are CORRECT (mean=7.14 is intentional)
- llama.cpp uses same weights and works fine
- RMSNorm implementation is CORRECT

---

## 🔥 THE SMOKING GUN

**Team SEA's Critical Finding:**

> The sampling code is working correctly, but it's sampling from **corrupted logits**.

**Evidence from test output:**
```
First 10 logits: 0.83 0.79 -0.95 2.55 6.87 0.86 -1.98 -1.76 2.26 3.05
```

These logit values look reasonable in range, BUT:
- Token 104763 (Chinese mojibake) has highest logit → gets selected
- Token 119578 (Chinese) has high logit → selected first
- Tokens in 100k-150k range have abnormally high logits

**The Question:** Why does llama.cpp generate perfect haikus with the SAME model file, but we generate garbage?

---

## 🔍 Root Cause Hypothesis

**The logits coming out of the transformer forward pass are corrupted.**

Possible causes (in priority order):

### Priority 1: Embedding Scaling
**SUSPECT:** Token embeddings might need scaling after lookup

**Evidence:**
- llama.cpp may scale embeddings by `sqrt(hidden_dim)` or similar
- Our code does direct embedding lookup without scaling
- This would affect ALL subsequent computations

**Action:** Check llama.cpp's embedding lookup code

### Priority 2: Attention Mask Application
**SUSPECT:** Causal mask might be applied incorrectly

**Evidence:**
- Model generates tokens that don't follow prompt context
- Suggests attention isn't properly attending to previous tokens
- Mask might be inverted or offset by one

**Action:** Verify mask values and application in attention kernel

### Priority 3: Final Projection Matrix
**SUSPECT:** Matrix transpose or stride issue in final projection

**Evidence:**
- cuBLAS verified correct, but parameters might be wrong
- Row-major vs column-major confusion
- lda/ldb/ldc stride parameters

**Action:** Compare cuBLAS call with llama.cpp

### Priority 4: Hidden State Accumulation
**SUSPECT:** Residual connections accumulating error

**Evidence:**
- Hidden state grows through layers (Std 0.38 → 3.94 → 7.26)
- After final norm: Range [-28.1, 39.4] (outside expected [-20, 30])
- Might indicate accumulating numerical error

**Action:** Compare hidden state statistics with llama.cpp at same positions

---

## 📝 Investigation Plan

### Step 1: Add Comparative Logging
Add logging to dump intermediate values and compare with llama.cpp:

```cpp
// After embedding
fprintf(stderr, "[GREEN] Embedding[0..9]: ");
for (int i = 0; i < 10; i++) {
    fprintf(stderr, "%.4f ", __half2float(hidden[i]));
}
fprintf(stderr, "\n");

// After each layer
fprintf(stderr, "[GREEN] Layer %d output[0..9]: ", layer_idx);
for (int i = 0; i < 10; i++) {
    fprintf(stderr, "%.4f ", __half2float(hidden[i]));
}
fprintf(stderr, "\n");

// After final norm
fprintf(stderr, "[GREEN] Final norm[0..9]: ");
for (int i = 0; i < 10; i++) {
    fprintf(stderr, "%.4f ", __half2float(normed[i]));
}
fprintf(stderr, "\n");

// After projection (first 20 logits)
fprintf(stderr, "[GREEN] Logits[0..19]: ");
for (int i = 0; i < 20; i++) {
    fprintf(stderr, "%.4f ", logits[i]);
}
fprintf(stderr, "\n");
```

### Step 2: Run llama.cpp with Same Prompt
```bash
cd reference/llama.cpp
./llama-cli -m /path/to/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku about GPU computing that includes the word \"thirty-five\"" \
  -n 10 --temp 0.7 --log-disable
```

Compare first 10 token IDs with our output.

### Step 3: Check Embedding Scaling
Look for scaling factors in llama.cpp's embedding lookup.

### Step 4: Verify Attention Mask
Print mask values and verify they match expected causal pattern.

---

## 🚨 Key Files to Investigate

1. **`cuda/src/transformer/qwen_transformer.cpp`**
   - Lines 210-300: Embedding lookup
   - Lines 612-652: Final projection
   - Lines 1026-1034: Final RMSNorm

2. **`cuda/kernels/embedding.cu`**
   - Check if scaling is applied after lookup

3. **`cuda/kernels/gqa_attention.cu`**
   - Lines 100-200: Attention mask application
   - Verify mask values are correct

4. **`reference/llama.cpp/src/llama.cpp`**
   - Compare embedding, attention, and projection logic

---

## 🎯 Success Criteria

Mission complete when:
1. ✅ Model generates coherent English text (not mojibake)
2. ✅ No repetitive tokens
3. ✅ Haiku test passes with minute word in output
4. ✅ Output quality matches llama.cpp

---

**Team GREEN 🌿**  
*"Fresh eyes, fresh approach"*
