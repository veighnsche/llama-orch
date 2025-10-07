# Team AEGIS Investigation Report

**Date:** 2025-10-07T23:28Z  
**Status:** ❌ FALSE FIX - Repeated earlier failed path without llama.cpp verification  
**Result:** No progress made, wasted time on closed leads

---

## 🎯 Mission Summary

Inherited status from Team SENTINEL: matmul parity proven for Q/K/V but output still mojibake.  
Focus: Post-matmul pipeline (sampling, decoding, logits processing).

## ❌ FALSE FIX: lm_head cuBLAS Parameters

### What I Tried

Saw Team SENTINEL's manual verification failures:
- Position 0: Manual=-0.021, cuBLAS=-2.234, **Diff=2.21** ❌
- Position 8850: Manual=-4.650, cuBLAS=3.050, **Diff=7.70** ❌

Attempted fix: Change from **CUBLAS_OP_T + lda=896** to **CUBLAS_OP_N + lda=151936**

### Why This Failed

1. **This path was already explored by earlier teams** and discarded
2. **Manual verification passed** but output still mojibake/repetitive
3. **Never compared against llama.cpp** - only checked internal consistency
4. Passing manual checks doesn't prove correctness without ground truth

### Lesson Learned

**"Manual verification passes" ≠ "Fix is correct"**

Must compare outputs against llama.cpp at every stage:
- Hidden states after each layer
- Logits distribution  
- Generated token IDs

Without external parity checks, just validating our own math in a closed loop.

---

## ❌ FALSE LEADS Investigated

### 1. Temperature Parameter (FALSE_LEAD)

**SUSPECT:** Temperature shows 0.00 in logs instead of 0.7

**WHAT I MISSED:** Instrumentation only captured **prefill tokens** (which use temp=0.0 by design), not generation tokens

**RESULT:** Wasted time chasing a non-issue

### 2. UTF-8 Decode Path (FALSE_LEAD)

**PLAN:** Added byte-level decode logging to investigate mojibake

**OBSERVED:** Decode path works correctly - bytes match expected tokens

**RESULT:** Problem is **upstream** (model generating wrong tokens), not in decode

---

## 🔍 What Next Team Should Actually Do

### ⚠️ CRITICAL: Compare Against llama.cpp Ground Truth

**The Real Problem:** No team has compared our outputs against llama.cpp at the **per-layer level**.

**What's Needed:**

1. **Run llama.cpp with logging** for the SAME prompt:
   ```bash
   # Use llama.cpp with debug logging enabled
   llama-cli -m model.gguf -p "Write a haiku..." --log-disable 0
   ```

2. **Compare layer-by-layer:**
   - Hidden states after each transformer layer (all 24 layers)
   - Final logits (top-20 token IDs and values)
   - First 10 generated token IDs

3. **Find the divergence point:**
   - If layer 0 differs → embedding or first matmul wrong
   - If layer 5 differs → earlier layers propagate error
   - If final logits differ → lm_head wrong
   - If logits match but tokens differ → sampling wrong

**Why this matters:** We keep fixing things based on our own internal checks, but never verify against what llama.cpp actually produces. That's a closed loop.

### Lower Priority (only after llama.cpp parity)

- RNG seeding verification
- Temperature parameter tracing  
- Logits distribution analysis

---

## 📁 Files Modified (Then Reverted)

All changes have been **reverted** as they did not fix the underlying issue:

1. **cuda/src/transformer/qwen_transformer.cpp**
   - Added FALSE_FIX comment documenting the failed CUBLAS_OP_N attempt
   
2. **cuda/src/ffi_inference.cpp**
   - Removed instrumentation that only captured prefill tokens
   - Added FALSE_LEAD comment

3. **src/inference/cuda_backend.rs**
   - Removed byte-level decode logging
   - Added FALSE_LEAD comment

---

## ❌ Status: No Progress

- ❌ Repeated earlier failed fix without new evidence
- ❌ Did not compare against llama.cpp outputs
- ❌ Chased false leads (temperature, decode path)
- ❌ No actionable findings for next team

---

## 📝 Lessons Learned

1. **Read previous investigations carefully** - don't repeat closed paths
2. **External validation required** - llama.cpp parity, not just internal math
3. **Instrument the right phase** - generation tokens, not prefill
4. **Don't declare victory prematurely** - passing one check doesn't mean the fix works

---

**Team AEGIS**  
*"No progress made. Next team: compare against llama.cpp layer-by-layer."*
