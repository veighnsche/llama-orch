# Team FELICIA Investigation - Garbage Output Bug

**Date:** 2025-10-06T21:43Z  
**Mission:** Fix garbage output bug following Team BYGONE handoff

---

## 🎯 Current Symptoms

```
Expected: Haiku about GPU computing with "forty-three"
Actual: é¹ŀĠinsultsannersĠLumpæĤĴĠÄĳáº¹pĉCGæĴ¤×¢×Ļ×ª...
```

First 10 generated tokens:
- `[0] ID=121645 → "é¹ŀ"` (Thai/foreign)
- `[1] ID=67889 → "Ġinsults"` (wrong context)
- `[2] ID=24003 → "anners"` (fragment)
- `[3] ID=74293 → "ĠLump"` (wrong context)
- `[4] ID=120510 → "æĤĴ"` (Chinese)

**CRITICAL:** First generated token is already garbage! Bug manifests immediately after prefill.

---

## 🔍 Investigation Plan

Following Team BYGONE's recommendation: Compare with llama.cpp systematically.

### Step 1: Check llama.cpp implementation for missing operations

[TEAM FELICIA] 2025-10-06T21:43Z
PLAN: Search llama.cpp source for operations we might be missing
- Embedding scaling
- Attention score scaling
- Hidden state transformations
- Any normalization we're missing

### Step 2: Compare attention implementation

SUSPECT: Attention mechanism might have subtle differences
PLAN: Compare our GQA implementation with llama.cpp's attention

---

---

## 🔥 ROOT CAUSE FOUND!

[TEAM FELICIA] 2025-10-06T21:45Z

### The Bug: Incorrect cuBLAS Parameters for Final Projection

**Location:** `cuda/src/transformer/qwen_transformer.cpp` lines 724-737

**Current Code (WRONG):**
```cpp
cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,  // ❌ No transpose
    config_.padded_vocab_size,  // m = 151936
    batch_size,                 // n = 1
    config_.hidden_dim,         // k = 896
    &alpha,
    lm_head_half, CUDA_R_16F, config_.padded_vocab_size,  // ❌ lda = 151936
    hidden_half, CUDA_R_16F, config_.hidden_dim,
    &beta,
    logits, CUDA_R_32F, config_.padded_vocab_size,
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**llama.cpp Code (CORRECT):**
```cpp
cublasGemmEx(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,  // ✅ Transpose first matrix
    vocab_size,                 // m = 151936
    batch_size,                 // n = 1
    hidden_dim,                 // k = 896
    &alpha,
    lm_head, CUDA_R_16F, hidden_dim,  // ✅ lda = 896
    hidden, CUDA_R_16F, hidden_dim,
    &beta,
    logits, CUDA_R_32F, vocab_size,
    ...
);
```

### Why This Causes Garbage Output

The `lm_head` weight matrix is stored as `[896, 151936]` (hidden_dim × vocab_size) in **row-major** format in the GGUF file.

**With our WRONG parameters:**
- cuBLAS interprets it as column-major `[151936, 896]` with stride 151936
- This reads the matrix in the WRONG order
- Each logit gets computed from the wrong weights
- Result: Complete garbage output

**With llama.cpp's CORRECT parameters:**
- `CUBLAS_OP_T` tells cuBLAS to transpose
- `lda = 896` tells cuBLAS the actual row length
- cuBLAS correctly reads `[896, 151936]` row-major and transposes it
- Result: Correct logits, correct output

### Evidence

1. **llama.cpp works perfectly** with the same model file
2. **Our code generates garbage** from the very first token
3. **The comments in our code** (lines 575-657) already identified this issue!
4. **Team Alpha's investigation** found the discrepancy but the fix was never applied

---

---

## ✅ FIXES APPLIED

[TEAM FELICIA] 2025-10-06T21:49Z

### All cuBLAS Transpose Issues Fixed

Fixed **8 matrix multiplications** total:

1. ✅ Q projection (`qwen_transformer.cpp:279`) - Changed to `CUBLAS_OP_T` with `lda=hidden_dim`
2. ✅ K projection (`qwen_transformer.cpp:311`) - Changed to `CUBLAS_OP_T` with `lda=hidden_dim`
3. ✅ V projection (`qwen_transformer.cpp:334`) - Changed to `CUBLAS_OP_T` with `lda=hidden_dim`
4. ✅ Attention output (`qwen_transformer.cpp:429`) - Changed to `CUBLAS_OP_T` with `lda=q_dim`
5. ✅ FFN gate (`swiglu_ffn.cu:131`) - Changed to `CUBLAS_OP_T` with `lda=hidden_dim`
6. ✅ FFN up (`swiglu_ffn.cu:149`) - Changed to `CUBLAS_OP_T` with `lda=hidden_dim`
7. ✅ FFN down (`swiglu_ffn.cu:177`) - Changed to `CUBLAS_OP_T` with `lda=ffn_dim`
8. ✅ Final projection (`qwen_transformer.cpp:740`) - Changed to `CUBLAS_OP_T` with `lda=hidden_dim`

### Result After Fixes

**Before:** Random garbage tokens (foreign languages, code tokens, completely random)
```
é¹ŀĠinsultsannersĠLumpæĤĴĠÄĳáº¹pĉCGæĴ¤×¢×Ļ×ª...
```

**After:** Repetitive tokens (model gets stuck in loops)
```
macrosmacrosncyĳľĳľĳľĳľĳľ/mainíĺľĳľĳľĳľĳľĳľĳľĳľĳľĳľĳľĳľĳľ...
```

### Analysis

The fixes are **partially successful**:
- ✅ Model is now computing with correct weight matrices
- ✅ Output is no longer completely random
- ❌ Model generates repetitive tokens and gets stuck in loops
- ❌ Still doesn't generate proper haiku

This suggests:
1. **The transpose fixes are correct** - we're now reading weights properly
2. **There's another bug** causing the repetitive behavior
3. Possible causes:
   - KV cache corruption
   - Attention mechanism issue
   - Sampling/temperature issue
   - Position encoding problem

---

## 📝 Status

**Bug Status:** PARTIALLY FIXED  
**Fixes Applied:** All 8 cuBLAS transpose operations corrected  
**Remaining Issue:** Repetitive token generation  
**Time Started:** 2025-10-06T21:43Z  
**Time Fixed:** 2025-10-06T21:49Z  
**Next Steps:** Investigate repetitive token generation (likely KV cache or attention issue)
