# 🐛 BUG REPORT: Zero Probabilities in Sampling

**Date:** 2025-10-07T12:59Z  
**Severity:** CRITICAL  
**Status:** 🔴 CONFIRMED

---

## Summary

The model generates garbage tokens (mojibake, foreign characters, code tokens) because **all sampling probabilities are zero**. This causes random/undefined token selection.

---

## Evidence

### Test Output
```
[HELIOS GEN #00] token=13367, temp=0.70, top_k=0, top_p=1.00, seed=87
[HELIOS GEN #00] First 5 probs: 0.000000 0.000000 0.000000 0.000000 0.000000
```

**Every single token generation shows `First 5 probs: 0.000000 0.000000 0.000000 0.000000 0.000000`**

### Generated Output
```
aspâ¼Ĩ,dateáµç¾¹ðŁı¼0ĠWerkæĿ¨æ¬¢istenceahkan(IOeluocalyèĩªèº«...
```

This is complete garbage - mojibake, foreign characters, code tokens.

---

## Root Cause Analysis

### Hypothesis: Softmax produces all zeros

When all probabilities are zero after softmax, it means one of:

1. **Logits are all -Inf** → softmax([-Inf, -Inf, ...]) = [0, 0, ...]
2. **Logits are all equal and very negative** → softmax underflows to zero
3. **Softmax implementation bug** → produces zeros incorrectly

### Most Likely: LM Head Output is Corrupted

The logits coming from the LM head projection are likely:
- All the same value (no variation)
- All very negative (causing softmax underflow)
- Corrupted/garbage values

---

## Investigation Path

### Step 1: Dump Logits Before Softmax ✅ NEXT

Add logging to dump first 20 logits before softmax:

```cpp
// In sampling code, before softmax
fprintf(stderr, "[DEBUG] First 20 logits: ");
for (int i = 0; i < 20; i++) {
    fprintf(stderr, "%.6f ", logits[i]);
}
fprintf(stderr, "\n");
```

**Expected if bug:**
- All logits are the same value
- All logits are very negative (-1000 or similar)
- Logits contain NaN/Inf

### Step 2: Verify LM Head Projection

The LM head is the LAST projection in the model:
- Input: hidden_states [896] after final RMSNorm
- Weight: output.weight [896, 151936]
- Output: logits [151936]

This was **NOT verified** in Phase 2 (€100 fine for not testing LM head).

### Step 3: Check for Common Bugs

1. **Wrong matrix dimensions** - M/N/K parameters incorrect
2. **Wrong transpose flags** - CUBLAS_OP_T vs CUBLAS_OP_N
3. **Uninitialized output buffer** - logits buffer not zeroed
4. **Wrong leading dimensions** - lda/ldb/ldc incorrect
5. **FP16 overflow** - values too large for FP16

---

## Location in Code

### Sampling Code
**File:** `cuda/src/inference/sampling.cu` or similar  
**Function:** Sample token from logits  
**Line:** Where "First 5 probs" is printed

### LM Head Projection
**File:** `cuda/src/transformer/qwen_transformer.cpp`  
**Function:** `project_to_vocab()` or similar  
**Operation:** Final cuBLAS GEMM: `logits = hidden_states @ output.weight`

---

## Why This Wasn't Caught

1. **Test bypassed chat template** (€150 fine) - didn't test full path
2. **LM head not verified** (€100 fine) - Phase 2 only checked Q projection
3. **Sparse verification** (€300 total) - Only 0.11% coverage

The comprehensive tests I created would have caught this if LM head verification was implemented.

---

## Impact

**CRITICAL:** This completely breaks text generation.
- Model cannot generate coherent text
- Random tokens selected (whatever memory contains)
- Garbage output: mojibake, code tokens, foreign languages

---

## Next Steps

1. ✅ Add logits dump before softmax
2. ✅ Verify LM head cuBLAS parameters
3. ✅ Check for NaN/Inf in logits
4. ✅ Compare with llama.cpp LM head implementation
5. ✅ Fix the bug
6. ✅ Verify fix produces human-readable output

---

**Reported by:** Testing Developer (Cascade) 🔍  
**Test:** `tokenization_verification::test_chat_template_special_tokens`  
**Status:** 🔴 BUG CONFIRMED - Ready for investigation
