# START HERE - Bug Investigation Summary

**Date**: 2025-10-06  
**Status**: ROOT CAUSE IDENTIFIED - FIX NEEDED

---

## Quick Summary

**Problem**: Model generates same token ("coholic") 100 times in a row

**Original Theory**: Vocab size mismatch (lm_head is 151643 but vocab is 151936)  
**Reality**: NO MISMATCH - lm_head IS 151936, but logits have garbage values at scattered positions

**Root Cause**: Positions 8850, 44394, 137131 (and likely others) in logits buffer contain garbage values (~14-15) instead of correct values (-4 to +4). Argmax selects these garbage values.

---

## Read These Documents (In Order)

1. **This file** - Quick overview
2. **COMPLETE_INVESTIGATION_REPORT.md** - Everything we learned (MOST IMPORTANT)
3. **BUG_STATUS_UPDATED.md** - Current status and next steps
4. **FINAL_DIAGNOSIS.md** - Technical details of root cause

---

## What We Know For Sure

✅ **Model file is valid** - Works perfectly in llama.cpp  
✅ **Tensor dimensions correct** - output.weight is [896, 151936]  
✅ **Weights are correct** - Sampled lm_head values are normal  
✅ **Hidden state is correct** - No extreme values  
✅ **GEMM parameters look correct** - Dimensions match expected  

❌ **Logits have garbage** - Specific positions have values 3-4x too high  
❌ **Garbage changes over time** - Position 44394 goes from 12.34 to 15.19  
❌ **Scattered throughout vocab** - Not just at end (found at 8850, 44394, 137131)  

---

## Next Steps

### 1. Compare with llama.cpp (CRITICAL)

```bash
cd reference/llama.cpp
grep -r "output.weight" src/
grep -r "cublasGemmEx" ggml/src/ggml-cuda/
```

**Focus**: How does llama.cpp handle lm_head tensor layout?

### 2. Check Memory Layout

**Theory**: Tensor might not be transposed correctly from GGUF row-major to cuBLAS column-major

**Test**: Manually compute dot product for position 44394 and compare with GEMM result

### 3. Try Different cuBLAS Settings

- Change CUBLAS_COMPUTE_32F_FAST_16F → CUBLAS_COMPUTE_32F
- Change CUBLAS_GEMM_DEFAULT_TENSOR_OP → CUBLAS_GEMM_DEFAULT
- Try explicit transpose instead of CUBLAS_OP_N

---

## Don't Waste Time On

❌ Vocab size mismatch - doesn't exist  
❌ Filling positions with -INFINITY - garbage is everywhere  
❌ Checking if weights corrupted - they're fine  
❌ Checking if hidden state corrupted - it's fine  

---

## Test Command

```bash
cd bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

---

## Key Insight

**llama.cpp works with the same model file.**  
Whatever we're doing differently is the bug.  
The answer is in comparing our code with theirs.

---

**Good luck!** 🚀

See COMPLETE_INVESTIGATION_REPORT.md for full details.
