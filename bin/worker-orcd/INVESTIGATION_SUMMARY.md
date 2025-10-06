# Investigation Summary: lm_head Matrix Multiplication Bug

**Date**: 2025-10-06  
**Investigator**: AI Assistant  
**Status**: ✅ Root cause identified, fix ready for implementation

---

## Quick Summary

**Problem**: Model generates the same token repeatedly due to garbage values in logits at specific positions (8850, 44394, 137131).

**Root Cause**: Incorrect cuBLAS matrix multiplication parameters - wrong transpose flag AND wrong matrix dimensions.

**Solution**: Change both the transpose flag and matrix dimension order to match llama.cpp's working implementation.

---

## Investigation Timeline

1. ✅ **Analyzed llama.cpp BLAS implementation** - Found it uses row-major layout
2. ✅ **Analyzed llama.cpp CUDA implementation** - Found exact cuBLAS parameters
3. ✅ **Compared with current implementation** - Identified TWO critical differences
4. ✅ **Understood why previous fix failed** - Only changed transpose flag, not dimensions
5. ✅ **Created detailed action plan** - Ready for implementation

---

## Key Documents

### Investigation Reports
- **`LLAMA_CPP_MATRIX_ANALYSIS.md`** - Analysis of llama.cpp implementation
- **`CURRENT_VS_LLAMA_CPP_COMPARISON.md`** - Side-by-side parameter comparison
- **`ACTION_PLAN_REVISED.md`** - Step-by-step implementation guide

### Historical Context
- **`FIX_ATTEMPT_FAILED.md`** - Why the previous fix failed
- **`COMPLETE_INVESTIGATION_REPORT.md`** - Full investigation history
- **`FINAL_DIAGNOSIS.md`** - Technical root cause analysis

---

## The Problem

### Current Implementation (WRONG)

```cpp
cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,  // ❌ No transpose
    config_.vocab_size, batch_size, config_.hidden_dim,  // ❌ vocab, batch, hidden
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,
    hidden_half, CUDA_R_16F, config_.hidden_dim,
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Issues**:
1. ❌ `CUBLAS_OP_N` for lm_head (should be `CUBLAS_OP_T`)
2. ❌ `m = vocab_size` (should be `m = hidden_dim`)

### llama.cpp Implementation (CORRECT)

```cpp
cublasGemmStridedBatchedEx(handle, 
    CUBLAS_OP_T, CUBLAS_OP_N,  // ✅ Transpose lm_head
    896, 1, 896,               // ✅ hidden, batch, hidden
    alpha, 
    src0_ptr, cu_data_type_a, 151936, sma,
    src1_ptr, cu_data_type_b, 896, smb,
    beta,  
    dst_t, cu_data_type, 151936, ne1*ne0,
    ne12*ne13, cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

**Correct parameters**:
1. ✅ `CUBLAS_OP_T` for lm_head (transpose it)
2. ✅ `m = 896` (hidden_dim)

---

## The Solution

### Change Required

**File**: `cuda/src/transformer/qwen_transformer.cpp` line 557

**Change 1**: `CUBLAS_OP_N` → `CUBLAS_OP_T` (first parameter)  
**Change 2**: `config_.vocab_size` → `config_.hidden_dim` (m dimension)

```cpp
cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,  // ✅ CHANGED
    config_.hidden_dim,        // ✅ CHANGED
    batch_size,                // unchanged
    config_.hidden_dim,        // unchanged
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,  // unchanged
    hidden_half, CUDA_R_16F, config_.hidden_dim,   // unchanged
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,        // unchanged
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

---

## Why This Should Work

### Mathematical Correctness

Given:
- lm_head: `[896, 151936]` (hidden × vocab) in GGUF
- hidden: `[1, 896]` (batch × hidden)
- Want: logits `[1, 151936]` (batch × vocab)

**Correct operation**: `logits = hidden @ lm_head^T`

With our fix:
- cuBLAS treats lm_head as `[hidden, vocab]` = `[896, 151936]`
- `CUBLAS_OP_T` transposes it to `[vocab, hidden]` = `[151936, 896]`
- Computes: `C = A^T @ B = [151936, 896] @ [896, 1] = [151936, 1]` ✅

### Why Previous Fix Failed

**Previous attempt**: Changed `CUBLAS_OP_N, CUBLAS_OP_N` to `CUBLAS_OP_T, CUBLAS_OP_N`  
**But kept**: `m = vocab_size` (151936)

**Problem**: This told cuBLAS:
- "Take matrix A of size [vocab, hidden] and transpose it"
- But the actual matrix is [hidden, vocab]!
- Result: Catastrophic failure

**Our fix**: Changes BOTH transpose flag AND dimensions
- "Take matrix A of size [hidden, vocab] and transpose it"
- Matches the actual matrix layout ✅

---

## Confidence Level

**High (85%)**

**Reasons for confidence**:
1. ✅ Exact match with llama.cpp's working implementation
2. ✅ Mathematically correct
3. ✅ Explains why previous fix failed
4. ✅ Addresses the root cause

**Remaining uncertainty (15%)**:
- Why does the current wrong code produce mostly correct results?
- Could there be additional issues we haven't discovered?

---

## Next Steps

1. **Implement the fix** - Follow `ACTION_PLAN_REVISED.md`
2. **Test thoroughly** - Run the haiku generation test
3. **Verify results** - Check logits at problematic positions
4. **Document outcome** - Update investigation docs

---

## Success Criteria

- ✅ No CUDA errors
- ✅ All logits in expected range (-4 to +8)
- ✅ No garbage at positions 8850, 44394, 137131
- ✅ Model generates different tokens each step
- ✅ Test passes with coherent output

---

## If the Fix Fails

Investigate alternative theories:
1. **Tensor loading issue** - Check if lm_head is loaded correctly from GGUF
2. **Memory corruption** - Something overwrites specific positions
3. **Alignment issue** - GPU memory alignment problems
4. **cuBLAS bug** - Edge case with specific dimensions

See `ACTION_PLAN_REVISED.md` for detailed fallback procedures.

---

## Key Learnings

1. **Don't change parameters in isolation** - Transpose flags and dimensions must match
2. **Always compare with working reference** - llama.cpp was the key to solving this
3. **Understand the math** - Matrix dimensions must make mathematical sense
4. **Test incrementally** - Add debug output before and after changes

---

**Status**: Investigation complete, ready for implementation  
**Confidence**: High  
**Risk**: Medium (previous fix failed, but we understand why)  
**Next Action**: Implement the fix following `ACTION_PLAN_REVISED.md`
