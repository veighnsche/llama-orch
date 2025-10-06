# Root Cause Analysis - Matrix Multiplication Bug

**Date**: 2025-10-06  
**Status**: ✅ ROOT CAUSE IDENTIFIED AND FIXED

**UPDATE 2025-10-06 10:49**: Fix applied and tested successfully.  
Q values now in correct range (0.01-0.26). Matrix layout issue resolved.  
See `TEST_RESULTS_AFTER_FIX.md` for current status and remaining issues.

---

## The Bug

Your implementation produces Q values that are **10-100x larger** than llama.cpp, causing garbage output.

### Evidence
- **llama.cpp Q values**: ~0.01 to 1.13 (reasonable)
- **Your Q values**: ~0.02 to -13.34 (HUGE outliers)

---

## Root Cause: Matrix Layout Mismatch

### How llama.cpp Does It

1. **Weight Storage**: `wq` is stored as `[hidden_dim, q_dim]` in GGUF
2. **Matrix Multiplication**: `ggml_mul_mat(wq, cur)` where:
   - `wq`: `[hidden_dim, q_dim]`
   - `cur`: `[hidden_dim, batch]`
3. **ggml_mul_mat Semantics** (from ggml.h):
   ```
   // A: k columns, n rows => [ne03, ne02, n, k]
   // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
   // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
   ```
4. **What it computes**: `result = wq^T @ cur`
   - Treats `wq` as `[q_dim, hidden_dim]` (conceptually transposed)
   - Transposes `cur` internally to `[batch, hidden_dim]`
   - Result: `[batch, q_dim]`

### How Your cuBLAS Code Does It

```cpp
cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose W_q, don't transpose input
    q_dim, batch_size, config_.hidden_dim,
    &alpha,
    layer.attn_q_weight, CUDA_R_16F, config_.hidden_dim,  // ❌ WRONG lda
    normed_half, CUDA_R_16F, config_.hidden_dim,
    &beta,
    q_half, CUDA_R_16F, q_dim,
    ...
);
```

**The Problem**: The `lda` (leading dimension) parameter is **WRONG**.

---

## Understanding cuBLAS Leading Dimensions

In cuBLAS, matrices are stored in **column-major** order. The leading dimension (`lda`, `ldb`, `ldc`) is the **stride between columns** in memory.

### For CUBLAS_OP_T (Transpose)

When you specify `CUBLAS_OP_T`, cuBLAS:
1. Reads the matrix in its **original** (non-transposed) layout
2. Applies the transpose **logically** during computation
3. The `lda` parameter must be the leading dimension of the **original** matrix

### Your Weight Matrix

- **Stored shape**: `[hidden_dim, q_dim]` = `[896, 896]`
- **Memory layout** (column-major): 
  - Each column has `hidden_dim` elements
  - Leading dimension = `hidden_dim`
- **When transposed** (CUBLAS_OP_T):
  - Logical shape becomes `[q_dim, hidden_dim]`
  - But `lda` should still be `hidden_dim` (the original LD)

### What You're Doing Wrong

You're setting `lda = config_.hidden_dim`, which is **correct**!

Wait... let me re-check the actual issue.

---

## Re-Analysis: The Real Problem

Actually, looking more carefully at the cuBLAS call:

```cpp
cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,
    q_dim, batch_size, config_.hidden_dim,  // m, n, k
    &alpha,
    layer.attn_q_weight, CUDA_R_16F, config_.hidden_dim,  // A, lda
    normed_half, CUDA_R_16F, config_.hidden_dim,          // B, ldb
    &beta,
    q_half, CUDA_R_16F, q_dim,                            // C, ldc
    ...
);
```

This computes: `C = alpha * op(A) @ op(B) + beta * C`

Where:
- `op(A) = A^T` (because CUBLAS_OP_T)
- `op(B) = B` (because CUBLAS_OP_N)
- `A` is `[hidden_dim, q_dim]` stored in column-major
- `A^T` is `[q_dim, hidden_dim]`
- `B` is `[hidden_dim, batch]` stored in column-major
- Result `C` is `[q_dim, batch]`

**This looks correct!**

---

## The ACTUAL Problem: Weight Storage Order

The issue is that GGUF stores weights in **row-major** order, but cuBLAS expects **column-major** order!

### GGUF Storage
- Weights are stored in **row-major** order
- `wq` with shape `[hidden_dim, q_dim]` means:
  - First `q_dim` elements are row 0
  - Next `q_dim` elements are row 1
  - etc.

### cuBLAS Expectation
- Matrices are in **column-major** order
- A matrix with shape `[rows, cols]` means:
  - First `rows` elements are column 0
  - Next `rows` elements are column 1
  - etc.

### The Fix

When you load a GGUF tensor with shape `[hidden_dim, q_dim]` in row-major order, it's equivalent to a cuBLAS matrix with shape `[q_dim, hidden_dim]` in column-major order!

So you need to:
1. **Don't transpose** the weight matrix (use `CUBLAS_OP_N`)
2. **Adjust the leading dimension** to `q_dim` (not `hidden_dim`)

---

## Correct cuBLAS Call

```cpp
// Q projection: q = W_q @ x
// W_q in GGUF: [hidden_dim, q_dim] row-major
// W_q in cuBLAS: [q_dim, hidden_dim] column-major (equivalent)
// Input x: [hidden_dim, batch] column-major
// Result: [q_dim, batch] column-major

cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose needed!
    q_dim, batch_size, config_.hidden_dim,
    &alpha,
    layer.attn_q_weight, CUDA_R_16F, q_dim,  // ✅ lda = q_dim (row-major → col-major)
    normed_half, CUDA_R_16F, config_.hidden_dim,
    &beta,
    q_half, CUDA_R_16F, q_dim,
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

---

## Summary

**The bug**: You're treating GGUF's row-major weights as if they were column-major, causing incorrect matrix multiplication.

**The fix**: Change `CUBLAS_OP_T` to `CUBLAS_OP_N` and set `lda = q_dim` instead of `lda = hidden_dim`.

This applies to:
- Q projection
- K projection  
- V projection
- Attention output projection
- FFN projections

---

## Next Steps

1. ✅ Update all cuBLAS matrix multiplications to use correct layout
2. ⏳ Test with debug output to verify Q values match llama.cpp
3. ⏳ Re-enable bias addition (investigate bias values separately)
4. ⏳ Run full generation test
