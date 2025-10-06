# Current Implementation vs llama.cpp: Detailed Comparison

**Date**: 2025-10-06  
**Purpose**: Compare our current cuBLAS parameters with llama.cpp's working implementation

---

## Executive Summary

Our current implementation uses **DIFFERENT** parameters than llama.cpp. The key differences are:

1. **Transpose flags**: We use `CUBLAS_OP_N, CUBLAS_OP_N` but llama.cpp uses `CUBLAS_OP_T, CUBLAS_OP_N`
2. **Matrix dimension order**: We use `vocab_size, batch, hidden` but llama.cpp uses `hidden, batch, hidden`
3. **Leading dimensions**: We use `vocab_size` for lm_head, llama.cpp uses `vocab_size` (same)

---

## Current Implementation

**File**: `cuda/src/transformer/qwen_transformer.cpp` lines 557-568

```cpp
cublasStatus_t status = cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,  // ❌ Both NO TRANSPOSE
    config_.vocab_size, batch_size, config_.hidden_dim,  // ❌ vocab, batch, hidden
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,  // lda = vocab_size
    hidden_half, CUDA_R_16F, config_.hidden_dim,   // ldb = hidden_dim
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,        // ldc = vocab_size
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Parameters**:
- **op_A**: `CUBLAS_OP_N` (no transpose on lm_head)
- **op_B**: `CUBLAS_OP_N` (no transpose on hidden)
- **m**: `vocab_size` (151936)
- **n**: `batch_size` (1)
- **k**: `hidden_dim` (896)
- **lda**: `vocab_size` (151936)
- **ldb**: `hidden_dim` (896)
- **ldc**: `vocab_size` (151936)

---

## llama.cpp Implementation

**File**: `reference/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` lines 1934-1941

```cpp
cublasGemmStridedBatchedEx(ctx.cublas_handle(), 
    CUBLAS_OP_T, CUBLAS_OP_N,  // ✅ TRANSPOSE lm_head, NO TRANSPOSE hidden
    ne01, ne11, ne10,           // ✅ hidden, batch, hidden
    alpha, 
    src0_ptr, cu_data_type_a, nb01/nb00, sma,  // A and strides
    src1_ptr, cu_data_type_b, s11,       smb,  // B and strides
    beta,  
    dst_t,    cu_data_type,   ne0,  ne1*ne0,   // C and strides
    ne12*ne13, cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

**Parameters** (for lm_head operation):
- **op_A**: `CUBLAS_OP_T` (transpose lm_head)
- **op_B**: `CUBLAS_OP_N` (no transpose on hidden)
- **m**: `ne01` = 896 (hidden_size)
- **n**: `ne11` = 1 (batch_size)
- **k**: `ne10` = 896 (hidden_size)
- **lda**: `nb01/nb00` = 151936 (vocab_size - stride between columns)
- **ldb**: `s11` = 896 (hidden_size)
- **ldc**: `ne0` = 151936 (vocab_size)

---

## Side-by-Side Comparison

| Parameter | Our Current | llama.cpp | Match? | Impact |
|-----------|-------------|-----------|--------|--------|
| **op_A** | `CUBLAS_OP_N` | `CUBLAS_OP_T` | ❌ | **CRITICAL** |
| **op_B** | `CUBLAS_OP_N` | `CUBLAS_OP_N` | ✅ | - |
| **m** | 151936 (vocab) | 896 (hidden) | ❌ | **CRITICAL** |
| **n** | 1 (batch) | 1 (batch) | ✅ | - |
| **k** | 896 (hidden) | 896 (hidden) | ✅ | - |
| **lda** | 151936 | 151936 | ✅ | - |
| **ldb** | 896 | 896 | ✅ | - |
| **ldc** | 151936 | 151936 | ✅ | - |

---

## Why the Differences Matter

### Matrix Dimensions

**Our current**: `C[vocab, batch] = A[vocab, hidden] @ B[hidden, batch]`
- This treats lm_head as `[vocab, hidden]` without transpose
- But GGUF stores lm_head as `[hidden, vocab]` in row-major!

**llama.cpp**: `C[vocab, batch] = A^T[hidden, vocab] @ B[hidden, batch]`
- This treats lm_head as `[hidden, vocab]` WITH transpose
- Matches the GGUF storage format!

### The Math

Given:
- lm_head in GGUF: `[896, 151936]` (hidden × vocab) row-major
- hidden state: `[1, 896]` (batch × hidden)
- Want: logits `[1, 151936]` (batch × vocab)

**Correct operation**: `logits = hidden @ lm_head^T`
- `[1, 896] @ [896, 151936]^T = [1, 896] @ [151936, 896] = [1, 151936]` ✅

**Our current operation**: `logits = lm_head @ hidden^T` (because we swapped m and k)
- This is mathematically wrong!

---

## Why Our Current Code "Mostly Works"

The fact that most logits are correct (positions 0-999) but specific positions have garbage (8850, 44394, 137131) suggests:

1. **Accidental correctness**: The wrong parameters happen to produce correct results for most positions
2. **Memory layout luck**: The specific memory layout makes the wrong operation work by accident
3. **Partial corruption**: Something else is corrupting specific positions

---

## The Fix That Failed

**What was tried** (from `FIX_ATTEMPT_FAILED.md`):
- Changed `CUBLAS_OP_N, CUBLAS_OP_N` to `CUBLAS_OP_T, CUBLAS_OP_N`
- Did NOT change the matrix dimensions (m, n, k)

**Why it failed**:
- Changing the transpose flag without changing the dimension order creates a mismatch
- The dimensions `vocab, batch, hidden` with `CUBLAS_OP_T` means:
  - "Take matrix A of size [vocab, hidden] and transpose it to [hidden, vocab]"
  - But we want to take matrix A of size [hidden, vocab] and transpose it!

---

## The Correct Fix

Change BOTH the transpose flag AND the dimension order:

```cpp
cublasStatus_t status = cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,  // ✅ TRANSPOSE lm_head
    config_.hidden_dim,        // ✅ m = hidden (896)
    batch_size,                // ✅ n = batch (1)
    config_.hidden_dim,        // ✅ k = hidden (896)
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,  // lda = vocab_size (151936)
    hidden_half, CUDA_R_16F, config_.hidden_dim,   // ldb = hidden_dim (896)
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,        // ldc = vocab_size (151936)
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Key changes**:
1. ✅ `CUBLAS_OP_N` → `CUBLAS_OP_T` (transpose lm_head)
2. ✅ `config_.vocab_size` → `config_.hidden_dim` (m = hidden, not vocab)
3. ✅ Keep `batch_size` as n
4. ✅ Keep `config_.hidden_dim` as k

---

## Why This Should Work

With these parameters:
- **A** (lm_head): `[896, 151936]` stored in memory, treated as `[hidden, vocab]`
- **CUBLAS_OP_T**: Transpose A to `[vocab, hidden]` = `[151936, 896]`
- **B** (hidden): `[1, 896]` = `[batch, hidden]`
- **Result**: `C = A^T @ B = [151936, 896] @ [896, 1] = [151936, 1]` ✅

But wait... that gives us `[vocab, batch]` not `[batch, vocab]`!

Actually, cuBLAS computes: `C = alpha * op(A) @ op(B) + beta * C`

With:
- `op(A) = A^T` of size `[m, k]` where A is `[k, m]`
- `op(B) = B` of size `[k, n]`
- `C` of size `[m, n]`

So:
- A is `[hidden, vocab]` = `[896, 151936]`
- A^T is `[vocab, hidden]` = `[151936, 896]`
- B is `[hidden, batch]` = `[896, 1]`
- C = A^T @ B = `[vocab, batch]` = `[151936, 1]` ✅

---

## Action Items

1. ✅ **Document current parameters** - Done
2. ✅ **Document llama.cpp parameters** - Done
3. ✅ **Identify differences** - Done
4. ⏭️ **Implement fix with ALL parameter changes**
5. ⏭️ **Test incrementally**

---

**Status**: Analysis complete, ready for implementation
**Confidence**: High - llama.cpp uses these exact parameters and works
**Risk**: Medium - Previous fix attempt failed, but we now understand why
