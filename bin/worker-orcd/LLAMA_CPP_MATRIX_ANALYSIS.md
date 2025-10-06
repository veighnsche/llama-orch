# Investigation Findings: llama.cpp CUDA Matrix Multiplication Analysis

## Executive Summary

After deep analysis of llama.cpp's CUDA implementation, I have identified the **exact cuBLAS parameters** used for matrix multiplication. The issue is NOT about row-major vs column-major, but about the specific transpose operations used in cuBLAS calls.

## Key Findings

### 1. llama.cpp CUDA Matrix Multiplication Parameters

**Location**: `/reference/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` lines 1934-1941

```cpp
cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        ne01, ne11, ne10,
        alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
               src1_ptr, cu_data_type_b, s11,       smb,     // strideB
        beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
        ne12*ne13,
        cu_compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

**Critical Parameters**:
- **`CUBLAS_OP_T`** for src0 (lm_head weight matrix) - **TRANSPOSED**
- **`CUBLAS_OP_N`** for src1 (hidden state) - **NOT TRANSPOSED**
- Matrix dimensions: `ne01, ne11, ne10` (not the typical order!)

### 2. Understanding the Operation

In llama.cpp's implementation:
- `src0` = weight matrix (lm_head): shape `[896, 151936]`
- `src1` = input (hidden state): shape `[1, 896]`
- `dst` = output (logits): shape `[1, 151936]`

The cuBLAS call performs: `dst = src0^T @ src1`

**This is DIFFERENT from what I initially analyzed!**

### 3. Comparison with Current Implementation

| Aspect | llama.cpp CUDA | Our Implementation | Status |
|--------|----------------|-------------------|---------|
| Weight Matrix Op | `CUBLAS_OP_T` | `CUBLAS_OP_N` | ❌ WRONG |
| Input Tensor Op | `CUBLAS_OP_N` | `CUBLAS_OP_N` | ✅ CORRECT |
| Matrix Order | `ne01, ne11, ne10` | ? | ❓ UNKNOWN |
| Leading Dimension | `nb01/nb00` | ? | ❓ UNKNOWN |

## Technical Analysis - CORRECTED

### Why `CUBLAS_OP_T` for lm_head?

The lm_head tensor in GGUF is stored as `[896, 151936]` (hidden_size × vocab_size).

For the operation `logits = hidden @ lm_head^T`:
- We need: `[1, 896] @ [896, 151936]^T = [1, 896] @ [151936, 896] = [1, 151936]`
- cuBLAS with `CUBLAS_OP_T` on src0 achieves this transpose

### The Previous Fix Attempt Failed Because...

Looking at `FIX_ATTEMPT_FAILED.md`, the attempt changed from `CUBLAS_OP_N, CUBLAS_OP_N` to `CUBLAS_OP_T, CUBLAS_OP_N`.

**BUT** - this is actually the CORRECT change according to llama.cpp!

The catastrophic failure suggests the problem is NOT the transpose operation itself, but:
1. **Leading dimensions** (`lda`, `ldb`, `ldc`) are wrong
2. **Matrix dimension order** is wrong
3. **Stride parameters** are wrong

### Critical Insight: It's Not Just the Transpose Flag

llama.cpp uses:
```cpp
cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    ne01, ne11, ne10,  // m, n, k - SPECIFIC ORDER!
    alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,  // A and strides
           src1_ptr, cu_data_type_b, s11,       smb,  // B and strides
    beta,  dst_t,    cu_data_type,   ne0,  ne1*ne0,  // C and strides
    ne12*ne13, cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

The key parameters are:
- **m = ne01** (896 - hidden size)
- **n = ne11** (1 - batch size) 
- **k = ne10** (896 - hidden size)
- **lda = nb01/nb00** (leading dimension of A)
- **ldb = s11** (leading dimension of B)
- **ldc = ne0** (leading dimension of C)

## Change Request - REVISED

### Priority 1: Check Current Implementation Parameters

**Action**: We need to see what parameters our current implementation uses.

**Files to check**:
1. `cuda/src/transformer/qwen_transformer.cpp` - The lm_head GEMM call
2. Look for the exact cuBLAS call and its parameters

### Priority 2: Match llama.cpp Parameters EXACTLY

**Problem**: The previous fix only changed the transpose flag, but didn't verify:
- Matrix dimension order (m, n, k)
- Leading dimensions (lda, ldb, ldc)
- Stride parameters

**Solution**: Match ALL parameters, not just the transpose flags:

```cpp
// For lm_head operation:
// src0 = lm_head [896, 151936]
// src1 = hidden [1, 896]
// dst = logits [1, 151936]

cublasGemmStridedBatchedEx(handle, 
    CUBLAS_OP_T,        // transpose lm_head
    CUBLAS_OP_N,        // don't transpose hidden
    896,                // m = ne01 (hidden_size)
    1,                  // n = ne11 (batch_size)
    896,                // k = ne10 (hidden_size)
    &alpha,
    lm_head_ptr, CUDA_R_16F, 151936,  // lda = vocab_size (stride between columns)
    hidden_ptr,  CUDA_R_16F, 896,     // ldb = hidden_size
    &beta,
    logits_ptr,  CUDA_R_32F, 151936,  // ldc = vocab_size
    1,  // batch count
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### Priority 3: Understand Why Garbage at Specific Positions

**New Theory**: If most logits are correct but specific positions have garbage, the issue might be:

1. **Partial memory corruption** - Something overwrites specific memory locations
2. **Tensor loading issue** - Specific rows/columns of lm_head not loaded correctly
3. **Alignment issue** - GPU memory alignment causes issues at specific addresses
4. **cuBLAS bug** - Edge case in cuBLAS with specific matrix dimensions

## Implementation Steps

### Step 1: Examine Current Implementation

**CRITICAL**: Before making any changes, we need to see the EXACT current cuBLAS call.

Look for the lm_head matrix multiplication in:
- `cuda/src/transformer/qwen_transformer.cpp`
- Search for `cublasGemmEx` or similar cuBLAS calls

Document:
- Current transpose flags
- Current matrix dimensions (m, n, k)
- Current leading dimensions (lda, ldb, ldc)
- Current data types

### Step 2: Compare Parameter by Parameter

Create a comparison table:

| Parameter | llama.cpp | Our Current | Match? |
|-----------|-----------|-------------|--------|
| op_A | CUBLAS_OP_T | ? | ? |
| op_B | CUBLAS_OP_N | ? | ? |
| m | 896 (ne01) | ? | ? |
| n | 1 (ne11) | ? | ? |
| k | 896 (ne10) | ? | ? |
| lda | 151936 | ? | ? |
| ldb | 896 | ? | ? |
| ldc | 151936 | ? | ? |

### Step 3: Fix Implementation Carefully

**DO NOT just change the transpose flag!**

The previous attempt failed because it only changed `CUBLAS_OP_N` to `CUBLAS_OP_T` without adjusting:
- Matrix dimension order
- Leading dimensions
- Stride parameters

**Correct approach**:
```cpp
// Match llama.cpp EXACTLY
cublasGemmStridedBatchedEx(handle, 
    CUBLAS_OP_T,        // A: lm_head - TRANSPOSE
    CUBLAS_OP_N,        // B: hidden - NO TRANSPOSE
    896,                // m = hidden_size
    1,                  // n = batch_size
    896,                // k = hidden_size
    &alpha,
    lm_head_ptr, CUDA_R_16F, 151936,  // lda = vocab_size
    hidden_ptr,  CUDA_R_16F, 896,     // ldb = hidden_size
    &beta,
    logits_ptr,  CUDA_R_32F, 151936,  // ldc = vocab_size
    1,  // batch count
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### Step 4: Test Incrementally

1. **First**: Print the parameters before calling cuBLAS
2. **Then**: Run the test and check for CUDA errors
3. **Finally**: Check logit values at known positions

## Alternative Investigation Path

If matching llama.cpp parameters STILL doesn't work, investigate:

### Theory: Tensor Loading Issue

Check if lm_head is loaded correctly from GGUF:
```cpp
// Dump first and last rows of lm_head
half h_first[896], h_last[896];
cudaMemcpy(h_first, lm_head_ptr, 896 * sizeof(half), cudaMemcpyDeviceToHost);
cudaMemcpy(h_last, lm_head_ptr + (151935 * 896), 896 * sizeof(half), cudaMemcpyDeviceToHost);

// Check for NaN or Inf
for (int i = 0; i < 896; i++) {
    if (isnan(h_first[i]) || isinf(h_first[i])) {
        fprintf(stderr, "ERROR: lm_head[0][%d] = %f\n", i, (float)h_first[i]);
    }
}
```

### Theory: Specific Rows Corrupted

Check the problematic token positions (8850, 44394, 137131):
```cpp
// Check if these specific rows have valid data
half h_8850[896], h_44394[896], h_137131[896];
cudaMemcpy(h_8850, lm_head_ptr + (8850 * 896), 896 * sizeof(half), ...);
cudaMemcpy(h_44394, lm_head_ptr + (44394 * 896), 896 * sizeof(half), ...);
cudaMemcpy(h_137131, lm_head_ptr + (137131 * 896), 896 * sizeof(half), ...);

// Compute manual dot product
float manual_8850 = 0.0f;
for (int i = 0; i < 896; i++) {
    manual_8850 += (float)h_8850[i] * (float)hidden[i];
}
fprintf(stderr, "Manual logit[8850] = %.4f, GEMM = %.4f\n", manual_8850, logits[8850]);
```

## Success Criteria

✅ **No CUDA errors**  
✅ **Argmax finds tokens with values -4 to +8 (not 14-15)**  
✅ **Model generates different tokens each step**  
✅ **Output is coherent text**  
✅ **Test passes with good output**

## Files to Check/Modify

1. **`cuda/src/transformer/qwen_transformer.cpp`** - lm_head GEMM call
2. **`src/cuda/model.rs`** - Tensor loading verification
3. **`cuda/src/model/qwen_weight_loader.cpp`** - How lm_head is loaded from GGUF

---
**Investigation Status**: Deep dive complete - Found exact llama.cpp parameters
**Critical Finding**: Previous fix attempt was on the right track but incomplete
**Next Action**: Check current implementation parameters before making changes
