# Action Plan: Fix lm_head Matrix Multiplication

**Date**: 2025-10-06 13:08  
**Status**: Ready for implementation  
**Confidence**: High

---

## Summary of Investigation

After deep analysis of llama.cpp's CUDA implementation, we found:

1. ✅ **Current implementation parameters documented** - See `CURRENT_VS_LLAMA_CPP_COMPARISON.md`
2. ✅ **llama.cpp parameters documented** - Exact cuBLAS call identified
3. ✅ **Root cause identified** - Wrong matrix dimensions AND transpose flag

**Key Finding**: The previous fix attempt changed the transpose flag but NOT the matrix dimensions, causing catastrophic failure.

---

## The Fix

### Current Code (WRONG)

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

### Corrected Code (MATCH llama.cpp)

```cpp
cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,  // ✅ Transpose lm_head
    config_.hidden_dim, batch_size, config_.hidden_dim,  // ✅ hidden, batch, hidden
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,  // lda = vocab_size
    hidden_half, CUDA_R_16F, config_.hidden_dim,   // ldb = hidden_dim
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,        // ldc = vocab_size
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Changes**:
1. `CUBLAS_OP_N` → `CUBLAS_OP_T` (first parameter)
2. `config_.vocab_size` → `config_.hidden_dim` (m dimension)

---

## Implementation Steps

### Step 1: Add Debug Output BEFORE Making Changes

Add this before the cuBLAS call to verify current behavior:

```cpp
// Debug: Print current parameters
fprintf(stderr, "\n=== cuBLAS GEMM Parameters ===\n");
fprintf(stderr, "op_A: CUBLAS_OP_N, op_B: CUBLAS_OP_N\n");
fprintf(stderr, "m=%u (vocab), n=%u (batch), k=%u (hidden)\n", 
        config_.vocab_size, batch_size, config_.hidden_dim);
fprintf(stderr, "lda=%u, ldb=%u, ldc=%u\n",
        config_.vocab_size, config_.hidden_dim, config_.vocab_size);
fprintf(stderr, "lm_head shape: [%u, %u]\n", config_.hidden_dim, config_.vocab_size);
fprintf(stderr, "hidden shape: [%u, %u]\n", batch_size, config_.hidden_dim);
fprintf(stderr, "==============================\n");
```

### Step 2: Make the Change

Update the cuBLAS call in `cuda/src/transformer/qwen_transformer.cpp` line 557:

```cpp
cublasStatus_t status = cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,  // CHANGED: Transpose lm_head
    config_.hidden_dim,        // CHANGED: m = hidden_dim (896)
    batch_size,                // UNCHANGED: n = batch_size (1)
    config_.hidden_dim,        // UNCHANGED: k = hidden_dim (896)
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,  // UNCHANGED
    hidden_half, CUDA_R_16F, config_.hidden_dim,   // UNCHANGED
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,        // UNCHANGED
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

### Step 3: Add Debug Output AFTER the Change

```cpp
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "❌ cuBLAS GEMM failed with status: %d\n", status);
    return;
}

// Debug: Check logit values
float h_logits[10];
cudaMemcpy(h_logits, logits, 10 * sizeof(float), cudaMemcpyDeviceToHost);

fprintf(stderr, "✅ cuBLAS GEMM succeeded\n");
fprintf(stderr, "First 10 logits: ");
for (int i = 0; i < 10; i++) {
    fprintf(stderr, "%.2f ", h_logits[i]);
}
fprintf(stderr, "\n");

// Check problematic positions
float logit_8850, logit_44394, logit_137131;
cudaMemcpy(&logit_8850, logits + 8850, sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(&logit_44394, logits + 44394, sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(&logit_137131, logits + 137131, sizeof(float), cudaMemcpyDeviceToHost);

fprintf(stderr, "Logits at problematic positions:\n");
fprintf(stderr, "  [8850] = %.4f\n", logit_8850);
fprintf(stderr, "  [44394] = %.4f\n", logit_44394);
fprintf(stderr, "  [137131] = %.4f\n", logit_137131);
```

### Step 4: Remove Workarounds

If the fix works, remove these workarounds:
- Lines 575-595: The `-INFINITY` fill workaround
- Lines 597-627: The debug logging

### Step 5: Test

```bash
cd bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

---

## Expected Results

### If the fix works ✅

- No CUDA errors
- Logits in range -4 to +8 (not 14-15)
- No garbage at positions 8850, 44394, 137131
- Model generates different tokens each step
- Test passes with coherent output

### If the fix fails ❌

**Scenario 1: CUDA errors**
- Check the error message
- Verify matrix dimensions are correct
- Check for memory alignment issues

**Scenario 2: Wrong logit values**
- Compare with manual dot product
- Check tensor loading from GGUF
- Verify memory layout

**Scenario 3: Still garbage at specific positions**
- This suggests a different root cause
- Investigate tensor loading
- Check for memory corruption

---

## Rollback Plan

If the fix fails, revert to the current code:

```bash
git checkout cuda/src/transformer/qwen_transformer.cpp
```

Then investigate alternative theories:
1. Tensor loading issue
2. Memory corruption
3. cuBLAS bug with specific dimensions

---

## Success Criteria

- ✅ No CUDA errors
- ✅ All logits in expected range (-4 to +8)
- ✅ No garbage values
- ✅ Model generates diverse tokens
- ✅ Test passes
- ✅ Output is coherent

---

## Files to Modify

1. **`cuda/src/transformer/qwen_transformer.cpp`** - Lines 557-568

---

## Confidence Level

**High (85%)** - This fix matches llama.cpp exactly, and we understand why the previous attempt failed.

**Risk**: The previous fix attempt caused catastrophic failure, but we now know it was because we only changed the transpose flag without changing the dimensions.

---

**Status**: Ready for implementation  
**Next**: Make the change and test
