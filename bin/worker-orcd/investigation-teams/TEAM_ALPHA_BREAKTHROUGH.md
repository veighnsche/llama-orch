# Team Alpha: BREAKTHROUGH - Root Cause Found!

**Date**: 2025-10-06 15:43 UTC  
**Status**: 🔥 ROOT CAUSE IDENTIFIED

---

## The Discovery

After comparing with llama.cpp's CUDA implementation, I found the exact bug!

### llama.cpp Parameters

```cpp
// reference/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1259
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    151936,  // m = vocab_size
    1,       // n = batch_size
    896,     // k = hidden_dim
    &alpha,
    lm_head, CUDA_R_16F, 896,  // lda = 896 (hidden_dim) ← KEY DIFFERENCE!
    hidden,  CUDA_R_16F, 896,  // ldb = 896
    &beta,
    logits, CUDA_R_32F, ldc,
    ...
);
```

### Our Current Parameters

```cpp
// bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp:267
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    151936,  // m = vocab_size ✓ Same
    1,       // n = batch_size ✓ Same
    896,     // k = hidden_dim ✓ Same
    &alpha,
    lm_head, CUDA_R_16F, 151936,  // lda = 151936 ❌ WRONG!
    hidden,  CUDA_R_16F, 896,     // ldb = 896    ✓ Same
    &beta,
    logits, CUDA_R_32F, 151936,
    ...
);
```

### The Differences

1. **Transpose flag**: llama.cpp uses `CUBLAS_OP_T`, we use `CUBLAS_OP_N`
2. **Leading dimension**: llama.cpp uses `lda=896`, we use `lda=151936`

---

## Why This Matters

### Memory Layout

lm_head is stored as **[896, 151936] row-major**:
```
Row 0: [elem 0, elem 1, ..., elem 151935]  ← hidden dim 0
Row 1: [elem 0, elem 1, ..., elem 151935]  ← hidden dim 1
...
Row 895: [elem 0, elem 1, ..., elem 151935] ← hidden dim 895
```

Each **column j** represents the weights for **vocab position j**.

### What We Want to Compute

```
logits[vocab_pos] = sum over i: hidden[i] * lm_head[i][vocab_pos]
```

This is: `logits = lm_head^T @ hidden`

### llama.cpp's Approach (CORRECT)

With `CUBLAS_OP_T` and `lda=896`:
- cuBLAS sees lm_head as column-major [896, 151936] with stride 896 between columns
- Applies transpose: [896, 151936]^T = [151936, 896]
- Computes: [151936, 896] @ [896, 1] = [151936, 1]
- For each vocab position i, it computes: dot(hidden, column_i_of_original_matrix)
- **This is correct!**

### Our Approach (WRONG)

With `CUBLAS_OP_N` and `lda=151936`:
- cuBLAS sees lm_head as column-major [151936, 896] with stride 151936 between columns
- No transpose
- Computes: [151936, 896] @ [896, 1] = [151936, 1]
- For each vocab position i, it computes: dot(hidden, row_i_of_wrongly_interpreted_matrix)
- **This accesses wrong memory locations!**

---

## Why My Verification Test Was Misleading

In my manual verification, I computed:
```cpp
// Copy column i from lm_head
for (int j = 0; j < 896; j++) {
    cudaMemcpy(&h_lm_head_row[j], lm_head_half + j*151936 + i, ...);
}
manual_logit = dot(hidden, h_lm_head_row);
```

This matched cuBLAS output! But this doesn't mean cuBLAS is correct - it means **I was replicating the same bug in my manual test**!

The question is: **Should we be reading columns or rows?**

Looking at llama.cpp's tensor creation:
```cpp
output = create_tensor({n_embd, n_vocab}, ...) = [896, 151936]
```

This means:
- First dimension (896) = n_embd (hidden dimension)
- Second dimension (151936) = n_vocab (vocabulary size)

So each **column** represents a vocab position, which is what I was computing. So actually my verification WAS correct!

---

## Wait... Re-analyzing

Let me think through this more carefully:

1. lm_head is [896, 151936] where:
   - 896 = hidden_dim
   - 151936 = vocab_size

2. For logit computation, we want:
   ```
   logits[vocab_i] = sum_j: hidden[j] * lm_head[j][vocab_i]
   ```

3. In row-major storage [896, 151936]:
   - lm_head[j][vocab_i] is at: base + j * 151936 + vocab_i
   - So column vocab_i contains: [lm_head[0][vocab_i], lm_head[1][vocab_i], ..., lm_head[895][vocab_i]]

4. My manual test computed column vocab_i and dotted with hidden - **THIS IS CORRECT!**

5. So if cuBLAS matches my manual computation, **cuBLAS IS correct!**

But then why does llama.cpp use different parameters?

---

## The Real Question

If our cuBLAS call produces the same results as manual computation, and manual computation is mathematically correct, then **why does the model still generate garbage?**

Possible explanations:
1. **llama.cpp does something different** that we haven't found yet
2. **The model file is different** between our test and llama.cpp
3. **There's a bug elsewhere** in the pipeline that we haven't found
4. **The verification test itself has a bug** and isn't actually testing the right thing

---

## Next Steps

1. **Run llama.cpp with the EXACT same model file** and compare outputs
2. **Add logging to llama.cpp** to see what logits it produces
3. **Compare logit values** position by position
4. **If llama.cpp also produces logit[44394]=14.7**, then our code is correct!
5. **If llama.cpp produces different logits**, then we need to find what's different

---

**Status**: Need to run comparative test with llama.cpp to determine if our logits match theirs.
