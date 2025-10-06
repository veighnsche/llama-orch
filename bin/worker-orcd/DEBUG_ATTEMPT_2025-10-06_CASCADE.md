# Debug Attempt Report - 2025-10-06 14:37 UTC

**Engineer**: Cascade (AI Assistant)  
**Session**: Second debugging session  
**Status**: **FAILED** - Made problem worse, reverted changes

---

## Summary

I attempted to fix the repetitive token generation bug by modifying the `cublasGemmEx` call in `project_to_vocab`. Both attempts catastrophically failed, producing worse results than the original bug. The code has been reverted to its original state.

---

## Initial Understanding

### What I Read
1. **INVESTIGATION_INDEX.md** - Claims bug was "resolved" by correcting cuBLAS parameters
2. **LLAMA_CPP_MATRIX_ANALYSIS.md** - States llama.cpp uses `CUBLAS_OP_T` for lm_head
3. **HANDOFF_TO_NEXT_TEAM.md** - Says to focus on cuBLAS GEMM call
4. **FINAL_SUMMARY_AND_ROOT_CAUSE.md** - Mentions build system issues (later disproven)

### The Original Bug
- Model generates same token repeatedly (e.g., "coholic" 100 times)
- Logits at specific positions (8850, 44394, 137131) have abnormal values (~14-15)
- Normal logits range from -4 to +4
- Garbage values change over time (not static memory corruption)

### The Matrix Operation
```
Operation: logits[vocab_size] = hidden[hidden_dim] @ lm_head[hidden_dim, vocab_size]
Dimensions: logits[151936] = hidden[896] @ lm_head[896, 151936]
```

**Storage Format**:
- lm_head is stored in GGUF as [896, 151936] in **row-major** format
- cuBLAS expects **column-major** format
- Element at (i, j) in row-major [896, 151936]: `address = base + i * 151936 + j`

---

## Attempt #1: Change to CUBLAS_OP_T with Wrong Dimensions

### What I Changed
```cpp
// BEFORE:
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    config_.vocab_size,  // m = 151936
    batch_size,          // n = 1
    config_.hidden_dim,  // k = 896
    ...);

// AFTER (WRONG):
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    config_.hidden_dim,   // m = 896  ❌ WRONG
    batch_size,           // n = 1
    config_.vocab_size,   // k = 151936  ❌ WRONG
    ...);
```

### Results
```
First 10 logits: 0.00 0.00 0.00 0.00 0.00 0.00 -144805391855483444084057871573057536.00 0.00 0.00 0.00
Null pointer in embedding lookup
RMSNorm kernel launch failed: operation not supported on global/shared address space
RoPE ex kernel launch failed: operation not supported on global/shared address space
```

**Analysis**: Completely catastrophic. Logits contained values like `-1.4×10^35`. This caused complete model crash with null pointer errors and CUDA errors. The wrong dimensions meant cuBLAS was accessing memory far outside the allocated buffers.

---

## Attempt #2: Change to CUBLAS_OP_T with Correct Dimensions

### What I Changed
```cpp
// AFTER (STILL WRONG):
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    config_.vocab_size,   // m = 151936  ✓ Correct
    batch_size,           // n = 1       ✓ Correct
    config_.hidden_dim,   // k = 896     ✓ Correct
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,  // lda = 151936
    hidden_half, CUDA_R_16F, config_.hidden_dim,    // ldb = 896
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,        // ldc = 151936
    ...);
```

### Results
```
First 10 logits: 0.00 0.00 0.00 0.00 3098249111849719037952.00 0.00 165574579289622754598179856889937920.00 0.00 2915051686207542001664.00 0.00
Null pointer in embedding lookup
RMSNorm kernel launch failed: operation not supported on global/shared address space
```

**Analysis**: Still catastrophic. Logits contained values like `3.1×10^21` and `1.7×10^35`. Even though m, n, k dimensions were correct, the transpose operation with these lda/ldb/ldc values produced garbage.

---

## Why Both Attempts Failed

### The Core Issue: Memory Layout Mismatch

When you have a row-major matrix [R, C] and cuBLAS interprets it as column-major:
- Row-major `[896, 151936]` stores element (i,j) at: `base + i * 151936 + j`
- Column-major interpretation sees this as `[151936, 896]` with element (i,j) at: `base + i + j * 151936`
- These are **NOT equivalent** unless you carefully manage the transpose and leading dimensions

### What I Got Wrong

**For `CUBLAS_OP_T` on matrix A:**
- A is stored with dimensions and leading dimension based on its **physical storage**
- When using `CUBLAS_OP_T`, cuBLAS expects `lda` to be the leading dimension of A **before** transpose
- If A is stored row-major [896, 151936], and we tell cuBLAS it's column-major [151936, 896] with lda=151936:
  - cuBLAS will try to access elements assuming stride of 151936 between columns
  - But our data has stride of 151936 between **rows**, not columns
  - This causes it to read from completely wrong memory locations

### The Mathematical Disconnect

For the operation `C = A^T @ B` with cuBLAS:
- A^T should transform [k, m] → [m, k]  
- B is [k, n]
- C is [m, n]

For our case:
- We want: logits[151936] = hidden[896] @ lm_head[896, 151936]
- This is: C[151936, 1] = B[896, 1] @ A[896, 151936]
- But cuBLAS does `C = A @ B`, not `C = B @ A`
- So we need to compute: C = lm_head^T @ hidden
- Where lm_head is [896, 151936], so lm_head^T is [151936, 896]
- Then: C[151936, 1] = lm_head^T[151936, 896] @ hidden[896, 1]

**But the storage format makes this complex!**

---

## What the Original Code Does

```cpp
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    config_.vocab_size,   // m = 151936
    batch_size,           // n = 1
    config_.hidden_dim,   // k = 896
    &alpha,
    lm_head_half, CUDA_R_16F, config_.vocab_size,  // lda = 151936
    hidden_half, CUDA_R_16F, config_.hidden_dim,    // ldb = 896
    &beta,
    logits, CUDA_R_32F, config_.vocab_size,        // ldc = 151936
    ...);
```

This computes: `C[m, n] = A[m, k] @ B[k, n]`
- C[151936, 1] = A[151936, 896] @ B[896, 1]

But our lm_head is stored as [896, 151936] row-major!

**The disconnect**: cuBLAS is being told A is [151936, 896] with lda=151936, but our data is actually [896, 151936] row-major.

---

## The Real Problem (Hypothesis)

I believe the issue is that the original code is **treating row-major data as if it were column-major** without proper handling.

### What Might Be Happening

1. lm_head is loaded from GGUF as [896, 151936] row-major
2. cuBLAS is told it's [151936, 896] column-major with lda=151936
3. This causes cuBLAS to access wrong memory locations for some elements
4. **Most** elements happen to work out correctly by accident
5. **Some** elements (like 8850, 44394, 137131) access garbage memory

### Why Only Some Positions Are Wrong

The row-major to column-major reinterpretation might work for some elements but not others, depending on memory alignment and access patterns.

For element at position `i` in the output logits:
- Correct access: should compute dot product of hidden[896] with lm_head[i][0:896]
- With row-major storage: these elements are at addresses `base + i * 151936 + j` for j in 0..896
- With wrong interpretation: cuBLAS might read from addresses `base + i + j * 151936`
- For some values of `i`, these addresses might overlap with valid data
- For other values of `i` (like 8850, 44394), they read garbage

---

## What Should Be Tried Next

### Option 1: Explicit Transpose in Memory

Before calling cuBLAS, explicitly transpose lm_head in GPU memory:
```cpp
// Transpose lm_head from [896, 151936] to [151936, 896]
transpose_kernel<<<blocks, threads>>>(
    lm_head_half,          // input: [896, 151936] row-major
    lm_head_transposed,    // output: [151936, 896] row-major
    896, 151936
);

// Then use with CUBLAS_OP_N
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, ...);
```

**Pros**: Makes the memory layout explicit and clear  
**Cons**: Requires extra memory and computation

### Option 2: Use cuBLAS-XT or Custom Kernel

Use cuBLAS-XT which has better support for row-major matrices, or write a custom CUDA kernel that directly computes the dot products without relying on cuBLAS.

### Option 3: Carefully Match llama.cpp's Exact Implementation

I tried to do this but failed. A more careful analysis is needed:
1. Dump the exact memory layout of lm_head in our implementation
2. Dump the exact memory layout in llama.cpp
3. Instrument both cuBLAS calls with detailed logging
4. Compare element-by-element to understand the difference

### Option 4: Manual Dot Product Test

Compute a few logits manually and compare with cuBLAS output:
```cpp
// Manually compute logits[8850]
float manual_logit = 0.0f;
for (int j = 0; j < 896; j++) {
    manual_logit += hidden[j] * lm_head[8850 * 151936 + j];  // row-major access
}
fprintf(stderr, "Manual logit[8850]: %.4f vs cuBLAS: %.4f\n", manual_logit, logits[8850]);
```

This would definitively show if cuBLAS is accessing the right memory addresses.

---

## Recommendations for Next Engineer

1. **DO NOT** try random transpose flags. It will make things worse.
2. **DO NOT** trust the previous investigation's claim that the bug was "resolved". The documentation appears to be out of sync.
3. **DO** start with careful analysis of memory layout
4. **DO** add manual dot product computation to verify what the correct answer should be
5. **DO** instrument the code heavily before making changes
6. **Consider** that this might be a fundamental architectural issue requiring more than a simple parameter change

---

## Files Modified (Then Reverted)

- `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
  - Line 275-293: Modified `cublasGemmEx` call (REVERTED)

All changes have been reverted to the original state.

---

**Status**: Investigation continues. The bug remains unsolved.
