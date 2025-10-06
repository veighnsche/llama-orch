# Team Echo: First Principles Analysis

**Team Mission**: Build understanding from cuBLAS documentation and first principles

**Team Expertise**: cuBLAS API, CUDA programming, matrix operations, documentation analysis

**YOU CAN CHANGE CODE TO TEST HYPOTHESES** - Try different parameters, gather data, prove your theory. Just revert after!

---

## Your Investigation Strategy

You are the first principles team. Ignore all previous assumptions. Start from the cuBLAS documentation and build up correct understanding.

### Phase 1: Study cuBLAS Documentation

**Task**: Read and document cuBLAS GEMM behavior

Study the official cuBLAS documentation for `cublasGemmEx`:
```
C = α(op(A))(op(B)) + βC
```

Document in comments:

```cpp
// [TEAM_ECHO] cuBLAS GEMM Specification:
//
// From NVIDIA cuBLAS documentation:
// - C has dimensions m × n
// - op(A) has dimensions m × k  
// - op(B) has dimensions k × n
// - op(X) = X if CUBLAS_OP_N, X^T if CUBLAS_OP_T
//
// Leading dimension (ld*):
// - lda: Leading dimension of A (stride between columns in column-major)
// - ldb: Leading dimension of B
// - ldc: Leading dimension of C
//
// Column-major storage:
// - Element A[i,j] is at address: A_base + i + j * lda
// - For matrix [R rows, C cols], lda >= R
```

### Phase 2: Define Our Operation Mathematically

**Task**: Write out exactly what operation we need

```cpp
// [TEAM_ECHO] Required Operation:
//
// Goal: Compute final logits from hidden state
//
// Mathematical operation:
//   logits[vocab_size] = hidden[hidden_dim] · lm_head[hidden_dim, vocab_size]
//
// Dimensions:
//   logits: [151936]  (output)
//   hidden: [896]     (input activation)
//   lm_head: [896, 151936]  (weight matrix from GGUF)
//
// Matrix form:
//   logits^T[1, 151936] = hidden^T[1, 896] @ lm_head[896, 151936]
//
// Or equivalently:
//   logits[151936, 1] = lm_head^T[151936, 896] @ hidden[896, 1]
```

### Phase 3: Analyze Storage Format

**Task**: Document how data is actually stored

```cpp
// [TEAM_ECHO] Storage Format Analysis:
//
// GGUF Format (from file):
// - Tensors are stored in ROW-MAJOR format
// - lm_head dimensions in GGUF: [896, 151936]
// - Element at (i, j) is at offset: i * 151936 + j
// - Total size: 896 * 151936 * 2 bytes (FP16)
//
// GPU Memory (after loading):
// - Data is copied as-is from GGUF (still row-major)
// - Base pointer: lm_head_half
// - Element lm_head[i][j] is at: lm_head_half + (i * 151936 + j)
//
// cuBLAS Expectation:
// - cuBLAS expects COLUMN-MAJOR format
// - For column-major [R, C], element (i, j) is at: base + i + j * R
// - This is incompatible with row-major storage!
```

### Phase 4: Calculate Correct Parameters

**Task**: Derive the correct cuBLAS call from first principles

```cpp
// [TEAM_ECHO] Deriving Correct Parameters:
//
// Option 1: Transpose A to match cuBLAS expectations
// We have: lm_head stored as row-major [896, 151936]
// cuBLAS sees it as: column-major [151936, 896] (WRONG)
// Fix: Use CUBLAS_OP_T to transpose it back
//
// With CUBLAS_OP_T on A:
//   op(A) = A^T
//   Original A (as cuBLAS sees it): [151936, 896] column-major
//   After transpose: [896, 151936]
//   This matches what we need!
//
// Correct call:
//   cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//       vocab_size,  // m = 151936 (output rows)
//       1,           // n = 1 (output cols)
//       hidden_dim,  // k = 896 (shared dimension)
//       &alpha,
//       lm_head, CUDA_R_16F, vocab_size,  // A: lda = ?
//       hidden, CUDA_R_16F, hidden_dim,   // B: ldb = 896
//       &beta,
//       logits, CUDA_R_32F, vocab_size);  // C: ldc = 151936
//
// Question: What should lda be?
// Answer: lda is the leading dimension of A BEFORE transpose
//         A before transpose is [151936, 896] column-major (how cuBLAS sees it)
//         Leading dimension = 151936 ✓
```

### Phase 5: Verify Against Current Implementation

**Task**: Compare derived parameters with actual code

```cpp
// [TEAM_ECHO] Current vs Correct Comparison:
//
// Current Implementation:
//   op_A: CUBLAS_OP_N  ❌ Should be CUBLAS_OP_T
//   m: vocab_size ✓
//   n: batch_size ✓  
//   k: hidden_dim ✓
//   lda: vocab_size ✓
//   ldb: hidden_dim ✓
//   ldc: vocab_size ✓
//
// Diagnosis:
//   Only the transpose flag is wrong!
//   Everything else is already correct.
//
// Predicted Fix:
//   Change CUBLAS_OP_N to CUBLAS_OP_T for first operand
//   Keep all other parameters the same
```

### Phase 6: Test Your Hypothesis

**Task**: Implement a test to PROVE your theory

```cpp
// [TEAM_ECHO] Hypothesis test - try the fix you derived!
// 
// Based on first principles, you believe the fix is to change CUBLAS_OP_N to CUBLAS_OP_T
// and possibly adjust lda. TEST IT:

if (first_call) {
    fprintf(stderr, "\n[TEAM_ECHO] === TESTING DERIVED FIX ===\n");
    
    // Save original logits for comparison
    float original_logits[10];
    int test_positions[] = {0, 100, 8850, 10000, 44394, 50000, 137131, 140000, 150000, 151935};
    
    // Try your derived parameters (example - adjust based on your derivation):
    cublasStatus_t test_status = cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,  // Your derived transpose flags
        config_.vocab_size,         // Your derived m
        batch_size,                 // Your derived n
        config_.hidden_dim,         // Your derived k
        &alpha,
        lm_head_half, CUDA_R_16F, 896,  // Your derived lda (THIS IS KEY!)
        hidden_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        logits, CUDA_R_32F, config_.vocab_size,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // Check results
    for (int i = 0; i < 10; i++) {
        float test_logit;
        cudaMemcpy(&test_logit, logits + test_positions[i], sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[TEAM_ECHO] Test logits[%6d] = %.6f\n", test_positions[i], test_logit);
    }
    
    fprintf(stderr, "[TEAM_ECHO] Does this look better? Document in RESULTS.md\n");
}

// CRITICAL: REVERT THIS TEST CODE before committing!

### Phase 7: Explain Why Current Code Fails

**Task**: Explain the failure mechanism based on your test results

```cpp
// [TEAM_ECHO] Failure Mechanism Explanation:
//
// Current code uses CUBLAS_OP_N (no transpose):
//   cuBLAS thinks A is [151936, 896] column-major
//   To access element A[i, j], it computes: base + i + j * 151936
//
// But actual storage is row-major [896, 151936]:
//   Element at logical position (i, j) is at: base + i * 151936 + j
//
// Mismatch:
//   cuBLAS access: base + i + j * 151936
//   Actual location: base + i * 151936 + j
//   These are equal only when: i + j * 151936 = i * 151936 + j
//   Simplifying: j * (151936 - 1) = i * (151936 - 1)
//   This holds when i = j (diagonal elements only!)
//
// For other elements:
//   cuBLAS reads from wrong memory location
//   Reads garbage data
//   Produces garbage logits
//
// Why only SOME positions fail:
//   Some accesses happen to hit valid memory by chance
//   Others hit uninitialized or wrong data
//   Pattern depends on memory layout
```

---

## Deliverable

Create: `investigation-teams/TEAM_ECHO_RESULTS.md`

Include:
1. **Mathematical Derivation** - Complete first-principles derivation
2. **Parameter Calculation** - Step-by-step calculation of correct values
3. **Failure Mechanism** - Detailed explanation of why current code fails
4. **Proposed Fix** - Exact change needed (likely just transpose flag)
5. **Verification Strategy** - How to verify the fix is correct

---

## Key Resources

- NVIDIA cuBLAS documentation
- cuBLAS samples
- Matrix multiplication theory
- Row-major vs column-major storage

---

## Success Criteria

- [ ] Documented cuBLAS GEMM behavior from official docs
- [ ] Derived correct parameters from first principles
- [ ] Explained failure mechanism mathematically
- [ ] Proposed specific fix with justification
- [ ] No assumptions, only documented facts
