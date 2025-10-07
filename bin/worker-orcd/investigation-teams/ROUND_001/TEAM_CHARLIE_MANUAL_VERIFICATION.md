# Team Charlie: Mathematical Verification

**Team Mission**: Manually compute the correct answer and prove what cuBLAS should produce

**Team Expertise**: Linear algebra, numerical methods, ground truth computation

**YOU CAN CHANGE CODE TO EXTRACT DATA** - Implement verification tests, compute ground truth. Just revert after!

---

## Your Investigation Strategy

You are the mathematical verification team. Your job is to compute the **ground truth** logits manually and compare with cuBLAS output.

### Phase 1: Manual Dot Product Computation

**Task**: Compute logits[8850] by hand to establish ground truth

1. **Extract the data** (add instrumentation comments in `qwen_transformer.cpp`):
   ```cpp
   // [TEAM_CHARLIE] Data extraction for manual verification:
   // 1. Copy hidden[0:896] to host memory
   // 2. Copy lm_head row 8850 (elements [8850*151936 : 8850*151936+896]) to host
   // 3. Compute: logit_8850 = sum(hidden[i] * lm_head_row[i]) for i in [0,896)
   // 4. Compare with cuBLAS output logits[8850]
   ```

2. **Document the extraction plan**:
   - What GPU memory addresses to read?
   - How to interpret the data (FP16 → FP32)?
   - What is the expected range of values?

### Phase 2: Implement Verification Function

**Task**: ACTUALLY IMPLEMENT and RUN the verification function

Add this code to `qwen_transformer.cpp`:

```cpp
// [TEAM_CHARLIE] Verification function - IMPLEMENT THIS NOW!

float verify_logit_manual(int position) {
    // Step 1: Copy hidden state to host
    half h_hidden[896];
    cudaMemcpy(h_hidden, hidden_half, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // Step 2: Copy lm_head row at 'position' to host
    // Using row-major offset: position * 151936
    half h_lm_head_row[896];
    cudaMemcpy(h_lm_head_row, lm_head_half + position * 151936, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // Step 3: Manual dot product
    float manual_result = 0.0f;
    for (int i = 0; i < 896; i++) {
        manual_result += __half2float(h_hidden[i]) * __half2float(h_lm_head_row[i]);
    }
    
    // Step 4: Compare with cuBLAS
    float cublas_result;
    cudaMemcpy(&cublas_result, logits + position, sizeof(float), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[TEAM_CHARLIE] Position %d: Manual=%.6f, cuBLAS=%.6f, Diff=%.6f\n",
           position, manual_result, cublas_result, fabs(manual_result - cublas_result));
    
    return manual_result;
}

// Then call this in project_to_vocab:
if (first_call) {
    fprintf(stderr, "\n[TEAM_CHARLIE] === GROUND TRUTH VERIFICATION ===\n");
    verify_logit_manual(0);      // Should be correct
    verify_logit_manual(8850);   // Known garbage
    verify_logit_manual(44394);  // Known garbage
    verify_logit_manual(137131); // Known garbage
}

// REMEMBER: Revert this code after gathering data!
```

### Phase 3: Analyze Multiple Positions

**Task**: Create a verification plan for multiple positions

Design tests for:
1. Position 0 (should be correct - baseline)
2. Position 1000 (should be correct - random middle position)
3. Position 8850 (known to have garbage)
4. Position 44394 (known to have garbage)
5. Position 137131 (known to have garbage)

For each, document:
```cpp
// [TEAM_CHARLIE] Position ??? verification:
// Expected value range: [min, max]
// Current cuBLAS output: ???
// Manual computation offset: base + ??? * 151936
// Hypothesis: If manual != cuBLAS, confirms wrong memory access
```

### Phase 4: Matrix Dimension Verification

**Task**: Verify the matrix multiplication dimensions mathematically

Document the operation:

```cpp
// [TEAM_CHARLIE] Mathematical operation verification:
//
// Operation: logits = lm_head^T @ hidden
// 
// Shapes:
// - hidden: [896] (column vector)
// - lm_head in GGUF: [896, 151936] (row-major)
// - lm_head for computation: needs to be [896, 151936] to multiply with [896]
// - This means we need lm_head^T: [151936, 896]
// - Result: [151936, 896] @ [896, 1] = [151936, 1] ✓
//
// Current cuBLAS call:
// - op_A = CUBLAS_OP_N (no transpose)
// - A shape interpreted as: [151936, 896] (column-major interpretation)
// - But stored as: [896, 151936] (row-major in memory)
// - MISMATCH: cuBLAS thinks A is [151936, 896] column-major
//             but memory is [896, 151936] row-major
//
// Mathematical proof that this is wrong:
// ...
```

### Phase 5: Boundary Case Analysis

**Task**: Test extreme positions to understand the pattern

Propose tests for:
1. Position 0 (first element)
2. Position 895 (last element of first "row" if misinterpreted)
3. Position 896 (first element of second "row" if misinterpreted)
4. Position 151935 (last element)

Document expected vs actual for each:
```cpp
// [TEAM_CHARLIE] Boundary analysis:
// If cuBLAS is reading row-major as column-major incorrectly:
// - Position 0: reads correct data (lucky coincidence)
// - Position 895: reads correct data (end of first row)
// - Position 896: reads WRONG data (should be row 1, reads wrong location)
// - Pattern: Failures occur at positions where row/column mismatch is non-zero
```

### Phase 6: Create Test Data Set

**Task**: Design a synthetic test to isolate the issue

Propose a test with known values:
```cpp
// [TEAM_CHARLIE] Synthetic test proposal:
//
// 1. Create test lm_head with known pattern: lm_head[i][j] = i * 10000 + j
// 2. Create test hidden: hidden[i] = 1.0 for all i
// 3. Compute expected output: logit[i] = sum of row i = sum(i*10000 + j) for j in [0,896)
// 4. Run cuBLAS with these inputs
// 5. Compare output vs expected
// 6. Any position where output != expected proves the memory access is wrong
//
// Expected for position 8850:
// logit[8850] = sum(8850*10000 + j for j in [0,896)) = 8850*10000*896 + sum(0..895)
//             = 79,257,600,000 + 400,080 = 79,258,000,080
//
// If cuBLAS outputs something different, we know the exact nature of the error
```

---

## Deliverable: Investigation Results

Create file: `investigation-teams/TEAM_CHARLIE_RESULTS.md`

Include:
1. **Manual Computation Results** - Ground truth values for positions 0, 8850, 44394
2. **Comparison Table** - Manual vs cuBLAS for each position
3. **Error Pattern Analysis** - Which positions fail and why
4. **Mathematical Proof** - Why the current parameters are wrong
5. **Synthetic Test Design** - Proposed test with known values
6. **Verification Code** - Complete, ready-to-implement test functions

---

## Key Files to Analyze

1. `cuda/src/transformer/qwen_transformer.cpp` (lines 231-298)
2. Investigation docs showing current logit values:
   - `COMPLETE_INVESTIGATION_REPORT.md` (lines 99-106)

---

## Tools You'll Need

- Calculator or Python script for manual computations
- Understanding of FP16 representation
- Understanding of cuBLAS GEMM operation: `C = α(A)(B) + βC`

---

## Success Criteria

- [ ] Designed manual computation procedure for at least 3 positions
- [ ] Created detailed comments showing verification logic
- [ ] Proved mathematically that current parameters are incorrect
- [ ] Proposed synthetic test with known values
- [ ] Documented expected vs actual for boundary cases

**Remember**: You're establishing the **ground truth**. If manual computation differs from cuBLAS, you've proven the bug.
