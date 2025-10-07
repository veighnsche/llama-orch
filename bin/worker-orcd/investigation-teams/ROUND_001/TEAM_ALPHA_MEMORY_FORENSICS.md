# Team Alpha: Memory Layout Forensics

**Team Mission**: Understand the exact memory layout and access patterns to identify the mismatch

**Team Expertise**: Memory architecture, CUDA memory models, row-major vs column-major layouts

**YOU CAN CHANGE CODE TO EXTRACT DATA** - Add logging, run tests, gather evidence. Just revert changes after!

---

## Your Investigation Strategy

You are the memory forensics specialists. Your job is to trace **exactly** how data flows through memory from GGUF file to cuBLAS computation.

### Phase 1: Map the Memory Layout

**Task**: Document the exact memory layout at each stage

1. **GGUF File Format**
   - Add comments in `src/cuda/weight_loader.rs` explaining how lm_head is stored in the file
   - Document: Is it [896, 151936] or [151936, 896]?
   - Document: What is the stride between elements?

2. **GPU Memory After Load**
   - Add comments in `cuda/src/transformer/qwen_transformer.cpp` at line ~243 where lm_head is accessed
   - Document: How is lm_head laid out in GPU memory?
   - Document: What is `lm_head_half[i]` actually accessing?

3. **cuBLAS Interpretation**
   - Add comments explaining how cuBLAS interprets the memory with `CUBLAS_OP_N` and `lda=vocab_size`
   - Document: For element at position (i,j), what memory address does cuBLAS read?

### Phase 2: Trace Specific Failing Positions

**Task**: Trace why positions 8850, 44394, and 137131 have garbage

For position 8850 (known garbage position):

1. Calculate the **intended** memory addresses that should be read
2. Calculate the **actual** memory addresses cuBLAS reads with current parameters
3. Document the difference

Add comments in `qwen_transformer.cpp` showing:
```cpp
// [TEAM_ALPHA] For logit[8850]:
// - Should read: hidden[0:896] dot lm_head[8850][0:896]
// - Expected addresses: base + 8850*151936 + j for j in [0,896)
// - Actual cuBLAS access with lda=151936: ???
// - Mismatch: ???
```

### Phase 3: Memory Access Pattern Analysis

**Task**: Create a detailed memory access diagram

In your investigation results document, create:

1. **Visual diagram** showing:
   - How GGUF stores lm_head (row-major [896, 151936])
   - How cuBLAS thinks it's stored (column-major [?, ?])
   - Where the mismatch occurs

2. **Access pattern table** for first 5 positions:

```
Position | Intended Addresses | Actual cuBLAS Addresses | Match?
0        | [0 + j]           | [???]                   | ?
1        | [151936 + j]      | [???]                   | ?
8850     | [8850*151936 + j] | [???]                   | ?
```

### Phase 4: Calculate Leading Dimension

**Task**: Determine the correct `lda` value

Given:
- lm_head stored as [896, 151936] row-major
- cuBLAS treating it as [?, ?] column-major
- Current lda = 151936

Calculate:
1. What **should** lda be for correct access?
2. What do the current parameters tell cuBLAS about the matrix shape?
3. Create a formula: `correct_lda = f(rows, cols, storage_format)`

### Phase 5: Verification Strategy

**IMPLEMENT AND RUN** this test:

```cpp
// [TEAM_ALPHA] Verification test - IMPLEMENT THIS!
if (first_call) {
    fprintf(stderr, "\n[TEAM_ALPHA] === MANUAL VERIFICATION TEST ===\n");
    
    // 1. Copy hidden state to host
    half h_hidden[896];
    cudaMemcpy(h_hidden, hidden_half, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // 2. Copy lm_head row 8850 (row-major access)
    half h_lm_head_row[896];
    cudaMemcpy(h_lm_head_row, lm_head_half + 8850*151936, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // 3. Manual dot product
    float manual_logit = 0.0f;
    for (int i = 0; i < 896; i++) {
        manual_logit += __half2float(h_hidden[i]) * __half2float(h_lm_head_row[i]);
    }
    
    // 4. Get cuBLAS result
    float cublas_logit;
    cudaMemcpy(&cublas_logit, logits + 8850, sizeof(float), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[TEAM_ALPHA] Position 8850:\n");
    fprintf(stderr, "  Manual computation: %.6f\n", manual_logit);
    fprintf(stderr, "  cuBLAS output:      %.6f\n", cublas_logit);
    fprintf(stderr, "  Difference:         %.6f\n", fabs(manual_logit - cublas_logit));
    
    if (fabs(manual_logit - cublas_logit) > 0.01) {
        fprintf(stderr, "  ⚠️  MISMATCH CONFIRMED!\n");
    }
}

// REMEMBER TO REVERT THIS CODE AFTER GATHERING DATA!
```

**RUN THE TEST** and document the results in your RESULTS.md file.

---

## Deliverable: Investigation Results

Create file: `investigation-teams/TEAM_ALPHA_RESULTS.md`

Include:
1. **Memory Layout Diagram** - Visual representation of the mismatch
2. **Access Pattern Analysis** - Detailed table of address calculations
3. **Root Cause Hypothesis** - Your theory on why positions 8850, 44394, 137131 fail
4. **Proposed Fix** - What parameter(s) need to change and why
5. **Verification Plan** - How to test your hypothesis without breaking things

---

## Key Files to Analyze

1. `src/cuda/weight_loader.rs` - How tensors are loaded from GGUF
2. `cuda/src/transformer/qwen_transformer.cpp` - The cuBLAS call (lines 275-293)
3. `src/cuda/model.rs` - Tensor dimension logging
4. Investigation docs: `COMPLETE_INVESTIGATION_REPORT.md` (lines 196-224)

---

## Success Criteria

- [ ] Documented exact memory layout at each stage
- [ ] Calculated memory addresses for failing positions
- [ ] Identified the specific parameter causing wrong access
- [ ] Proposed a fix with mathematical justification
- [ ] No code changes, only detailed comments

**Remember**: You're looking for **where** the data is stored vs **where** cuBLAS is reading from. The mismatch is the key.
