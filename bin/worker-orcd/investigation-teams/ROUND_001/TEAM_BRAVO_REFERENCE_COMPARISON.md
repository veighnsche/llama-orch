# Team Bravo: Reference Implementation Comparison

**Team Mission**: Deep dive into llama.cpp to understand exactly how they solve this problem

**Team Expertise**: Comparative analysis, reverse engineering, llama.cpp internals

**YOU CAN CHANGE CODE TO EXTRACT DATA** - Add logging, run tests, compare with llama.cpp. Just revert after!

---

## Your Investigation Strategy

You are the reference implementation specialists. Your job is to **precisely** understand how llama.cpp handles the same model and operation.

### Phase 1: Locate llama.cpp's lm_head Operation

**Task**: Find where llama.cpp computes the final logits

1. **Find the GEMM call**
   ```bash
   cd reference/llama.cpp
   grep -r "output.weight" src/
   grep -r "cublasGemm" ggml/src/ggml-cuda/
   ```

2. **Document in comments** (add to `qwen_transformer.cpp`):
   ```cpp
   // [TEAM_BRAVO] llama.cpp equivalent operation found at:
   // File: ???
   // Line: ???
   // Function: ???
   ```

### Phase 2: Extract Exact Parameters

**Task**: Document llama.cpp's cuBLAS parameters line-by-line

In `reference/llama.cpp/ggml/src/ggml-cuda/`, find the GEMM call and document:

1. **Transpose operations**: `CUBLAS_OP_?` for each matrix
2. **Dimensions**: m, n, k values
3. **Leading dimensions**: lda, ldb, ldc values
4. **Data types**: Input/output types
5. **Compute mode**: CUBLAS_COMPUTE_* flag

Create a comparison table in your results:

```
Parameter    | llama.cpp | Our Code | Match?
-------------|-----------|----------|-------
op_A         | ???       | N        | ?
op_B         | ???       | N        | ?
m            | ???       | 151936   | ?
n            | ???       | 1        | ?
k            | ???       | 896      | ?
lda          | ???       | 151936   | ?
ldb          | ???       | 896      | ?
ldc          | ???       | 151936   | ?
```

### Phase 3: Understand llama.cpp's Tensor Loading

**Task**: How does llama.cpp load lm_head from GGUF?

1. **Find tensor loading code**
   ```bash
   cd reference/llama.cpp
   grep -r "ggml_set_name.*output" src/
   grep -r "load.*tensor" src/
   ```

2. **Document**:
   - Does llama.cpp transpose lm_head after loading?
   - What dimensions does llama.cpp report for output.weight?
   - How does llama.cpp handle row-major vs column-major?

3. **Add comments in `src/cuda/weight_loader.rs`**:
   ```rust
   // [TEAM_BRAVO] llama.cpp tensor loading:
   // - Loads as dimensions: ???
   // - Performs transpose: YES/NO
   // - Stores in memory as: ???
   ```

### Phase 4: Test llama.cpp with Same Inputs

**Task**: Run llama.cpp with debugging to capture actual values

1. **Build llama.cpp with debug output** (if not already done)
2. **Run with same model**:
   ```bash
   cd reference/llama.cpp/build
   ./bin/main -m /path/to/model.gguf -p "test" -n 1 --verbose
   ```

3. **Capture**:
   - What are the first 10 logits values?
   - What token does it select?
   - Any warnings about matrix dimensions?

4. **Document in your results**: Comparison of llama.cpp vs our output

### Phase 5: Reverse Engineer the Difference

**Task**: Identify the ONE key difference that matters

Given that llama.cpp works and ours doesn't, there must be ONE critical difference. Find it.

Possibilities to investigate:
1. Different transpose flags
2. Different dimension order
3. Different leading dimensions
4. Explicit tensor transpose before GEMM
5. Different memory allocation strategy

Add detailed comments in `qwen_transformer.cpp` explaining the difference:
```cpp
// [TEAM_BRAVO] CRITICAL DIFFERENCE:
// llama.cpp does: ???
// We do: ???
// Impact: ???
```

### Phase 6: Trace llama.cpp's GGML → cuBLAS Translation

**Task**: Understand how GGML operations map to cuBLAS

llama.cpp uses GGML (a higher-level abstraction) which then calls cuBLAS. Trace:

1. GGML operation definition
2. How GGML determines transpose flags
3. How GGML calculates leading dimensions
4. The final cuBLAS call

Document the translation logic.

---

## Deliverable: Investigation Results

Create file: `investigation-teams/TEAM_BRAVO_RESULTS.md`

Include:
1. **Parameter Comparison Table** - Side-by-side llama.cpp vs ours
2. **Key Difference Identification** - The ONE critical difference
3. **Code Flow Diagram** - How llama.cpp goes from GGUF → GEMM
4. **Tensor Layout Proof** - Evidence of how llama.cpp stores lm_head
5. **Proposed Fix** - Exact parameter changes to match llama.cpp

---

## Key Files to Analyze

**Our Code**:
1. `cuda/src/transformer/qwen_transformer.cpp` (lines 275-293)
2. `src/cuda/weight_loader.rs`

**llama.cpp Code**:
1. `reference/llama.cpp/src/llama-model.cpp` - Model loading
2. `reference/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` - CUDA kernels
3. `reference/llama.cpp/ggml/include/ggml.h` - GGML API

**Investigation Docs**:
1. `LLAMA_CPP_MATRIX_ANALYSIS.md` (previous analysis, may be incomplete)

---

## Success Criteria

- [ ] Found exact llama.cpp GEMM call with all parameters documented
- [ ] Created parameter comparison table
- [ ] Identified the ONE critical difference
- [ ] Verified our understanding by testing llama.cpp
- [ ] Proposed specific parameter changes to match llama.cpp

**Remember**: llama.cpp works with the exact same model. The answer is in their code.
