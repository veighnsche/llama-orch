# Team Bravo - Investigation Results (UPDATED)

**Mission**: Compare our implementation with the `llama.cpp` reference to find the root cause of the repetitive token bug.

**Date**: 2025-10-06  
**Status**: 🔴 INVESTIGATION INCONCLUSIVE - DEEPER ISSUE FOUND

---

## Executive Summary

**CRITICAL FINDING**: The simple fix proposed in the initial Team Bravo results **DOES NOT WORK**. 

ATTEMPT #4 (documented in `qwen_transformer.cpp` lines 282-289) already tried the exact llama.cpp parameters (`CUBLAS_OP_T`, `lda=896`) but **FAILED** - the model still generated repetitive tokens (token 68396 instead of 44394).

This means the issue is **NOT** simply a cuBLAS parameter mismatch. There is a deeper problem.

---

## Investigation Timeline

### Phase 1: Initial Hypothesis (INCORRECT ❌)

**Theory**: Our code uses wrong cuBLAS parameters compared to llama.cpp.

**Evidence**:
- llama.cpp uses `CUBLAS_OP_T, lda=896`
- Our code uses `CUBLAS_OP_N, lda=151936`

**Proposed Fix**: Change to match llama.cpp parameters.

### Phase 2: Testing the Fix (FAILED ❌)

**ATTEMPT #4 Results** (from code comments):
```
Changed: CUBLAS_OP_T, CUBLAS_OP_N, lda=896
Result: STILL BROKEN - Different repetitive token
  - Generates token 68396 repeatedly (was 44394)
  - Max logit: 13.64 (still abnormally high)
  - Manual verification FAILS (cuBLAS != manual)
Conclusion: Copying llama.cpp's cuBLAS params alone doesn't fix it
```

**This is the smoking gun**: Simply matching llama.cpp's cuBLAS parameters is insufficient.

### Phase 3: Deeper Analysis (IN PROGRESS 🔍)

**New Questions**:
1. Why does ATTEMPT #4 fail if the parameters match llama.cpp?
2. What else is different between our implementation and llama.cpp?
3. Is there an issue with how we load the tensor from GGUF?
4. Is there a memory alignment or stride issue?

---

## Parameter Comparison Table

| Parameter | `llama.cpp` (`ggml-cuda.cu`) | Our Code (Original) | Our Code (ATTEMPT #4) | Match llama.cpp? |
|---|---|---|---|---|
| `op_A` (lm_head) | `CUBLAS_OP_T` | `CUBLAS_OP_N` | `CUBLAS_OP_T` | ✅ (in ATTEMPT #4) |
| `op_B` (hidden_state) | `CUBLAS_OP_N` | `CUBLAS_OP_N` | `CUBLAS_OP_N` | ✅ Yes |
| `m` | `row_diff` (vocab_size) | `config_.vocab_size` | `config_.vocab_size` | ✅ Yes |
| `n` | `src1_ncols` (batch_size) | `batch_size` | `batch_size` | ✅ Yes |
| `k` | `ne10` (hidden_dim) | `config_.hidden_dim` | `config_.hidden_dim` | ✅ Yes |
| `lda` | `ne00` (hidden_dim=896) | `config_.vocab_size` (151936) | `config_.hidden_dim` (896) | ✅ (in ATTEMPT #4) |
| `ldb` | `ne10` (hidden_dim) | `config_.hidden_dim` | `config_.hidden_dim` | ✅ Yes |
| `ldc` | `ldc` (vocab_size) | `config_.vocab_size` | `config_.vocab_size` | ✅ Yes |

**Result**: ATTEMPT #4 matched ALL parameters, yet still failed! ❌

---

## Tensor Dimensions Analysis

### GGUF File Storage

From test output:
```
🔍 [Rust] output.weight dimensions: [896, 151936]
```

The tensor is stored in GGUF as `[hidden_dim, vocab_size]` = `[896, 151936]` in **row-major** format.

### llama.cpp Tensor Creation

From `llama-model.cpp:2365`:
```cpp
output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, ...);
```

llama.cpp also expects `{n_embd, n_vocab}` = `{896, 151936}`.

**Conclusion**: Both implementations expect the same tensor dimensions. ✅

---

## Memory Layout Mathematics

### Row-Major Storage (GGUF)

Tensor shape: `[896, 151936]`  
Element at `(i, j)`: `offset = i * 151936 + j`

```
Memory layout:
[row 0: 151936 elements] [row 1: 151936 elements] ... [row 895: 151936 elements]
```

### Our Original Approach (CUBLAS_OP_N, lda=151936)

cuBLAS interprets as **column-major** `[151936, 896]` with `lda=151936`:
- To read column `j`: stride through memory at offsets `j, j+151936, j+2*151936, ...`
- This reads: `lm_head[0][j], lm_head[1][j], lm_head[2][j], ...`
- **This is CORRECT** for computing `logits[j] = dot(hidden, lm_head[:, j])`

### llama.cpp Approach (CUBLAS_OP_T, lda=896)

cuBLAS interprets as **column-major** `[896, 151936]` with `lda=896`, then transposes:
- Before transpose: reads rows of 896 elements
- After transpose: effectively reads columns of the original matrix
- **This should also be CORRECT**

### Why Do Both Approaches Seem Mathematically Equivalent?

**They are!** Both should compute the same result:
- Original: `C = A @ B` where A is interpreted as `[151936, 896]`
- llama.cpp: `C = A^T @ B` where A is interpreted as `[896, 151936]`

The transpose in llama.cpp's approach should cancel out the different interpretation.

---

## The Mystery: Why Does ATTEMPT #4 Fail?

### Possible Explanations

1. **Memory Alignment Issue**
   - cuBLAS might have alignment requirements
   - `lda=896` might not be properly aligned
   - Our memory allocation might differ from llama.cpp's

2. **Tensor Loading Difference**
   - We might be loading the tensor incorrectly from GGUF
   - Byte order, padding, or stride issues
   - llama.cpp might do some preprocessing we're missing

3. **Hidden State Issue**
   - The bug might not be in lm_head projection at all
   - Team Alpha found that cuBLAS output matches manual computation
   - The hidden state itself might be wrong (upstream bug)

4. **Quantization/Dequantization**
   - Our FP16 handling might differ from llama.cpp
   - Precision loss in different code paths

5. **Context/State Corruption**
   - KV cache corruption
   - Attention mechanism bug
   - Layer norm bug

---

## Team Alpha's Conflicting Evidence

Team Alpha's verification (lines 292-356 in `qwen_transformer.cpp`):

```
✅ MANUAL DOT PRODUCT TEST - cuBLAS is CORRECT!
Position 8850:   manual=14.264349  cuBLAS=14.264330  diff=0.000019 ✅
Position 44394:  manual=12.341835  cuBLAS=12.341816  diff=0.000019 ✅
Position 137131: manual=14.712263  cuBLAS=14.712248  diff=0.000015 ✅
```

**Team Alpha's Conclusion**: cuBLAS is computing correctly, the bug is upstream (attention/hidden state).

**But**: This verification was done with the ORIGINAL parameters (`CUBLAS_OP_N, lda=151936`), not with llama.cpp's parameters!

---

## Reconciling the Evidence

### Scenario A: Original Parameters Are Correct

- Team Alpha's manual verification shows original parameters work
- The bug is upstream (attention, layer norm, etc.)
- llama.cpp's different parameters are just an alternative correct formulation
- **Action**: Focus investigation on attention mechanism (Team Alpha's recommendation)

### Scenario B: Both Parameter Sets Are Wrong

- Original parameters: accidentally correct for some positions, wrong for others
- llama.cpp parameters: wrong in a different way
- There's a third correct formulation we haven't tried
- **Action**: Derive correct parameters from first principles (Team Echo's mission)

### Scenario C: Tensor Loading Is Wrong

- Both parameter sets would work if tensor was loaded correctly
- We're loading the tensor with wrong dimensions or stride
- llama.cpp loads it differently
- **Action**: Deep dive into GGUF tensor loading (investigate `weight_loader.rs`)

---

## Recommended Next Steps

### Priority 1: Verify Team Alpha's Manual Computation

Re-run Team Alpha's verification test with **ATTEMPT #4 parameters** (`CUBLAS_OP_T, lda=896`):
- Does manual computation still match cuBLAS?
- If NO: The parameters are wrong
- If YES: The bug is definitely upstream

### Priority 2: Compare Tensor Loading with llama.cpp

Instrument both implementations to dump:
- First 100 bytes of lm_head after loading
- Memory addresses and strides
- Any preprocessing steps

### Priority 3: Test with llama.cpp Directly

Run llama.cpp with the same model and prompt:
- Does it also generate "coholic"?
- If YES: Model file issue, not code bug
- If NO: Implementation difference to investigate

### Priority 4: Investigate Upstream Components

If lm_head projection is truly correct (as Team Alpha claims):
- Attention mechanism (softmax, KV cache)
- Layer normalization (RMSNorm)
- FFN (feed-forward network)
- Residual connections

---

## Conclusion

**The initial Team Bravo hypothesis was INCORRECT.** Simply matching llama.cpp's cuBLAS parameters does not fix the bug.

**Current Status**:
- ❓ Root cause unknown
- ✅ llama.cpp parameters identified and tested (ATTEMPT #4)
- ❌ llama.cpp parameters do not fix the issue
- ⚠️ Conflict between Team Alpha (cuBLAS correct) and Team Bravo (cuBLAS wrong)

**Recommendation**: This investigation needs to be escalated. Either:
1. Team Alpha's conclusion is correct → investigate upstream components
2. Team Alpha's verification has a flaw → re-verify with different parameters
3. There's a subtle issue neither team has identified → bring in Team Charlie (manual verification) and Team Echo (first principles)

---

## Files Referenced

- `cuda/src/transformer/qwen_transformer.cpp` (lines 249-474)
- `src/cuda/weight_loader.rs` (lines 549-598)
- `reference/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` (lines 1297-1303)
- `reference/llama.cpp/src/llama-model.cpp` (line 2365)
- `investigation-teams/TEAM_ALPHA_FINAL_CONCLUSION.md`
- `investigation-teams/INVESTIGATION_COMPLETE_SUMMARY.md`

---

**Status**: Investigation continues. Team Bravo recommends cross-team sync to reconcile findings.
