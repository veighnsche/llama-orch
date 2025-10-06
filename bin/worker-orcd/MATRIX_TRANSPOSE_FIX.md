# Matrix Transpose Fix - 2025-10-06

**UPDATE 2025-10-06 10:49**: ⚠️ **This approach was incorrect**  
The issue was not about transposing, but about GGUF row-major vs cuBLAS column-major layout.  
See `MATRIX_LAYOUT_FIX_SUMMARY.md` for the correct fix.

---

## Root Cause Identified (INCORRECT ANALYSIS)

The bug was in the **cuBLAS matrix multiplication** for QKV projections. PyTorch/HuggingFace store weight matrices **transposed** compared to how we were using them.

## Changes Made

### Fixed Files
- `cuda/src/transformer/qwen_transformer.cpp`

### Specific Fixes

1. **Q Projection** (line ~230):
   - Changed from `CUBLAS_OP_N, CUBLAS_OP_N` to `CUBLAS_OP_T, CUBLAS_OP_N`
   - Changed `lda` from `q_dim` to `config_.hidden_dim`

2. **K Projection** (line ~280):
   - Changed from `CUBLAS_OP_N, CUBLAS_OP_N` to `CUBLAS_OP_T, CUBLAS_OP_N`
   - Changed `lda` from `kv_dim` to `config_.hidden_dim`

3. **V Projection** (line ~307):
   - Changed from `CUBLAS_OP_N, CUBLAS_OP_N` to `CUBLAS_OP_T, CUBLAS_OP_N`
   - Changed `lda` from `kv_dim` to `config_.hidden_dim`

4. **Attention Output Projection** (line ~462):
   - Changed from `CUBLAS_OP_N, CUBLAS_OP_N` to `CUBLAS_OP_T, CUBLAS_OP_N`
   - Changed `lda` to `config_.hidden_dim`

## Secondary Issue: Bias Values

**CRITICAL**: The bias tensors (`attn_q_bias`, `attn_k_bias`, `attn_v_bias`) contain **huge values** like `-14.4375`, `-15.4375`, `-34.0000`.

These values are **10-100x larger** than expected. Normal bias values should be < 0.1.

### Bias Debug Output
```
attn_q_bias[0:10]: -0.0150 0.0255 -0.1035 -0.1357 -14.4375 0.2656 0.3242 0.1240 -15.4375 -34.0000
```

### Temporary Workaround
Disabled bias addition in all QKV projections to test if this fixes output:
- Line 298: Commented out `cuda_bias_add` for Q
- Line 327: Commented out `cuda_bias_add` for K  
- Line 355: Commented out `cuda_bias_add` for V

### Q Values After Fix (without bias)
```
Q after bias[0:10]: 0.0340 0.0155 -0.0390 0.0190 1.4258 0.0690 0.1758 -0.0993 -0.3733 0.3232
```

These are **much more reasonable** (range ~0.01 to 1.8) compared to before (range ~-34 to 1.4).

## Next Steps

1. **Test output quality** with bias disabled
2. **Investigate bias loading** - why are bias values so large?
   - Check if bias tensors are quantized incorrectly
   - Check if bias tensor dimensions are wrong
   - Compare with llama.cpp bias values
3. **Re-enable bias** once root cause is found

## Files Modified
- `cuda/src/transformer/qwen_transformer.cpp` - Fixed matrix transposes, disabled bias temporarily
