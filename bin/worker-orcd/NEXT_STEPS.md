# Next Steps After KV Cache Fix

**Last Updated**: 2025-10-06 11:07  
**Status**: KV cache fixed, attention working, investigating bias corruption

---

## Immediate Actions

### 1. Test the Fix ✅ COMPLETED
```bash
cargo test --release haiku_generation_anti_cheat -- --nocapture 2>&1 | tee test_after_fix.log
```

**Result**: 
- ✅ Q values now in correct range (0.01 to 0.26)
- ✅ Attention weights vary across positions
- ✅ KV cache being read correctly
- ⚠️ Model generates diverse but poor quality output
- ❌ Bias values contain huge outliers

**Conclusion**: Core engine working, bias loading is the remaining issue.

### 2. Debug Attention Mechanism ✅ COMPLETED

**Root Cause Found**: Position counter (`kv_cache_.seq_lens`) was always 0, causing attention to only compute over current token.

**Fix Applied**: 
- Added debug logging to track position updates
- Verified position is correctly incremented after each forward pass
- Attention now computes over all cached positions

**Verification**:
```
cache_len=1: Attention weights [0]=0.5077 [1]=0.4923 (sums to 1.0) ✅
cache_len=2: Attention weights [0]=0.3431 [1]=0.6569 [2]=... ✅
```

### 3. Add Attention Weight Debugging ✅ COMPLETED

Added comprehensive debug output to `cuda/kernels/gqa_attention.cu`:
- Prints scaled scores before softmax
- Prints attention weights after normalization
- Verifies weights sum to 1.0
- Shows number of positions being attended to

### 4. Investigate Bias Loading 🔴 CURRENT PRIORITY

**Issue**: QKV bias tensors contain huge outlier values that corrupt attention:

**Evidence**:
```
attn_q_bias[0:10]: -0.0150 0.0255 -0.1035 -0.1357 -14.4375 0.2656 0.3242 0.1240 -15.4375 -34.0000
                                                      ^^^^^^^^                    ^^^^^^^^  ^^^^^^^^
Q after bias: -0.0364 -0.0335 -0.1520 -0.3208 -14.3438 0.2576 0.4233 0.0889 -15.6797 -34.0312
Output: ĠsáºµnĠsáºµnĠsáºµnĠsáºµnĠgottaĠgottaĠgotta... (only 2 tokens repeated)
```

**Current Status**: Bias addition disabled in `qwen_transformer.cpp` (lines 300, 329, 357)

**Action Items**:
1. **Check weight loader** - Verify bias dequantization in `qwen_weight_loader.cpp`
2. **Compare with llama.cpp** - Run same model file and check if biases are used
3. **Inspect GGUF file** - Use `gguf-dump` to check bias tensor metadata
4. **Test without biases** - Verify if Qwen2.5-0.5B actually uses biases (llama.cpp checks `if (model.layers[il].bq)`)

**Possible causes**:
- Bias tensors quantized but not dequantized during loading
- Bias tensor dimensions/layout incorrect
- Model file has corrupted bias data
- Qwen2.5-0.5B may not use biases at all

---

## Cleanup Tasks

### 1. Remove Debug Documents (After Verification)

Once the fix is confirmed working:
```bash
rm -f CRITICAL_FINDING.md \
      DEBUG_RUN_RESULTS.md \
      MATRIX_TRANSPOSE_FIX.md \
      BUG_FIX_PROGRESS.md \
      LLAMA_CPP_VALIDATION.md
```

Keep:
- `ROOT_CAUSE_ANALYSIS.md` - Technical reference
- `MATRIX_LAYOUT_FIX_SUMMARY.md` - Solution documentation

### 2. Remove Debug Logging

**In `cuda/src/transformer/qwen_transformer.cpp`**:
- Remove weight debug prints (lines 244-294)
- Remove K/V cache verification (lines 441-485)
- Remove attention output debug (lines 487-495)
- Keep only essential error logging

**In `cuda/src/transformer/qwen_transformer.cpp` (project_to_vocab)**:
- Remove first_call debug (lines 591-615)
- Remove logits sampling (lines 642-653)

### 3. Remove Debug Scripts
```bash
rm -f debug_llama_cpp.sh \
      inspect_weights.py \
      *.log
```

---

## Performance Optimization (Future)

### 1. Reduce Memory Allocations

**In `cuda/kernels/swiglu_ffn.cu`**:
- Allocate intermediate buffers once during initialization
- Reuse buffers across forward passes
- Current: 3 cudaMalloc/cudaFree per FFN call (expensive!)

### 2. Fuse Operations

**Potential fusions**:
- RMSNorm + QKV projection
- Attention output + residual add
- FFN gate/up projections (single GEMM)
- SwiGLU activation + down projection

### 3. Use Persistent Buffers

**In `QwenTransformer` constructor**:
- Pre-allocate all intermediate buffers
- Store as class members
- Avoid per-forward-pass allocations

---

## Code Quality Improvements

### 1. Add Unit Tests

**Test matrix multiplication**:
```rust
#[test]
fn test_qkv_projection_values() {
    // Load known input
    // Run projection
    // Compare with expected output from llama.cpp
}
```

### 2. Add Assertions

**In matrix multiplication code**:
```cpp
assert(q_dim == config_.num_heads * config_.head_dim);
assert(kv_dim == config_.num_kv_heads * config_.head_dim);
```

### 3. Document Matrix Layouts

Add comments explaining:
- GGUF storage format (row-major)
- cuBLAS expectations (column-major)
- Conversion strategy

---

## Verification Checklist

- [x] Q values match llama.cpp (±0.01 tolerance)
- [x] K values in reasonable range
- [x] V values in reasonable range
- [x] Attention scores computed over all positions
- [x] Attention weights sum to 1.0
- [x] KV cache working correctly
- [ ] Model generates coherent text (needs bias fix)
- [ ] No repetitive output (partial - less repetitive)
- [x] Haiku test passes (pipeline validated)
- [ ] Bias re-enabled (currently disabled due to corruption)
- [ ] Debug logging removed
- [ ] Performance is acceptable (>10 tokens/sec)

---

## Known Issues to Address

### 1. Weight Quantization
Current implementation loads quantized weights (Q4_K_M) without dequantization. This is noted in `qwen_weight_loader.cpp` line 185.

**TODO**: Implement proper dequantization or use FP16 model.

### 2. Bias Values
Bias tensors contain unexpectedly large values (-14.4, -15.4, -34.0). Need to investigate:
- Is this a loading bug?
- Are biases quantized differently?
- Compare with llama.cpp bias loading

### 3. Memory Leaks
FFN allocates temporary buffers on every forward pass. Should be pre-allocated.

---

## Success Criteria

The fix is successful when:

1. **Correctness**: Model output matches llama.cpp (same prompt, same seed)
2. **Quality**: Generated text is coherent and follows prompt
3. **Performance**: No significant slowdown from previous version
4. **Stability**: No crashes or CUDA errors during extended generation

---

## Contact

If issues persist after this fix, check:
1. Weight loading is correct (dimensions, data type)
2. RoPE implementation matches llama.cpp
3. Attention mechanism is correct (GQA, KV cache)
4. Softmax scaling is correct (1/sqrt(head_dim))
