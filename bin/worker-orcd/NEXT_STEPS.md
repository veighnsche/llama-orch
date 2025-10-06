# Next Steps After Vocab Size Bug Discovery

**Last Updated**: 2025-10-06 12:18  
**Status**: Core engine working, but vocab size mismatch causes garbage token selection

---

## Immediate Actions

### 1. Fix Vocab Size Mismatch 🔴 CRITICAL

**Problem**: lm_head tensor is [896, 151643] but vocab_size is 151936. Argmax finds garbage beyond actual vocab.

**Options**:

**Option A: Mirror llama.cpp — Use Actual Vocab End-to-End** (RECOMMENDED)
```rust
// In Rust weight loader, read actual lm_head dimensions from GGUF
let lm_head_info = metadata.get_tensor_info("output.weight")?;
let actual_vocab = lm_head_info.dimensions[1]; // Should be 151643
// Pass actual_vocab to C++ instead of padded vocab_size
```

Concrete changes:
- Rust: In `src/inference/cuda_backend.rs`, derive `actual_vocab` from tokenizer or `"output.weight"` dims; pass to `ffi::cuda_inference_init()`.
- C++: In `cuda/src/transformer/qwen_transformer.cpp::project_to_vocab()`, remove hardcoded `151643`; use `config_.vocab_size` for GEMM `m` and `ldc` and loops.
- C++: In `cuda/src/ffi_inference.cpp::cuda_inference_generate_token()`, remove hardcoded `actual_vocab_size = 151643`; call `cuda_sample_token(logits, config.vocab_size, ...)`.
- CUDA: `cuda/kernels/sampling_wrapper.cu` already honors `vocab_size`; no change needed.
- Sanity: At init, assert LM head leading dim equals `config_.vocab_size` and log mismatch.

**Option B: CUDA Kernel to Initialize Padding**
```cuda
__global__ void fill_padding_kernel(float* logits, int actual_vocab, int total_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + actual_vocab;
    if (idx < total_vocab) {
        logits[idx] = -INFINITY;
    }
}
```

**Option C: Modify Argmax to Limit Search**
```cuda
// In argmax_kernel, change loop:
for (int i = 0; i < actual_vocab; i++) {  // Not vocab_size!
    if (logits[i] > max_val) {
        max_val = logits[i];
        max_idx = i;
    }
}
```

**Status**: Implement Option A to match llama.cpp behavior

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

### 2. Verify Bias Handling ✅ COMPLETED

**Issue**: QKV bias tensors appeared to contain huge outlier values

**Resolution**: Checked llama.cpp reference implementation - confirmed Qwen2.5 architecture does NOT use bias tensors. The code correctly skips bias addition.

**Status**: No action needed - bias handling is correct

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
- [x] Bias handling correct (Qwen2.5 doesn't use biases)
- [x] Hidden states change with different inputs
- [x] Logits change with different inputs
- [ ] **Vocab size mismatch fixed** ← BLOCKING
- [ ] Model generates diverse tokens
- [ ] Model generates coherent text
- [ ] No repetitive output
- [x] Haiku test passes (pipeline validated)
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

## llama.cpp Reference Summary

- **n_vocab** is derived from tokenizer: `vocab.n_tokens()`.
- **lm_head** allocated as `{n_embd, n_vocab}` (no padding exposed).
- **logits** buffers sized to `n_vocab` and copied as `n_tokens * n_vocab`.
- **sampling** iterates `token_id in [0, n_vocab)` only.

Action: adopt the same constraints throughout our pipeline (Option A above).

## Contact

If issues persist after this fix, check:
1. Weight loading is correct (dimensions, data type)
2. RoPE implementation matches llama.cpp
3. Attention mechanism is correct (GQA, KV cache)
4. Softmax scaling is correct (1/sqrt(head_dim))
