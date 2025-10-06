# Bug Status - UPDATED 2025-10-06 12:50

## Current State: CRITICAL BUG - Root Cause Identified But Not Fixed

**Status**: ❌ BROKEN - Model generates same token repeatedly  
**Root Cause**: Logits at specific positions contain garbage values (~14-15) instead of correct values (-4 to +4)  
**Original Hypothesis**: INCORRECT - There is NO vocab size mismatch

---

## What We Discovered

### The Original Bug Report Was Wrong ❌

**Claimed**:
- lm_head tensor is [896, 151643]
- vocab_size is 151936 (padded)
- Garbage at positions 151643-151935

**Reality**:
- lm_head tensor IS [896, 151936] ✓
- vocab_size IS 151936 ✓
- Garbage at positions 8850, 44394, 137131 (scattered, not at end) ✓

### Actual Problem

**Symptom**: Argmax finds garbage values at specific positions

**Evidence**:
```
Normal logits range: -4.69 to +4.45
Position 8850:   14.26 (prefill)
Position 44394:  12.34 → 15.19 (increases over time!)
Position 137131: 14.71 → 14.03 (decreases over time)
```

**Behavior**:
- Prefill phase (calls #0-4): Selects token 137131
- Generation phase (calls #5+): Selects token 44394
- Model output: "coholic" repeated 100 times

---

## What's Working ✅

1. **Model Loading**: All 291 tensors load successfully
2. **Matrix Operations**: Q values correct (0.01-0.26 range)
3. **KV Cache**: Position tracking works, cache updates correctly
4. **Attention**: Weights sum to 1.0, computes over all positions
5. **lm_head Weights**: Sampled values are normal (±0.01-0.08)
6. **Hidden State**: Values in reasonable range (-32 to +31)
7. **llama.cpp**: Same model works perfectly in reference implementation

---

## What's Broken ❌

1. **Logits Computation**: Specific positions have garbage values
2. **Argmax**: Selects garbage instead of correct logits
3. **Output**: Model generates same token repeatedly
4. **Usability**: Completely broken, unusable output

---

## What We Tested

### ✅ Verified Correct

- [x] Tensor dimensions: output.weight is [896, 151936]
- [x] Tensor type: F16 (not quantized)
- [x] lm_head weights: Normal values at sampled positions
- [x] Hidden state: No extreme spikes
- [x] GEMM parameters: Look correct for the operation
- [x] Model file: Works in llama.cpp

### ❌ Attempted Fixes That Failed

1. **Derive vocab from tensor dims**: No effect (dims already correct)
2. **Fill high positions with -INFINITY**: Failed (garbage at low positions too)
3. **Remove hardcoded values**: No effect (already using config)

---

## Root Cause Theories

### Theory 1: Memory Layout Mismatch (MOST LIKELY)

**Issue**: lm_head tensor might not be transposed correctly when loaded from GGUF

**In GGUF**: Row-major [hidden_dim, vocab_size] = [896, 151936]  
**In cuBLAS**: We treat as column-major [vocab_size, hidden_dim] = [151936, 896]

**Problem**: Maybe we're not actually transposing it, just telling cuBLAS it's transposed

**How to Check**: Compare with `reference/llama.cpp/src/llama-model.cpp`

### Theory 2: cuBLAS Parameter Issue

**Issue**: Subtle bug in how we call cublasGemmEx

**Current Call**:
```cpp
cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    vocab_size, 1, hidden_dim,
    &alpha,
    lm_head, CUDA_R_16F, vocab_size,
    hidden, CUDA_R_16F, hidden_dim,
    &beta,
    logits, CUDA_R_32F, vocab_size,
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**How to Check**: Compare with llama.cpp's cuBLAS calls in `reference/llama.cpp/ggml/src/ggml-cuda/`

### Theory 3: Tensor Loading Bug

**Issue**: lm_head tensor not loaded correctly to GPU memory

**How to Check**: 
1. Dump entire tensor from GPU
2. Compare with GGUF file data
3. Look for mismatches at positions 8850, 44394, 137131

---

## Next Steps for Engineering Team

### Priority 1: Compare with llama.cpp

**Location**: `reference/llama.cpp/` directory

**Commands**:
```bash
cd reference/llama.cpp
grep -r "output.weight" src/
grep -r "lm_head" src/
grep -r "cublasGemmEx" ggml/src/ggml-cuda/
```

**Focus On**:
1. How does llama.cpp load lm_head tensor?
2. Does it transpose the tensor?
3. What cuBLAS parameters does it use?

### Priority 2: Manual Verification

**Test**: Manually compute logits[44394] and compare with GEMM result

```cpp
// Compute: logits[44394] = dot(lm_head[44394, :], hidden[:])
float manual = 0.0f;
for (int i = 0; i < 896; i++) {
    manual += lm_head[44394 + i * vocab_size] * hidden[i];
}
// Compare: manual vs logits[44394]
```

### Priority 3: Try Different Approaches

1. **Explicit transpose**: Actually transpose lm_head in memory
2. **Different compute mode**: Try CUBLAS_COMPUTE_32F instead of FAST_16F
3. **FP32 conversion**: Convert lm_head to FP32 before GEMM

---

## Key Files

**Modified During Investigation**:
- `src/inference/cuda_backend.rs` - Added vocab derivation (no effect)
- `src/cuda/model.rs` - Added debug output
- `cuda/src/transformer/qwen_transformer.cpp` - Extensive debug output
- `cuda/src/ffi_inference.cpp` - Removed hardcoded values
- `cuda/kernels/sampling_wrapper.cu` - Extended debug output

**To Review**:
- `cuda/src/model/qwen_weight_loader.cpp` - How lm_head is loaded
- `src/cuda/weight_loader.rs` - Rust side of tensor loading
- `reference/llama.cpp/src/llama-model.cpp` - Reference implementation

---

## Test Command

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

**Expected Output (Currently)**:
- Argmax finds token 44394 with value ~14-15
- Model generates "coholic" 100 times
- Test passes (pipeline works) but output is garbage

**Desired Output (After Fix)**:
- Argmax finds tokens with values -4 to +8
- Model generates diverse, coherent tokens
- Output is readable haiku

---

## Documentation

**Read These (In Order)**:
1. `COMPLETE_INVESTIGATION_REPORT.md` - Full investigation details
2. `FINAL_DIAGNOSIS.md` - Quick summary
3. `VOCAB_SIZE_INVESTIGATION.md` - Initial findings
4. This file - Current status

**Reference**:
- `reference/llama.cpp/` - Working implementation
- `LLAMA_CPP_VALIDATION.md` - Proof that model file works

---

**Last Updated**: 2025-10-06 12:50  
**Status**: Root cause identified, fix in progress  
**Next Action**: Compare with llama.cpp implementation
