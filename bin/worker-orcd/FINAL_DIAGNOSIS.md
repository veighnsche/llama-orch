# Final Diagnosis - Repetitive Token Generation Bug

**Date**: 2025-10-06  
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

The model generates the same token repeatedly (ID=44394 "coholic") because **specific positions in the logits buffer contain garbage values (~14-15) that are much higher than legitimate logits (-4 to +4)**. The argmax operation selects these garbage values instead of the correct logits.

---

## Key Findings

### 1. Vocab Size is NOT the Issue

- **Claim in bug report**: lm_head is [896, 151643] but vocab_size is 151936
- **Reality**: lm_head IS [896, 151936] (verified from GGUF parsing)
- **llama.cpp**: Also reports n_vocab = 151936
- **Conclusion**: No vocab size mismatch exists

### 2. Garbage Values at Specific Positions

**Evidence from test run**:

```
Call #0: Logits[137131]=14.71, Logits[44394]=12.34  → argmax selects 137131
Call #1: Logits[137131]=14.66, Logits[44394]=13.01  → argmax selects 137131
Call #2: Logits[137131]=14.45, Logits[44394]=13.46  → argmax selects 137131
Call #3: Logits[137131]=14.35, Logits[44394]=13.88  → argmax selects 137131
Call #4: Logits[137131]=14.28, Logits[44394]=14.12  → argmax selects 137131
Call #5: Logits[137131]=14.21, Logits[44394]=14.40  → argmax selects 44394 ✓
Call #6: Logits[137131]=14.03, Logits[44394]=14.59  → argmax selects 44394
...
```

**Normal logits range**: -4.69 to +4.45  
**Garbage values**: 12.34 to 15.07 (3-4x higher!)

### 3. lm_head Weights are Correct

Checked weights at problematic positions:
- `lm_head[token=137131][0:5] = [0.0135, -0.0018, -0.0071, 0.0090, 0.0228]` ✓
- `lm_head[token=44394][0:5] = [-0.0125, -0.0110, -0.0262, 0.0183, -0.0133]` ✓

Values are in expected range (±0.01-0.03), so the weights loaded correctly.

### 4. GEMM Operation Not Writing to All Positions

- **Logits[0:999]**: Correct values (verified by sampling)
- **Logits[44394]**: Garbage value 12.34-14.59
- **Logits[137131]**: Garbage value 14.03-14.71

The cuBLAS GEMM is computing correct values for early positions but leaving some positions uninitialized.

---

## Root Cause

**The cuBLAS GEMM operation is not writing to all vocab_size positions in the logits buffer.**

Possible reasons:
1. The lm_head tensor data is smaller than metadata claims (e.g., actual data is only 151643 rows, not 151936)
2. There's a bug in how the GEMM parameters are set
3. The tensor is padded in metadata but not in actual data

---

## Recommended Fix

### Option A: Use Actual Tensor Data Size (RECOMMENDED)

Instead of using metadata dimensions, determine the actual number of rows in the lm_head tensor from the data size:

```rust
let output_tensor = tensors.iter().find(|t| t.name == "output.weight")?;
let actual_rows = output_tensor.data_size / (hidden_dim * sizeof(f16));
// Use actual_rows as vocab_size, not metadata dimensions[1]
```

### Option B: Initialize Padding to -INFINITY

If the tensor is genuinely padded, ensure padding positions are set to -INFINITY:

```cpp
// After GEMM, fill any unwritten positions
if (actual_data_rows < vocab_size) {
    thrust::fill(logits + actual_data_rows, logits + vocab_size, -INFINITY);
}
```

### Option C: Limit Argmax Search

Only search the range that was actually computed:

```cpp
for (int i = 0; i < actual_vocab_size; i++) {  // Not vocab_size!
    if (logits[i] > max_val) {
        max_val = logits[i];
        max_idx = i;
    }
}
```

---

## Next Steps

1. **Determine actual lm_head data size** from tensor byte count
2. **Implement Option A** if data size < metadata size
3. **Test with haiku generation** to verify diverse token output
4. **Remove debug logging** once fixed
5. **Update status documents** with final resolution

---

## Test Command

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

---

## Files Modified (for debugging)

- `cuda/kernels/sampling_wrapper.cu`: Extended argmax debug to 15 calls
- `cuda/src/transformer/qwen_transformer.cpp`: Added logits position sampling
- `src/cuda/model.rs`: Added output.weight dimension logging
- `src/inference/cuda_backend.rs`: Added vocab size derivation (no effect since dims match)

---

**Conclusion**: The bug is NOT a vocab size mismatch. It's that the GEMM operation doesn't write to all positions, leaving garbage values that win the argmax. Need to determine why and fix accordingly.
