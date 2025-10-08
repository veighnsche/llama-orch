# CHECKPOINT 1: RMSNorm Output (Llama-2)

**Created by:** TEAM-008  
**Phase:** 3.2 - Normalization  
**Component:** RMSNorm (Root Mean Square Normalization)  
**File:** `src/layers/rms_norm.rs`  
**Imports:** ndarray only (NO worker-crates)  
**Tolerance:** 1e-5  
**Critical Level:** 🔴 CRITICAL - Errors propagate to all 32 layers  
**Prerequisites:** ✅ Checkpoint 0 passed (HTTP server + model loading)

---

## Purpose

Validate that RMSNorm computation is correct. This is the **first model component** in Llama-2 and affects all subsequent operations.

**Why This Matters:**
- RMSNorm is used 64 times in Llama-2 7B (2x per block × 32 blocks)
- Errors here compound through all layers
- This validates your tensor operations work correctly
- First test of Llama-2 architecture implementation
- **Different from GPT-2's LayerNorm** - simpler but critical to get right

**Key Difference from GPT-2:**
- GPT-2 uses LayerNorm: `(x - mean) / sqrt(variance + eps) * weight + bias`
- Llama-2 uses RMSNorm: `x / sqrt(mean(x²) + eps) * weight` (NO mean subtraction, NO bias)

---

## When to Check

- **Location:** After first RMSNorm in first transformer block (layer 0, attention norm)
- **Input:** Token embedding output (no position embeddings in Llama-2)
- **Timing:** Week 2, Day 1-2 (after GGUF loading works)
- **Before:** Implementing attention (Checkpoint 2)

---

## Expected Behavior

RMSNorm MUST:
1. Compute mean of squared values across hidden dimension (dim=4096)
2. Compute RMS: `sqrt(mean(x²) + eps)`
3. Normalize: `x / rms`
4. Apply learned scale parameter (weight only, NO bias)
5. Use epsilon = 1e-5

**Mathematical Formula:**
```
RMS = sqrt(mean(x²) + eps)
output = (x / RMS) * weight
```

---

## Implementation File

**File:** `src/layers/rms_norm.rs`

**Imports:**
```rust
use ndarray::{Array1, Array2, Axis};
// NO worker-crates imports - pure implementation
```

**Structure:**
```rust
// Created by: TEAM-008

pub struct RMSNorm {
    weight: Array1<f32>,  // Scale parameter [hidden_size] = [4096]
    eps: f32,             // Epsilon (1e-5)
}

impl RMSNorm {
    pub fn new(weight: Array1<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // x shape: [seq_len, hidden_size] = [seq_len, 4096]
        
        // 1. Compute mean of squared values across last dimension
        let x_squared = x.mapv(|val| val * val);
        let mean_squared = x_squared.mean_axis(Axis(1)).unwrap();
        
        // 2. Compute RMS
        let rms = mean_squared.mapv(|val| (val + self.eps).sqrt());
        
        // 3. Normalize: x / rms (broadcast rms across hidden_size)
        let normalized = x / &rms.insert_axis(Axis(1));
        
        // 4. Apply scale (weight)
        let output = &normalized * &self.weight;
        
        output
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ NO worker-crates imports
- ✅ NO bias term (unlike LayerNorm)
- ✅ NO mean subtraction (unlike LayerNorm)
- ✅ Simpler than LayerNorm but critical to get right

---

## Test Input

```
Prompt: "Hello"
Tokens: [1, 15043]  # [BOS, "Hello"]
Model: Llama-2 7B Q8_0
Temperature: 0 (greedy)
```

**Model Configuration:**
- Vocab size: 32000
- Hidden size: 4096
- Num layers: 32
- Num heads: 32
- Head dim: 128

---

## Validation Checklist

### ✓ Pre-Check
- [ ] Model loaded successfully from GGUF
- [ ] Weights loaded (check `blk.0.attn_norm.weight` exists)
- [ ] Weight shape: `[4096]`
- [ ] Input embeddings shape: `[seq_len=2, hidden_size=4096]`
- [ ] Epsilon = 1e-5

### ✓ Output Validation
- [ ] **Shape Check:** Output shape matches input `[2, 4096]`
- [ ] **RMS Check:** RMS values should be positive (> 0)
- [ ] **Normalized Check:** After normalization (before weight), values should have RMS ≈ 1
- [ ] **Value Range:** Output values typically in range [-5, 5]
- [ ] **No NaN/Inf:** Check for numerical stability

### ✓ Reference Comparison
- [ ] **llama.cpp:** Extract checkpoint using Team 006's tool
- [ ] **Tolerance:** Max absolute difference < 1e-5
- [ ] **Mean Absolute Error:** < 1e-6
- [ ] **First 10 values:** Print and compare manually

---

## Reference Locations

### llama.cpp
**File:** `llama.cpp`
**Function:** `llm_build_norm()` with `LLM_NORM_RMS`
**Line:** Search for `ggml_rms_norm`

**Key Code:**
```c
// RMS normalization
struct ggml_tensor * cur = ggml_rms_norm(ctx, inpL, norm_eps);
// Apply weight
cur = ggml_mul(ctx, cur, model.layers[il].attn_norm);
```

### Candle
**File:** `candle-transformers/src/models/llama.rs`
**Struct:** `RmsNorm`
**Method:** `forward()`

**Key Code:**
```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + self.eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
}
```

### Mistral.rs
**File:** `mistralrs-core/src/layers.rs`
**Struct:** `RmsNorm`
**Method:** `forward()`

---

## Common Failures

### 1. Using LayerNorm Instead of RMSNorm
**Symptom:** Values completely wrong, large errors (> 1.0)
**Cause:** Implemented LayerNorm (mean subtraction) instead of RMSNorm
**Fix:** Remove mean subtraction, only use RMS normalization

### 2. Including Bias Term
**Symptom:** Shape mismatch or wrong values
**Cause:** Added bias term (Llama-2 has NO bias)
**Fix:** Remove bias, only use weight

### 3. Wrong Variance Calculation
**Symptom:** Values slightly off (error ~1e-3)
**Cause:** Used unbiased variance (divide by N-1) or wrong formula
**Fix:** Use `mean(x²)` not `variance(x)`

### 4. Wrong Epsilon Placement
**Symptom:** Numerical instability, NaN values
**Cause:** Added epsilon after sqrt instead of before
**Fix:** `sqrt(mean(x²) + eps)` not `sqrt(mean(x²)) + eps`

### 5. Broadcasting Error
**Symptom:** Shape mismatch during normalization
**Cause:** RMS shape doesn't broadcast correctly
**Fix:** Ensure RMS has shape `[seq_len, 1]` to broadcast over `[seq_len, 4096]`

---

## Debug Commands

### Extract Reference Checkpoint
```bash
cd bin/llorch-cpud/tools/checkpoint-extractor
./build/llorch-checkpoint-extractor \
  /.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  "Hello" \
  /tmp/llama2_checkpoints

# Check output
cat /tmp/llama2_checkpoints/checkpoint_01_rms_norm.txt
```

### Run Your Implementation
```bash
cd bin/llorch-cpud
LLORCH_CHECKPOINT=1 cargo run --release -- \
  --model /.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  --prompt "Hello" \
  --temp 0

# Output should print checkpoint 1 values
```

### Compare Values
```python
import numpy as np

# Load reference
ref = np.loadtxt('/tmp/llama2_checkpoints/checkpoint_01_rms_norm.txt')

# Load your output
yours = np.loadtxt('/tmp/your_checkpoint_01.txt')

# Compare
diff = np.abs(ref - yours)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
print(f"First 10 ref:  {ref[:10]}")
print(f"First 10 yours: {yours[:10]}")

# Should be < 1e-5
assert diff.max() < 1e-5, f"FAILED: max diff {diff.max()}"
print("✅ PASSED")
```

---

## Success Criteria

### Minimum
- ✅ Output shape correct: `[2, 4096]`
- ✅ No NaN or Inf values
- ✅ Values in reasonable range [-10, 10]

### Recommended
- ✅ Max absolute difference from llama.cpp < 1e-5
- ✅ Mean absolute error < 1e-6
- ✅ First 10 values match reference

### Production
- ✅ All validation checks pass
- ✅ Numerical stability verified
- ✅ Unit tests written
- ✅ Proof bundle generated

---

## Next Steps

### If Checkpoint 1 Passes ✅
1. **Celebrate!** First Llama-2 component working
2. Move to Checkpoint 2 (QKV Projection)
3. Document any implementation notes
4. Commit code with proof bundle

### If Checkpoint 1 Fails ❌
1. Check common failures above
2. Print intermediate values (mean_squared, rms, normalized)
3. Compare with reference implementation line-by-line
4. Verify weight loading (print first 10 weight values)
5. Check epsilon value (should be 1e-5)
6. Verify no bias term added

---

## Proof Bundle Output

**Location:** `bin/llorch-cpud/.proof_bundle/checkpoint/run_<timestamp>/`

**Files to generate:**
- `checkpoint_01_input.ndjson` - Input tensor
- `checkpoint_01_output.ndjson` - Output tensor
- `checkpoint_01_metadata.json` - Config, shapes, timing
- `checkpoint_01_comparison.md` - Comparison with reference
- `seeds.json` - RNG seeds (if any)

**Header (PB-1012):**
```
# Generated by llorch-cpud checkpoint validation
# Run ID: <run_id>
# Timestamp: <iso8601>
# Component: RMSNorm (Checkpoint 1)
```

---

## Mathematical Verification

**Input:** `x` with shape `[seq_len, hidden_size]`

**Step 1:** Compute mean of squares
```
mean_squared[i] = (1/hidden_size) * Σ(x[i,j]²) for j in 0..hidden_size
```

**Step 2:** Compute RMS
```
rms[i] = sqrt(mean_squared[i] + eps)
```

**Step 3:** Normalize
```
normalized[i,j] = x[i,j] / rms[i]
```

**Step 4:** Apply weight
```
output[i,j] = normalized[i,j] * weight[j]
```

**Properties:**
- Output has same shape as input
- After normalization (before weight), RMS should be ≈ 1
- Weight scales each dimension independently
- No bias term

---

## Sign-off

**Created by:** TEAM-008 (Foundation Implementation)  
**Date:** 2025-10-08  
**Status:** Ready for validation

This checkpoint validates the first critical component of Llama-2 architecture. Success here means your tensor operations and GGUF loading are correct.

**Critical Path:** This checkpoint MUST pass before proceeding to attention (Checkpoint 2).

---

*"Get normalization right, and everything else becomes easier."*  
— TEAM-008, Foundation Implementation Division

**END CHECKPOINT**
