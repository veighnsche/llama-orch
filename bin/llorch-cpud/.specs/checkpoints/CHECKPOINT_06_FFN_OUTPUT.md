# CHECKPOINT 6: FFN Output

**Phase:** 6.1 - Feedforward Network  
**Component:** MLP (c_fc → GELU → c_proj)  
**File:** `src/layers/ffn.rs`  
**Imports:** ndarray only (NO worker-crates)  
**Tolerance:** 1e-4  
**Critical Level:** ⚠️ HIGH - Half of each transformer block  
**Prerequisites:** ✅ Checkpoint 5 (Attention Output) passed

---

## Purpose

Validate feedforward network computation. FFN is the other half of each transformer block (alongside attention).

**Why This Matters:**
- FFN is 50% of each transformer block (attention is the other 50%)
- 4x expansion increases model capacity
- GELU activation is critical for non-linearity
- Used 24 times (once per block)
- Feeds into second residual connection

## When to Check

- **Location:** After FFN in first transformer block
- **Input:** Output from Checkpoint 5 (after residual + layer norm)
- **Timing:** Week 3, Day 1-2 (after Checkpoint 5 passes)
- **Before:** Implementing complete transformer block (Checkpoint 7)

## Validation Checklist

### ✓ Up Projection (c_fc)
- [ ] Input shape: `[1, 2, 1024]`
- [ ] Weight shape: `[1024, 4096]` (4x expansion)
- [ ] Bias shape: `[4096]`
- [ ] Output shape: `[1, 2, 4096]`

### ✓ GELU Activation
- [ ] Formula: `x * 0.5 * (1 + erf(x / sqrt(2)))` (exact)
- [ ] OR: `x * 0.5 * (1 + tanh(...))` (approximation)
- [ ] Output shape unchanged: `[1, 2, 4096]`
- [ ] Values modified (not identity)

### ✓ Down Projection (c_proj)
- [ ] Weight shape: `[4096, 1024]`
- [ ] Bias shape: `[1024]`
- [ ] Output shape: `[1, 2, 1024]`

### ✓ Value Validation
- [ ] After c_fc: values in range (typically [-5, 5])
- [ ] After GELU: positive bias (GELU keeps positive, dampens negative)
- [ ] Final output: values in range (typically [-2, 2])
- [ ] No NaN/Inf

### ✓ Cross-Reference (Real GPT-2 Validation)
- [ ] Load REAL GPT-2 c_fc and c_proj weights from HuggingFace
- [ ] Use REAL input from previous checkpoint
- [ ] Compare FFN output with HuggingFace transformers reference
- [ ] Difference within 1e-4
- [ ] Run negative tests: wrong GELU formula should fail
- [ ] Run negative tests: wrong expansion factor should fail
- [ ] Run determinism test: bit-exact across runs

## Reference Locations

**Tinygrad:** `gpt2.py` lines 55-56, 78-84  
**Candle:** `bigcode.rs` lines 285-295  
**Mistral.rs:** Model-specific MLP implementations

## Common Failures

- ❌ Wrong GELU formula (tanh vs exact)
- ❌ 4x expansion not applied (wrong hidden dim)
- ❌ Weights not transposed
- ❌ Missing bias terms

## Success Criteria

- ✅ All shapes correct
- ✅ GELU applied correctly
- ✅ 4x expansion (1024 → 4096 → 1024)
- ✅ Matches reference within 1e-4
- ✅ Values in reasonable range

---

## Implementation File

**File:** `src/layers/ffn.rs`

**Imports:**
```rust
use ndarray::Array2;
// NO worker-crates imports - pure implementation
```

**Structure:**
```rust
pub struct FFN {
    c_fc_weight: Array2<f32>,    // [dim, 4*dim]
    c_fc_bias: Array1<f32>,      // [4*dim]
    c_proj_weight: Array2<f32>,  // [4*dim, dim]
    c_proj_bias: Array1<f32>,    // [dim]
}

impl FFN {
    pub fn new(
        fc_weight: Array2<f32>,
        fc_bias: Array1<f32>,
        proj_weight: Array2<f32>,
        proj_bias: Array1<f32>,
    ) -> Self {
        Self {
            c_fc_weight: fc_weight,
            c_fc_bias: fc_bias,
            c_proj_weight: proj_weight,
            c_proj_bias: proj_bias,
        }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 1. Up projection: x @ c_fc_weight + c_fc_bias
        // 2. GELU activation
        // 3. Down projection: x @ c_proj_weight + c_proj_bias
        // Returns: output [batch, seq, dim]
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ NO worker-crates imports
- ✅ Two linear layers with GELU activation
- ✅ 4x expansion (dim → 4*dim → dim)

---

## Implementation Steps

### Step 1: Create File
```bash
touch src/layers/ffn.rs
```

### Step 2: Implement FFN
```rust
// src/layers/ffn.rs
use ndarray::{Array1, Array2};

pub struct FFN {
    c_fc_weight: Array2<f32>,
    c_fc_bias: Array1<f32>,
    c_proj_weight: Array2<f32>,
    c_proj_bias: Array1<f32>,
}

impl FFN {
    pub fn new(
        fc_weight: Array2<f32>,
        fc_bias: Array1<f32>,
        proj_weight: Array2<f32>,
        proj_bias: Array1<f32>,
    ) -> Self {
        Self {
            c_fc_weight: fc_weight,
            c_fc_bias: fc_bias,
            c_proj_weight: proj_weight,
            c_proj_bias: proj_bias,
        }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Up projection
        let hidden = x.dot(&self.c_fc_weight) + &self.c_fc_bias;
        
        // GELU activation
        let hidden = gelu(&hidden);
        
        // Down projection
        let output = hidden.dot(&self.c_proj_weight) + &self.c_proj_bias;
        
        output
    }
}

fn gelu(x: &Array2<f32>) -> Array2<f32> {
    // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    x.mapv(|v| {
        v * 0.5 * (1.0 + erf(v / std::f32::consts::SQRT_2))
    })
}
```

### Step 3: Write Tests (Positive + Negative)

**Positive Test:**
```rust
// tests/real_gpt2_checkpoint_06.rs
#[test]
fn test_checkpoint_06_real_gpt2() {
    let dir = weights_dir();
    
    // Load REAL FFN weights
    let c_fc_weight: Array2<f32> = load_npy(dir.join("h0_c_fc_weight.npy"));
    let c_fc_bias: Array1<f32> = load_npy(dir.join("h0_c_fc_bias.npy"));
    let c_proj_weight: Array2<f32> = load_npy(dir.join("h0_ffn_c_proj_weight.npy"));
    let c_proj_bias: Array1<f32> = load_npy(dir.join("h0_ffn_c_proj_bias.npy"));
    
    // Load input from previous checkpoint
    let input: Array2<f32> = load_npy(dir.join("checkpoint_05_output.npy"));
    
    // Load HuggingFace reference
    let expected: Array2<f32> = load_npy(dir.join("checkpoint_06_ffn.npy"));
    
    // Run our implementation
    let ffn = FFN::new(c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias);
    let output = ffn.forward(&input);
    
    // Compare
    let max_diff = compare_tensors(&output, &expected);
    assert!(max_diff < 1e-4, "Max diff {} exceeds 1e-4", max_diff);
    
    println!("✅ PASS: FFN output matches HuggingFace with REAL GPT-2");
}
```

**Negative Tests:**
```rust
#[test]
#[should_panic(expected = "Max difference")]
fn test_wrong_gelu_fails() {
    // Use tanh approximation instead of exact GELU
    let ffn = FFN::new_with_gelu_approx(...);
    let output = ffn.forward(&input);
    assert!(compare_tensors(&output, &expected) < 1e-4);
}

#[test]
#[should_panic]
fn test_wrong_expansion_fails() {
    // Use 2x expansion instead of 4x
    let wrong_fc_weight = Array2::zeros((768, 1536));  // WRONG: should be 3072
    let ffn = FFN::new(wrong_fc_weight, ...);
    // Should panic with dimension mismatch
}
```

### Step 4: Validate with Real GPT-2
```bash
# Positive test
cargo test --test real_gpt2_checkpoint_06 -- --nocapture

# Negative tests
cargo test --test proof_negative_checkpoint_06 -- --nocapture
```

**Expected:**
- Positive test: ✅ PASS (max diff < 1e-4)
- Negative tests: ❌ All should panic with large errors

---

## Integration with Overall System

**Where This Fits:**
```
Checkpoint 0: HTTP Server ✅
    ↓
Checkpoint 1: LayerNorm ✅
    ↓
Checkpoint 2: QKV Projection ✅
    ↓
Checkpoint 3: KV Cache ✅
    ↓
Checkpoint 4: Attention Scores ✅
    ↓
Checkpoint 5: Attention Output ✅
    ↓
Checkpoint 6: FFN Output ← YOU ARE HERE
    ↓
Checkpoint 7: Transformer Block
    ↓
...
```

**Files Involved:**
- `src/layers/ffn.rs` - Implementation
- `tests/checkpoint_06_ffn.rs` - Validation
- `src/layers/mod.rs` - Export FFN

**Dependencies:**
- **Depends on:** Checkpoint 5 (attention output + residual + layer norm)
- **Used by:** Checkpoint 7 (Transformer Block - second residual)

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ FFN is correct
- ✅ GELU activation works
- ✅ 4x expansion works
- ✅ Proceed to Checkpoint 7 (Transformer Block)
- ✅ Ready to combine attention + FFN!

If this checkpoint **FAILS**:
- ❌ Fix FFN before proceeding
- ❌ Do not continue - second residual will be wrong
- ❌ Debug: Check GELU formula, weight shapes, bias application
- ❌ Verify 4x expansion (1024 → 4096 → 1024)
- ❌ Compare intermediate values (after c_fc, after GELU, after c_proj)

---

## Notes

- GELU is exact formula (not tanh approximation)
- 4x expansion is standard for GPT-2
- FFN is applied after second layer norm
- Output feeds into second residual connection
- FFN is simpler than attention but equally important
