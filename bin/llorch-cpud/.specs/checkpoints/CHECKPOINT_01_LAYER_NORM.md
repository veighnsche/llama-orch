# CHECKPOINT 1: Layer Normalization Output

**Phase:** 5.1 - Attention Mechanism  
**Component:** LayerNorm  
**File:** `src/layers/layer_norm.rs`  
**Imports:** ndarray only (NO worker-crates)  
**Tolerance:** 1e-5  
**Critical Level:** ⚠️ HIGH - Errors propagate to all layers  
**Prerequisites:** ✅ Checkpoint 0 passed

---

## Purpose

Validate that LayerNorm computation is correct. This is the **first model component** and affects all subsequent operations.

**Why This Matters:**
- LayerNorm is used 48 times in GPT-2 Medium (2x per block × 24 blocks)
- Errors here compound through all layers
- This validates your tensor operations work correctly
- First test of checkpoint-driven development

## When to Check

- **Location:** After first layer norm in first transformer block
- **Input:** Embedding output (token + position embeddings)
- **Timing:** Week 1, Day 3-4 (after Checkpoint 0 passes)
- **Before:** Implementing attention (Checkpoint 2)

## Expected Behavior

LayerNorm should:
1. Compute mean across embedding dimension (dim=768)
2. Compute biased variance (divide by N, not N-1)
3. Normalize: `(x - mean) / sqrt(variance + eps)`
4. Apply learned scale and bias parameters
5. Use epsilon = 1e-5

## Implementation File

**File:** `src/layers/layer_norm.rs`

**Imports:**
```rust
use ndarray::{Array2, Axis};
// NO worker-crates imports - pure implementation
```

**Structure:**
```rust
pub struct LayerNorm {
    weight: Array1<f32>,  // Scale parameter [dim]
    bias: Array1<f32>,    // Bias parameter [dim]
    eps: f32,             // Epsilon (1e-5)
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 1. Compute mean across last dimension
        // 2. Compute biased variance
        // 3. Normalize
        // 4. Apply scale and bias
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ No worker-crates imports
- ✅ Simple, focused implementation

---

## Test Input

```
Prompt: "Hello."
Tokens: [15496, 13]
Model: GPT-2 Medium
Temperature: 0
```

## Validation Checklist

### ✓ Pre-Check
- [ ] Model loaded successfully
- [ ] Weights loaded (check ln_1.weight and ln_1.bias exist)
- [ ] Input embeddings shape: `[batch=1, seq=2, dim=1024]` (GPT-2 Medium)
- [ ] VALIDATE environment variable set

### ✓ Output Validation
- [ ] **Shape Check:** Output shape matches input `[1, 2, 1024]`
- [ ] **Mean Check:** Mean across dim=-1 should be ~0 (within 1e-6)
- [ ] **Variance Check:** Variance across dim=-1 should be ~1 (within 1e-5)
- [ ] **Value Range:** Values typically in range [-3, 3]
- [ ] **No NaN/Inf:** Check for invalid values

### ✓ Implementation Details
- [ ] Epsilon = 1e-5 (not 1e-6 or other value)
- [ ] Biased variance (denominator = N, not N-1)
- [ ] Scale parameter applied (ln_1.weight)
- [ ] Bias parameter applied (ln_1.bias)
- [ ] Computation on correct dimension (dim=-1 or dim=2)

### ✓ Cross-Reference Validation
- [ ] Compare first 5 values with tinygrad reference
- [ ] Compare first 5 values with Candle reference
- [ ] Difference within tolerance (1e-5)

## Reference Implementations

### Tinygrad Implementation

**File:** `/reference/tinygrad/examples/gpt2.py`

**LayerNorm Definition (Lines 62-63):**
```python
class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim, n_heads)
    self.mlp = FeedForward(dim, 4*dim)
    self.ln_1 = LayerNorm(dim, norm_eps)  # First layer norm
    self.ln_2 = LayerNorm(dim, norm_eps)  # Second layer norm
```

**Usage in Forward Pass (Line 66):**
```python
def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]):
    h = x + self.attn(self.ln_1(x), start_pos, mask).float()
    #                  ^^^^^^^^^^
    #                  LayerNorm applied BEFORE attention (pre-norm)
    return (h + self.mlp(self.ln_2(h))).contiguous()
```

**Validation Checkpoint (Lines 94-99):**
```python
# CHECKPOINT 1: Print layer norm output (first layer only)
ln1_out = self.ln_1(x)
# if getenv("VALIDATE") and not hasattr(self, "_checkpoint1_printed"):
#   print(f"[CHECKPOINT 1] LayerNorm output shape: {ln1_out.shape}")
#   print(f"[CHECKPOINT 1] Output sample: {ln1_out[0, 0, :5].numpy()}")
#   self._checkpoint1_printed = True
```

**LayerNorm Parameters:**
- **dim:** 1024 (for GPT-2 Medium)
- **norm_eps:** 1e-5 (from MODEL_PARAMS line 121)
- **Affine:** True (has learnable weight and bias)

---

### Candle Implementation

**File:** `/reference/candle/candle-transformers/src/models/bigcode.rs`

**LayerNorm Function (Lines 27-31):**
```rust
pub fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))  // epsilon = 1e-5
}
```

**Block Structure (Lines 278-280):**
```rust
impl Block {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let ln_1 = layer_norm(cfg.hidden_size, vb.pp("ln_1"))?;  // First layer norm
        let attn = Attention::load(vb.pp("attn"), cfg)?;
        let ln_2 = layer_norm(cfg.hidden_size, vb.pp("ln_2"))?;  // Second layer norm
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        Ok(Self { ln_1, attn, ln_2, mlp })
    }
}
```

**Usage in Forward Pass (Lines 323-325):**
```rust
fn forward(&mut self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let residual = hidden_states;
    let hidden_states = self.ln_1.forward(hidden_states)?;
    //                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                  LayerNorm applied BEFORE attention (pre-norm)
```

**Validation Checkpoint (Lines 326-332):**
```rust
// CHECKPOINT 1: Layer Normalization Output (uncomment to validate)
// if std::env::var("VALIDATE").is_ok() {
//     println!("[CHECKPOINT 1] LayerNorm output shape: {:?}", hidden_states.shape());
//     if let Ok(ln_sample) = hidden_states.i((0, 0, ..5))?.to_vec1::<f32>() {
//         println!("[CHECKPOINT 1] Output sample: {:?}", ln_sample);
//     }
// }
```

**LayerNorm Parameters:**
- **hidden_size:** 1024 (from Config line 47)
- **eps:** 1e-5 (hardcoded in layer_norm function)
- **Affine:** True (loads weight and bias from VarBuilder)

---

### Mistral.rs Implementation

**File:** `/reference/mistral.rs/mistralrs-core/src/layers.rs`

**LayerNorm Function (Lines 56-68):**
```rust
pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: ShardedVarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = vb.get(size, "weight")?;
    // CHECKPOINT 1: LayerNorm creation (uncomment to validate)
    // if std::env::var("VALIDATE").is_ok() {
    //     println!("[CHECKPOINT 1] LayerNorm created: size={}, eps={}", size, config.eps);
    // }
    if config.affine {
        let bias = vb.get(size, "bias")?;
        Ok(LayerNorm::new(weight, bias, config.eps))
    } else {
        Ok(LayerNorm::new_no_bias(weight, config.eps))
    }
}
```

**Usage in Models:**
- Mistral.rs uses this function across all model implementations
- Each model calls `layer_norm(hidden_size, eps, vb.pp("ln_1"))` for first norm
- Pre-norm architecture: LayerNorm → Attention → Residual

**LayerNorm Parameters:**
- **size:** Model-specific hidden dimension
- **eps:** Typically 1e-5 or 1e-6 (model-dependent)
- **Affine:** Configurable (usually True)

---

## Implementation Details Across Frameworks

### Common Ground
All three implementations:
1. ✅ Use pre-norm architecture (LayerNorm before sublayer)
2. ✅ Apply LayerNorm with epsilon = 1e-5
3. ✅ Use learnable scale (weight) and bias parameters
4. ✅ Normalize across last dimension (embedding dimension)
5. ✅ Compute biased variance (divide by N)

### Key Differences

| Aspect | Tinygrad | Candle | Mistral.rs |
|--------|----------|--------|------------|
| **Implementation** | Built-in `LayerNorm` class | `candle_nn::LayerNorm` | `candle_nn::LayerNorm` |
| **Epsilon** | Passed to constructor | Hardcoded in function | Configurable |
| **Weight Loading** | From state dict | Via VarBuilder | Via ShardedVarBuilder |
| **Validation** | Print in forward pass | Print in forward pass | Print at creation |

### Mathematical Formula (All Implementations)

```
mean = sum(x) / N
variance = sum((x - mean)^2) / N  # Biased variance
normalized = (x - mean) / sqrt(variance + eps)
output = normalized * weight + bias
```

Where:
- `x` is input tensor
- `N` is embedding dimension (1024 for GPT-2 Medium)
- `eps` = 1e-5
- `weight` and `bias` are learned parameters

---

## Weight Loading References

### Tinygrad Weight Keys
```python
# From lines 134-139
transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
# LayerNorm weights are NOT transposed
# Keys: 'h.0.ln_1.weight', 'h.0.ln_1.bias'
```

### Candle Weight Keys
```rust
// From lines 278-280
let ln_1 = layer_norm(cfg.hidden_size, vb.pp("ln_1"))?;
// VarBuilder path: "transformer.h.0.ln_1.weight"
// VarBuilder path: "transformer.h.0.ln_1.bias"
```

### Mistral.rs Weight Keys
```rust
// Uses ShardedVarBuilder with model-specific prefixes
// Typical path: "model.layers.0.input_layernorm.weight"
// Path varies by model architecture
```

## Common Failure Modes

### ❌ Mean Not Zero
**Symptom:** Mean is not close to 0  
**Cause:** Not subtracting mean before normalization  
**Fix:** Ensure `x_centered = x - mean(x, dim=-1, keepdim=True)`

### ❌ Variance Not One
**Symptom:** Variance is not close to 1  
**Cause:** Wrong variance calculation or missing normalization  
**Fix:** Check `normalized = x_centered / sqrt(variance + eps)`

### ❌ Wrong Epsilon
**Symptom:** Small numerical differences  
**Cause:** Using wrong epsilon value  
**Fix:** Set epsilon = 1e-5 (not 1e-6)

### ❌ Unbiased Variance
**Symptom:** Slight mismatch in values  
**Cause:** Using N-1 instead of N in variance  
**Fix:** Use biased variance: `variance = mean(x_centered^2)`

### ❌ Wrong Dimension
**Symptom:** Shape mismatch or incorrect normalization  
**Cause:** Normalizing across wrong dimension  
**Fix:** Normalize across last dimension (dim=-1 or dim=2)

### ❌ Missing Scale/Bias
**Symptom:** Values in wrong range  
**Cause:** Not applying learned parameters  
**Fix:** Apply `output = normalized * weight + bias`

## Debug Commands

### Print Shape
```python
# Tinygrad
print(f"Shape: {ln1_out.shape}")  # Expected: (1, 2, 1024)
```

```rust
// Rust
println!("Shape: {:?}", hidden_states.shape());  // Expected: [1, 2, 1024]
```

### Print Statistics
```python
# Tinygrad
print(f"Mean: {ln1_out.mean(axis=-1).numpy()}")  # Expected: ~0
print(f"Var: {ln1_out.var(axis=-1).numpy()}")    # Expected: ~1
```

### Print Sample Values
```python
# Tinygrad
print(f"First 5 values: {ln1_out[0, 0, :5].numpy()}")
```

```rust
// Rust
if let Ok(sample) = hidden_states.i((0, 0, ..5))?.to_vec1::<f32>() {
    println!("First 5 values: {:?}", sample);
}
```

## Success Criteria

- ✅ Shape matches input shape
- ✅ Mean ≈ 0 (within 1e-6)
- ✅ Variance ≈ 1 (within 1e-5)
- ✅ No NaN or Inf values
- ✅ Matches reference within 1e-5 tolerance
- ✅ Values in reasonable range [-3, 3]

## Implementation Steps

### Step 1: Create File Structure
```bash
touch src/layers/layer_norm.rs
```

### Step 2: Implement LayerNorm
```rust
// src/layers/layer_norm.rs
use ndarray::{Array1, Array2, Axis};

pub struct LayerNorm {
    weight: Array1<f32>,
    bias: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let variance = x.var_axis(Axis(1), 0.0);  // Biased variance
        let normalized = (x - &mean) / (variance + self.eps).mapv(f32::sqrt);
        normalized * &self.weight + &self.bias
    }
}
```

### Step 3: Write Test
```rust
// tests/checkpoint_01_layer_norm.rs
#[test]
fn checkpoint_01_matches_reference() {
    // Load reference output
    let reference = load_reference("checkpoint_01.npy");
    
    // Run our implementation
    let layer_norm = LayerNorm::new(weight, bias, 1e-5);
    let output = layer_norm.forward(&input);
    
    // Compare
    assert_tensors_close(&output, &reference, 1e-5);
}
```

### Step 4: Validate
```bash
cargo test checkpoint_01
```

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ LayerNorm is correct
- ✅ Tensor operations work
- ✅ Proceed to Checkpoint 2 (QKV Projection)
- ✅ Confidence in checkpoint-driven development

If this checkpoint **FAILS**:
- ❌ Fix LayerNorm implementation before proceeding
- ❌ Do not continue - errors will compound through all 48 LayerNorms
- ❌ Debug: Check mean, variance, epsilon, weight/bias application
- ❌ Compare intermediate values with reference

---

## Integration with Overall System

**Where This Fits:**
```
Checkpoint 0: HTTP Server ✅
    ↓
Checkpoint 1: LayerNorm ← YOU ARE HERE
    ↓
Checkpoint 2: QKV Projection
    ↓
Checkpoints 3-5: Attention
    ↓
Checkpoint 6: FFN
    ↓
Checkpoint 7: Transformer Block
    ↓
Checkpoints 8-12: Full Model
```

**Files Involved:**
- `src/layers/layer_norm.rs` - Implementation
- `tests/checkpoint_01_layer_norm.rs` - Validation
- `src/layers/mod.rs` - Export LayerNorm

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend

## Cross-Framework Comparison Example

### Expected Output for "Hello." Prompt

When running with the standard test case, the first LayerNorm output should look similar across all implementations:

**Input to LayerNorm:**
- Shape: `[1, 2, 1024]`
- First token embedding + position embedding
- Values typically in range [-0.5, 0.5]

**Output from LayerNorm:**
- Shape: `[1, 2, 1024]`
- Mean ≈ 0, Variance ≈ 1
- Values typically in range [-3, 3]

**Sample Values (first 5 elements of first token):**
```
Tinygrad:  [0.1234, -0.5678, 0.9012, -0.3456, 0.7890]
Candle:    [0.1234, -0.5678, 0.9012, -0.3456, 0.7890]
Mistral.rs:[0.1234, -0.5678, 0.9012, -0.3456, 0.7890]
```
*(Values should match within 1e-5)*

### How to Extract Values

**Tinygrad:**
```python
ln1_out = self.ln_1(x)
print(f"Shape: {ln1_out.shape}")
print(f"Sample: {ln1_out[0, 0, :5].numpy()}")
print(f"Mean: {ln1_out[0, 0].mean().numpy()}")
print(f"Std: {ln1_out[0, 0].std().numpy()}")
```

**Candle:**
```rust
let hidden_states = self.ln_1.forward(hidden_states)?;
println!("Shape: {:?}", hidden_states.shape());
if let Ok(sample) = hidden_states.i((0, 0, ..5))?.to_vec1::<f32>() {
    println!("Sample: {:?}", sample);
}
```

**Mistral.rs:**
```rust
// Similar to Candle, extract at forward pass
let normalized = layer_norm.forward(&input)?;
println!("Shape: {:?}", normalized.shape());
```

## Notes

- LayerNorm is used throughout the model (2x per block, 24 blocks = 48 times)
- Getting this wrong affects every layer
- This is the most critical checkpoint to get right first
- GPT-2 uses full LayerNorm (not RmsNorm like LLaMA)
- All three reference implementations use the same mathematical formula
- Epsilon value is critical: must be 1e-5 (not 1e-6)
- Biased variance is used (divide by N, not N-1)
