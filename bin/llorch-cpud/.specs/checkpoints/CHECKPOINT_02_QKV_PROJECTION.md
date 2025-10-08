# CHECKPOINT 2: QKV Projection

**Phase:** 5.2 - Attention Mechanism  
**Component:** QKV Split  
**File:** `src/layers/attention/qkv.rs`  
**Imports:** ndarray only (NO worker-crates)  
**Tolerance:** 1e-4  
**Critical Level:** 🔴 CRITICAL - Attention completely broken if wrong  
**Prerequisites:** ✅ Checkpoint 1 (LayerNorm) passed

---

## Purpose

Validate that the combined QKV projection and split is correct. This prepares Query, Key, and Value tensors for attention computation.

**Why This Matters:**
- QKV projection is the entry point to attention mechanism
- Errors here break all attention computation (16 heads × 24 layers = 384 attention operations)
- Conv1D weight transpose is a common source of bugs
- Validates matrix multiplication works correctly

## When to Check

- **Location:** After QKV projection and split in first attention layer
- **Input:** Output from Checkpoint 1 (normalized embeddings)
- **Timing:** Week 2, Day 1 (after Checkpoint 1 passes)
- **Before:** Implementing KV Cache (Checkpoint 3)

## Expected Behavior

QKV Projection should:
1. Apply single linear layer: `Linear(1024, 3072)` (for GPT-2 Medium)
2. Reshape to `[batch, seq, 3, n_heads, head_dim]`
3. Split into Q, K, V tensors
4. Each tensor shape: `[batch, seq, n_heads, head_dim]`
5. Handle Conv1D weight transpose correctly

## Test Input

```
Prompt: "Hello."
Tokens: [15496, 13]
Model: GPT-2 Medium (n_heads=16, head_dim=64)
Input: LayerNorm output from Checkpoint 1
```

## Validation Checklist

### ✓ Pre-Check
- [ ] Checkpoint 1 passed
- [ ] c_attn weights loaded (shape: `[1024, 3072]` or transposed)
- [ ] c_attn bias loaded (shape: `[3072]`)
- [ ] Input shape correct: `[1, 2, 1024]`

### ✓ Projection Output
- [ ] **Combined QKV shape:** `[1, 2, 3072]` after linear projection
- [ ] **Reshaped:** `[1, 2, 3, 16, 64]` after reshape
- [ ] **No NaN/Inf:** Check for invalid values

### ✓ Split Outputs
- [ ] **Q shape:** `[1, 2, 16, 64]`
- [ ] **K shape:** `[1, 2, 16, 64]`
- [ ] **V shape:** `[1, 2, 16, 64]`
- [ ] **Split correct:** Q from index 0, K from index 1, V from index 2

### ✓ Weight Handling
- [ ] Conv1D weights transposed if needed
- [ ] Weight shape after transpose: `[1024, 3072]`
- [ ] Bias applied correctly
- [ ] No dimension mismatch errors

### ✓ Value Validation
- [ ] Q values in reasonable range (typically [-2, 2])
- [ ] K values in reasonable range (typically [-2, 2])
- [ ] V values in reasonable range (typically [-2, 2])
- [ ] Values differ between Q, K, V (not identical)

### ✓ Cross-Reference Validation
- [ ] Compare Q[0, 0, 0, :5] with tinygrad reference
- [ ] Compare K[0, 0, 0, :5] with Candle reference
- [ ] Difference within tolerance (1e-4)

## Reference Outputs

### Tinygrad Location
```python
# File: gpt2.py, Line 29-35
xqkv = self.c_attn(x).reshape(None, None, 3, self.n_heads, self.head_dim)
xq, xk, xv = [xqkv[:, :, i, :, :] for i in range(3)]
# CHECKPOINT 2: Print Q, K, V for validation (first layer only)
# if getenv("VALIDATE") and not hasattr(self, "_checkpoint2_printed"):
#   print(f"[CHECKPOINT 2] QKV shapes: Q={xq.shape}, K={xk.shape}, V={xv.shape}")
#   print(f"[CHECKPOINT 2] Q sample: {xq[0, 0, 0, :5].numpy()}")
```

### Candle Location
```rust
// File: bigcode.rs, Line 214-224
let qkv = self.c_attn.forward(hidden_states)?;
let (query, key_value) = if self.multi_query {
    let query = qkv.i((.., .., ..self.embed_dim))?;
    let key_value = qkv.i((.., .., self.embed_dim..self.embed_dim + 2 * self.kv_dim))?;
    // if std::env::var("VALIDATE").is_ok() {
    //     println!("[CHECKPOINT 2] QKV shapes: query={:?}, key_value={:?}", query.shape(), key_value.shape());
    // }
    (query, key_value)
}
```

## Common Failure Modes

### ❌ Weight Not Transposed
**Symptom:** Shape mismatch or wrong output dimensions  
**Cause:** Conv1D weights need transpose  
**Fix:** Transpose weight matrix before applying linear layer

### ❌ Wrong Reshape Dimensions
**Symptom:** Runtime error or wrong tensor shapes  
**Cause:** Incorrect reshape parameters  
**Fix:** Reshape to `[batch, seq, 3, n_heads, head_dim]`

### ❌ Wrong Split Indexing
**Symptom:** Q, K, V have wrong values  
**Cause:** Incorrect indexing in split  
**Fix:** Use `xqkv[:, :, 0, :, :]` for Q, `[:, :, 1, :, :]` for K, etc.

### ❌ Missing Bias
**Symptom:** Values offset from reference  
**Cause:** Not applying bias term  
**Fix:** Ensure bias is added after weight multiplication

### ❌ Wrong Head Dimensions
**Symptom:** Shape mismatch in attention  
**Cause:** n_heads or head_dim calculated wrong  
**Fix:** For GPT-2 Medium: n_heads=16, head_dim=64 (1024/16)

## Debug Commands

### Print Shapes
```python
# Tinygrad
print(f"QKV combined: {xqkv.shape}")  # Expected: (1, 2, 3, 16, 64)
print(f"Q: {xq.shape}, K: {xk.shape}, V: {xv.shape}")  # Each: (1, 2, 16, 64)
```

### Print Sample Values
```python
# Tinygrad
print(f"Q sample (head 0): {xq[0, 0, 0, :5].numpy()}")
print(f"K sample (head 0): {xk[0, 0, 0, :5].numpy()}")
print(f"V sample (head 0): {xv[0, 0, 0, :5].numpy()}")
```

### Check Weight Shape
```python
# Tinygrad
print(f"c_attn weight shape: {self.c_attn.weight.shape}")  # Expected: (1024, 3072)
print(f"c_attn bias shape: {self.c_attn.bias.shape}")      # Expected: (3072,)
```

## Success Criteria

- ✅ All shapes correct
- ✅ Q, K, V values differ from each other
- ✅ No NaN or Inf values
- ✅ Matches reference within 1e-4 tolerance
- ✅ Values in reasonable range [-2, 2]
- ✅ Weight transpose handled correctly

## Implementation File

**File:** `src/layers/attention/qkv.rs`

**Imports:**
```rust
use ndarray::{Array2, Array3};
// NO worker-crates imports - pure implementation
```

**Structure:**
```rust
pub struct QKVProjection {
    weight: Array2<f32>,  // [dim, 3*dim]
    bias: Array1<f32>,    // [3*dim]
    n_heads: usize,
    head_dim: usize,
}

impl QKVProjection {
    pub fn new(weight: Array2<f32>, bias: Array1<f32>, n_heads: usize) -> Self {
        let head_dim = weight.shape()[0] / n_heads;
        Self { weight, bias, n_heads, head_dim }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        // 1. Linear projection: x @ weight + bias
        // 2. Reshape to [batch, seq, 3, n_heads, head_dim]
        // 3. Split into Q, K, V
        // Returns: (Q, K, V) each [batch, seq, n_heads, head_dim]
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ NO worker-crates imports
- ✅ Handle Conv1D weight transpose correctly
- ✅ Part of attention module (see ATTENTION_MODULE_STRUCTURE.md)

---

## Implementation Steps

### Step 1: Create File Structure
```bash
mkdir -p src/layers/attention
touch src/layers/attention/qkv.rs
touch src/layers/attention/mod.rs
```

### Step 2: Implement QKV Projection
```rust
// src/layers/attention/qkv.rs
use ndarray::{Array1, Array2, Array3, s};

pub struct QKVProjection {
    weight: Array2<f32>,
    bias: Array1<f32>,
    n_heads: usize,
    head_dim: usize,
}

impl QKVProjection {
    pub fn new(weight: Array2<f32>, bias: Array1<f32>, n_heads: usize) -> Self {
        let dim = weight.shape()[0];
        let head_dim = dim / n_heads;
        Self { weight, bias, n_heads, head_dim }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        // Linear projection
        let qkv = x.dot(&self.weight) + &self.bias;
        
        // Reshape to [batch, seq, 3, n_heads, head_dim]
        let shape = qkv.shape();
        let qkv = qkv.into_shape((shape[0], shape[1], 3, self.n_heads, self.head_dim)).unwrap();
        
        // Split into Q, K, V
        let q = qkv.slice(s![.., .., 0, .., ..]).to_owned();
        let k = qkv.slice(s![.., .., 1, .., ..]).to_owned();
        let v = qkv.slice(s![.., .., 2, .., ..]).to_owned();
        
        (q, k, v)
    }
}
```

### Step 3: Write Test
```rust
// tests/checkpoint_02_qkv.rs
#[test]
fn checkpoint_02_matches_reference() {
    // Load reference output
    let reference_q = load_reference("checkpoint_02_q.npy");
    let reference_k = load_reference("checkpoint_02_k.npy");
    let reference_v = load_reference("checkpoint_02_v.npy");
    
    // Run our implementation
    let qkv = QKVProjection::new(weight, bias, 16);
    let (q, k, v) = qkv.forward(&input);
    
    // Compare
    assert_tensors_close(&q, &reference_q, 1e-4);
    assert_tensors_close(&k, &reference_k, 1e-4);
    assert_tensors_close(&v, &reference_v, 1e-4);
}
```

### Step 4: Validate
```bash
cargo test checkpoint_02
```

---

## Integration with Overall System

**Where This Fits:**
```
Checkpoint 0: HTTP Server ✅
    ↓
Checkpoint 1: LayerNorm ✅
    ↓
Checkpoint 2: QKV Projection ← YOU ARE HERE
    ↓
Checkpoint 3: KV Cache
    ↓
Checkpoint 4: Attention Scores
    ↓
...
```

**Files Involved:**
- `src/layers/attention/qkv.rs` - Implementation
- `tests/checkpoint_02_qkv.rs` - Validation
- `src/layers/attention/mod.rs` - Export QKVProjection

**Dependencies:**
- **Depends on:** Checkpoint 1 (LayerNorm output)
- **Used by:** Checkpoint 3 (KV Cache) and Checkpoint 4 (Attention Scores)

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ QKV projection is correct
- ✅ Matrix multiplication works
- ✅ Conv1D transpose handled correctly
- ✅ Proceed to Checkpoint 3 (KV Cache)
- ✅ Ready to implement attention mechanism

If this checkpoint **FAILS**:
- ❌ Fix QKV projection before proceeding
- ❌ Do not continue - attention will be completely broken
- ❌ Debug: Check weight transpose, reshape dimensions, split indexing
- ❌ Compare intermediate values with reference (after linear, after reshape)

---

## Notes

- Conv1D in PyTorch stores weights transposed
- Tinygrad requires explicit transpose
- Multi-query attention (Candle) has different split logic
- This is the input to attention - must be perfect
- Part of attention module (see ATTENTION_MODULE_STRUCTURE.md)
