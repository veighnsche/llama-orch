# CHECKPOINT 1B: RoPE Application (Llama-2)

**Created by:** TEAM-008  
**Phase:** 3.3 - Position Encoding  
**Component:** RoPE (Rotary Position Embeddings)  
**File:** `src/layers/rope.rs`  
**Imports:** ndarray only (NO worker-crates)  
**Tolerance:** 1e-5  
**Critical Level:** 🔴 CRITICAL - Position encoding affects all tokens  
**Prerequisites:** ✅ Checkpoint 1 (RMSNorm) and Checkpoint 2 (QKV) passed

---

## Purpose

Validate that RoPE (Rotary Position Embeddings) is correctly applied to Q and K tensors. This is **unique to Llama-2** and replaces GPT-2's learned position embeddings.

**Why This Matters:**
- RoPE encodes position information through rotation
- Applied to Q and K (NOT V)
- Used in all 32 attention layers
- Wrong rotation breaks position awareness
- Critical for long-context understanding
- Different from GPT-2's absolute position embeddings

**Key Difference from GPT-2:**
- GPT-2: Learned position embeddings added to token embeddings
- Llama-2: RoPE rotates Q and K based on position (no learned embeddings)

---

## When to Check

- **Location:** After QKV projection, before attention scores
- **Input:** Q, K from Checkpoint 2
- **Timing:** Week 2, Day 2 (after Checkpoint 2 passes)
- **Before:** Computing attention scores (Checkpoint 4)

---

## Expected Behavior

RoPE should:
1. Precompute rotation frequencies: `θ_i = 10000^(-2i/d)` for i in [0, d/2)
2. Compute cos and sin for each position
3. Apply rotation to Q and K (dimension pairs)
4. NOT modify V
5. Preserve tensor shapes

---

## Implementation File

**File:** `src/layers/rope.rs`

**Imports:**
```rust
use ndarray::{Array1, Array3};
use std::f32::consts::PI;
// NO worker-crates imports - pure implementation
```

**Structure:**
```rust
pub struct RoPE {
    cos_cache: Array1<f32>,  // Precomputed cos values
    sin_cache: Array1<f32>,  // Precomputed sin values
    head_dim: usize,
    max_seq_len: usize,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        // Precompute rotation frequencies
        // theta = 10000.0 for Llama-2
    }
    
    pub fn apply(&self, q: &Array3<f32>, k: &Array3<f32>, position: usize) 
        -> (Array3<f32>, Array3<f32>) {
        // Apply rotation to Q and K
        // Returns: (rotated_q, rotated_k)
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ NO worker-crates imports
- ✅ Precompute cos/sin for efficiency
- ✅ Apply to Q and K only (NOT V)

---

## Mathematical Formula

### Frequency Computation
```
For dimension pairs (2i, 2i+1) where i ∈ [0, head_dim/2):
θ_i = 10000^(-2i/head_dim)
```

### Rotation Application
For each position `m` and dimension pair `(x, y)`:
```
x' = x * cos(m * θ_i) - y * sin(m * θ_i)
y' = y * cos(m * θ_i) + x * sin(m * θ_i)
```

### In Practice (Llama-2 7B)
- `head_dim = 128`
- `theta = 10000.0`
- Rotate 64 dimension pairs per head
- Apply to all 32 heads

---

## Test Input

```
Prompt: "Hello"
Tokens: [1, 15043]  # BOS + "Hello"
Model: Llama-2 7B
Position 0: BOS token
Position 1: "Hello" token
```

---

## Validation Checklist

### ✓ Pre-Check
- [ ] Checkpoint 1 (RMSNorm) passed
- [ ] Checkpoint 2 (QKV) passed
- [ ] Q shape: `[batch, seq, n_heads, head_dim]` = `[1, 2, 32, 128]`
- [ ] K shape: `[batch, seq, n_heads, head_dim]` = `[1, 2, 32, 128]`
- [ ] V shape: `[batch, seq, n_heads, head_dim]` = `[1, 2, 32, 128]`

### ✓ Frequency Computation
- [ ] Theta = 10000.0
- [ ] Frequencies computed for head_dim/2 = 64 pairs
- [ ] Formula: `θ_i = 10000^(-2i/128)`
- [ ] Frequencies decrease exponentially

### ✓ Cos/Sin Cache
- [ ] Cos cache shape: `[max_seq_len, head_dim/2]`
- [ ] Sin cache shape: `[max_seq_len, head_dim/2]`
- [ ] Values precomputed for all positions
- [ ] Cos values in [-1, 1]
- [ ] Sin values in [-1, 1]

### ✓ Rotation Application
- [ ] Applied to Q: shape unchanged `[1, 2, 32, 128]`
- [ ] Applied to K: shape unchanged `[1, 2, 32, 128]`
- [ ] NOT applied to V: V unchanged
- [ ] Rotation applied to dimension pairs (0,1), (2,3), ..., (126,127)
- [ ] Position-dependent: different rotation per token

### ✓ Output Validation
- [ ] Q_rotated shape: `[1, 2, 32, 128]`
- [ ] K_rotated shape: `[1, 2, 32, 128]`
- [ ] V unchanged: exact match with input V
- [ ] No NaN/Inf in rotated tensors
- [ ] Values in reasonable range (typically [-2, 2])

### ✓ Cross-Reference Validation
- [ ] Compare Q_rotated[0, 0, 0, :5] with llama.cpp reference
- [ ] Compare K_rotated[0, 0, 0, :5] with llama.cpp reference
- [ ] Difference within tolerance (1e-5)

---

## Reference Implementation

### llama.cpp Location
**File:** `reference/llama.cpp/ggml/src/ggml.c`

**RoPE Function:**
```c
// Lines ~12000-12100
static void ggml_compute_forward_rope_f32(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst,
    const bool forward) {
    
    const float theta_scale = powf(10000.0, -2.0f/n_dims);
    
    for (int i = 0; i < n_dims; i += 2) {
        const float theta = (float)p * theta_scale;
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);
        
        const float x0 = src[i];
        const float x1 = src[i+1];
        
        dst[i]   = x0*cos_theta - x1*sin_theta;
        dst[i+1] = x0*sin_theta + x1*cos_theta;
    }
}
```

---

## Common Failure Modes

### ❌ Wrong Theta Value
**Symptom:** Position encoding incorrect  
**Cause:** Using wrong base (not 10000.0)  
**Fix:** Set `theta = 10000.0` exactly

### ❌ Applied to V
**Symptom:** Attention output wrong  
**Cause:** Rotating V (should only rotate Q and K)  
**Fix:** Only apply RoPE to Q and K

### ❌ Wrong Dimension Pairing
**Symptom:** Rotation incorrect  
**Cause:** Not pairing consecutive dimensions  
**Fix:** Rotate (0,1), (2,3), ..., (126,127)

### ❌ Position Not Used
**Symptom:** All tokens have same rotation  
**Cause:** Not multiplying by position `m`  
**Fix:** `cos(m * θ_i)` and `sin(m * θ_i)`

### ❌ Shape Changed
**Symptom:** Dimension mismatch  
**Cause:** Rotation changed tensor shape  
**Fix:** Ensure output shape matches input shape

---

## Debug Commands

### Print Frequencies
```rust
println!("First 5 frequencies: {:?}", &freqs[..5]);
// Expected: decreasing values starting from ~0.0001
```

### Print Cos/Sin Cache
```rust
println!("Cos cache shape: {:?}", cos_cache.shape());
println!("Sin cache shape: {:?}", sin_cache.shape());
println!("Cos[0, 0]: {}", cos_cache[[0, 0]]);  // Should be 1.0
println!("Sin[0, 0]: {}", sin_cache[[0, 0]]);  // Should be 0.0
```

### Print Rotated Values
```rust
println!("Q before RoPE: {:?}", &q[[0, 0, 0, ..5]]);
println!("Q after RoPE:  {:?}", &q_rot[[0, 0, 0, ..5]]);
println!("K before RoPE: {:?}", &k[[0, 0, 0, ..5]]);
println!("K after RoPE:  {:?}", &k_rot[[0, 0, 0, ..5]]);
println!("V unchanged:   {:?}", &v[[0, 0, 0, ..5]]);
```

---

## Success Criteria

- ✅ Frequencies computed correctly
- ✅ Cos/sin cache precomputed
- ✅ Q rotated correctly
- ✅ K rotated correctly
- ✅ V unchanged
- ✅ Shapes preserved
- ✅ No NaN/Inf values
- ✅ Matches llama.cpp within 1e-5 tolerance

---

## Implementation Steps

### Step 1: Create File Structure
```bash
touch src/layers/rope.rs
```

### Step 2: Implement RoPE
```rust
// src/layers/rope.rs
use ndarray::{Array1, Array2, Array3};
use std::f32::consts::PI;

pub struct RoPE {
    cos_cache: Array2<f32>,  // [max_seq_len, head_dim/2]
    sin_cache: Array2<f32>,  // [max_seq_len, head_dim/2]
    head_dim: usize,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let dim_pairs = head_dim / 2;
        
        // Compute frequencies
        let mut freqs = Array1::zeros(dim_pairs);
        for i in 0..dim_pairs {
            freqs[i] = theta.powf(-2.0 * (i as f32) / (head_dim as f32));
        }
        
        // Precompute cos and sin for all positions
        let mut cos_cache = Array2::zeros((max_seq_len, dim_pairs));
        let mut sin_cache = Array2::zeros((max_seq_len, dim_pairs));
        
        for pos in 0..max_seq_len {
            for i in 0..dim_pairs {
                let angle = (pos as f32) * freqs[i];
                cos_cache[[pos, i]] = angle.cos();
                sin_cache[[pos, i]] = angle.sin();
            }
        }
        
        Self { cos_cache, sin_cache, head_dim }
    }
    
    pub fn apply(&self, q: &Array3<f32>, k: &Array3<f32>, position: usize) 
        -> (Array3<f32>, Array3<f32>) {
        let mut q_rot = q.clone();
        let mut k_rot = k.clone();
        
        let (batch, seq, n_heads, head_dim) = (
            q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]
        );
        
        // Apply rotation to each head
        for b in 0..batch {
            for s in 0..seq {
                let pos = position + s;
                for h in 0..n_heads {
                    for i in 0..(head_dim / 2) {
                        let cos = self.cos_cache[[pos, i]];
                        let sin = self.sin_cache[[pos, i]];
                        
                        // Rotate Q
                        let q0 = q[[b, s, h, 2*i]];
                        let q1 = q[[b, s, h, 2*i + 1]];
                        q_rot[[b, s, h, 2*i]] = q0 * cos - q1 * sin;
                        q_rot[[b, s, h, 2*i + 1]] = q0 * sin + q1 * cos;
                        
                        // Rotate K
                        let k0 = k[[b, s, h, 2*i]];
                        let k1 = k[[b, s, h, 2*i + 1]];
                        k_rot[[b, s, h, 2*i]] = k0 * cos - k1 * sin;
                        k_rot[[b, s, h, 2*i + 1]] = k0 * sin + k1 * cos;
                    }
                }
            }
        }
        
        (q_rot, k_rot)
    }
}
```

### Step 3: Write Test
```rust
// tests/checkpoint_01b_rope.rs
#[test]
fn checkpoint_01b_matches_reference() {
    // Load reference output from llama.cpp
    let reference_q = load_reference("checkpoint_01b_q_rope.npy");
    let reference_k = load_reference("checkpoint_01b_k_rope.npy");
    
    // Load Q, K from Checkpoint 2
    let q = load_checkpoint("checkpoint_02_q.npy");
    let k = load_checkpoint("checkpoint_02_k.npy");
    
    // Run our implementation
    let rope = RoPE::new(128, 4096, 10000.0);
    let (q_rot, k_rot) = rope.apply(&q, &k, 0);
    
    // Compare
    assert_tensors_close(&q_rot, &reference_q, 1e-5);
    assert_tensors_close(&k_rot, &reference_k, 1e-5);
}
```

### Step 4: Validate
```bash
cargo test checkpoint_01b
```

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ RoPE is correct
- ✅ Position encoding works
- ✅ Proceed to Checkpoint 3 (KV Cache)
- ✅ Attention mechanism ready

If this checkpoint **FAILS**:
- ❌ Fix RoPE implementation before proceeding
- ❌ Do not continue - position encoding affects all layers
- ❌ Debug: Check theta, frequency computation, rotation formula
- ❌ Verify cos/sin cache values
- ❌ Compare intermediate values with reference

---

## Integration with Overall System

**Where This Fits:**
```
Checkpoint 0: HTTP Server ✅
    ↓
Checkpoint 1: RMSNorm ✅
    ↓
Checkpoint 2: QKV Projection ✅
    ↓
Checkpoint 1B: RoPE Application ← YOU ARE HERE
    ↓
Checkpoint 3: KV Cache
    ↓
Checkpoint 4: Attention Scores
    ↓
...
```

**Files Involved:**
- `src/layers/rope.rs` - Implementation
- `tests/checkpoint_01b_rope.rs` - Validation
- `src/layers/mod.rs` - Export RoPE

**Dependencies:**
- **Depends on:** Checkpoint 2 (QKV Projection - provides Q, K)
- **Used by:** Checkpoint 4 (Attention Scores - uses rotated Q, K)

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend

---

## Notes

- RoPE is unique to Llama-2 (not in GPT-2)
- Applied to Q and K only (NOT V)
- Position-dependent rotation
- Precompute cos/sin for efficiency
- Theta = 10000.0 for Llama-2
- Critical for position awareness
- Used in all 32 attention layers

---

**Status:** ⬜ Not Started  
**Estimated Time:** 4-6 hours  
**Blocking:** Must pass before Checkpoint 3

---

Built by TEAM-008 🌊
