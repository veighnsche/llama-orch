# CHECKPOINT 4: Attention Scores

**Phase:** 5.4 - Attention Mechanism  
**Component:** Scaled Dot-Product (before softmax)  
**File:** `src/layers/attention/scores.rs`  
**Imports:** ndarray only (NO worker-crates)  
**Tolerance:** 1e-4  
**Critical Level:** ⚠️ HIGH - Wrong attention patterns  
**Prerequisites:** ✅ Checkpoint 3 (KV Cache) passed

---

## Purpose

Validate attention score computation before softmax. This determines which tokens attend to which.

**Why This Matters:**
- Attention scores determine information flow between tokens
- Wrong scores = wrong attention patterns = wrong model behavior
- Scaling factor (1/sqrt(head_dim)) is critical for numerical stability
- Causal mask ensures autoregressive property
- Used 384 times (16 heads × 24 layers)

## When to Check

- **Location:** After computing `(Q @ K.T) / sqrt(head_dim)` in first layer
- **Input:** Q, K from Checkpoint 2 (or cached K from Checkpoint 3)
- **Timing:** Week 2, Day 3 (after Checkpoint 3 passes)
- **Before:** Implementing attention output (Checkpoint 5)

## Validation Checklist

### ✓ Computation
- [ ] Q transposed to `[batch, n_heads, seq, head_dim]`
- [ ] K transposed to `[batch, n_heads, seq, head_dim]`
- [ ] K further transposed for matmul: `[batch, n_heads, head_dim, seq]`
- [ ] Matmul: `Q @ K.T` → `[batch, n_heads, seq_q, seq_k]`
- [ ] Scale factor = sqrt(64) = 8.0 (for head_dim=64)
- [ ] Scores = `(Q @ K.T) / 8.0`

### ✓ Shape Validation
- [ ] Scores shape: `[batch, n_heads, seq_q, seq_k]`
- [ ] For prompt: `[1, 16, 2, 2]` (GPT-2 Medium)
- [ ] For generation: `[1, 16, 1, seq_k]`

### ✓ Value Validation
- [ ] Scores typically in range [-10, 10]
- [ ] Diagonal values usually highest (self-attention)
- [ ] No NaN/Inf before mask
- [ ] Values reasonable for softmax

### ✓ Mask Application
- [ ] Causal mask shape matches scores
- [ ] Mask applied: `scores + mask`
- [ ] Future positions = -inf
- [ ] Past/current positions unchanged

### ✓ Cross-Reference (Real GPT-2 Validation)
- [ ] Load REAL GPT-2 Q, K from HuggingFace validation
- [ ] Use REAL embeddings from "Hello." tokens [15496, 13]
- [ ] Compare attention scores with HuggingFace transformers reference
- [ ] Difference within 1e-4
- [ ] Run negative tests: wrong scale factor should fail
- [ ] Run determinism test: bit-exact across runs

## Reference Locations

**Tinygrad:** `gpt2.py` line 48 (inside SDPA)  
**Candle:** `bigcode.rs` lines 187-195  
**Mistral.rs:** `attention/mod.rs` lines 118-122

## Common Failures

- ❌ Wrong scale factor (using 64 instead of 8.0)
- ❌ K not transposed
- ❌ Wrong matmul dimensions
- ❌ Mask not applied

## Success Criteria

- ✅ Scores shape correct
- ✅ Scale factor = 8.0
- ✅ Values in reasonable range
- ✅ Matches reference within 1e-4

---

## Implementation File

**File:** `src/layers/attention/scores.rs`

**Imports:**
```rust
use ndarray::{Array3, Array4};
// NO worker-crates imports - pure implementation
```

**Structure:**
```rust
pub struct AttentionScores {
    head_dim: usize,
    scale: f32,
}

impl AttentionScores {
    pub fn new(head_dim: usize) -> Self {
        let scale = (head_dim as f32).sqrt();
        Self { head_dim, scale }
    }
    
    pub fn forward(&self, q: &Array3<f32>, k: &Array3<f32>, mask: Option<&Array4<f32>>) -> Array4<f32> {
        // 1. Transpose Q, K to [batch, n_heads, seq, head_dim]
        // 2. Compute Q @ K.T
        // 3. Scale by 1/sqrt(head_dim)
        // 4. Apply causal mask if provided
        // Returns: scores [batch, n_heads, seq_q, seq_k]
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ NO worker-crates imports
- ✅ Part of attention module
- ✅ Causal mask for autoregressive generation

---

## Implementation Steps

### Step 1: Create File
```bash
touch src/layers/attention/scores.rs
```

### Step 2: Implement Attention Scores
```rust
// src/layers/attention/scores.rs
use ndarray::{Array3, Array4, s};

pub struct AttentionScores {
    scale: f32,
}

impl AttentionScores {
    pub fn new(head_dim: usize) -> Self {
        let scale = (head_dim as f32).sqrt();
        Self { scale }
    }
    
    pub fn forward(&self, q: &Array3<f32>, k: &Array3<f32>, mask: Option<&Array4<f32>>) -> Array4<f32> {
        // Transpose to [batch, n_heads, seq, head_dim]
        let q = q.permuted_axes([0, 2, 1, 3]);
        let k = k.permuted_axes([0, 2, 1, 3]);
        
        // Compute Q @ K.T
        let k_t = k.permuted_axes([0, 1, 3, 2]);
        let scores = q.dot(&k_t);
        
        // Scale
        let mut scores = scores / self.scale;
        
        // Apply mask
        if let Some(mask) = mask {
            scores = scores + mask;
        }
        
        scores
    }
}
```

### Step 3: Write Tests (Positive + Negative)

**Positive Test:**
```rust
// tests/real_gpt2_checkpoint_04.rs
#[test]
fn test_checkpoint_04_real_gpt2() {
    let dir = weights_dir();
    
    // Load REAL Q, K from Checkpoint 2
    let q: Array3<f32> = load_npy(dir.join("checkpoint_02_q.npy"));
    let k: Array3<f32> = load_npy(dir.join("checkpoint_02_k.npy"));
    
    // Load HuggingFace reference scores
    let expected: Array4<f32> = load_npy(dir.join("checkpoint_04_scores.npy"));
    
    // Run our implementation
    let scores_layer = AttentionScores::new(64);
    let scores = scores_layer.forward(&q, &k, None);  // No mask for now
    
    // Compare
    let max_diff = compare_tensors(&scores, &expected);
    assert!(max_diff < 1e-4, "Max diff {} exceeds 1e-4", max_diff);
    
    println!("✅ PASS: Attention scores match HuggingFace with REAL GPT-2");
}
```

**Negative Test:**
```rust
#[test]
#[should_panic(expected = "Max difference")]
fn test_wrong_scale_factor_fails() {
    // Use head_dim instead of sqrt(head_dim)
    let scores_layer = AttentionScores::new_with_scale(64.0);  // WRONG
    let scores = scores_layer.forward(&q, &k, None);
    assert!(compare_tensors(&scores, &expected) < 1e-4);
}
```

### Step 4: Validate with Real GPT-2
```bash
# Positive test
cargo test --test real_gpt2_checkpoint_04 -- --nocapture

# Negative tests
cargo test --test proof_negative_checkpoint_04 -- --nocapture
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
Checkpoint 4: Attention Scores ← YOU ARE HERE
    ↓
Checkpoint 5: Attention Output
    ↓
...
```

**Files Involved:**
- `src/layers/attention/scores.rs` - Implementation
- `tests/checkpoint_04_attention_scores.rs` - Validation
- `src/layers/attention/mod.rs` - Export AttentionScores

**Dependencies:**
- **Depends on:** Checkpoint 2 (Q), Checkpoint 3 (cached K)
- **Used by:** Checkpoint 5 (Attention Output - uses scores for weighted sum)

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ Attention scores are correct
- ✅ Scaling is correct
- ✅ Causal mask works
- ✅ Proceed to Checkpoint 5 (Attention Output)
- ✅ Attention mechanism nearly complete

If this checkpoint **FAILS**:
- ❌ Fix attention scores before proceeding
- ❌ Do not continue - wrong attention patterns break everything
- ❌ Debug: Check scale factor, matmul dimensions, mask application
- ❌ Verify Q @ K.T computation
- ❌ Compare intermediate values (before scaling, before mask)

---

## Notes

- Scale factor must be sqrt(head_dim), not head_dim
- Causal mask ensures tokens only attend to past/present
- Scores before softmax can be large ([-10, 10] is normal)
- Part of attention module (see ATTENTION_MODULE_STRUCTURE.md)
