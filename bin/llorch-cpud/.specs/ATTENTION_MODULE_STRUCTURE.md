# Attention Module Structure

**Date:** 2025-10-08  
**Purpose:** Define how to split attention.rs into multiple focused files  
**Status:** Architecture guide

---

## Problem

Attention is complex and covers Checkpoints 2-5:
- Checkpoint 2: QKV Projection
- Checkpoint 3: KV Cache
- Checkpoint 4: Attention Scores
- Checkpoint 5: Attention Output

Putting all of this in one `attention.rs` file would be:
- ❌ Too large (300+ lines)
- ❌ Hard to validate incrementally
- ❌ Mixes concerns (projection, cache, computation, output)

---

## Solution: Split into Focused Files

### Directory Structure

```
src/layers/attention/
├── mod.rs                  # Public API and Attention struct
├── qkv.rs                  # QKV projection (Checkpoint 2)
├── cache.rs                # KV cache management (Checkpoint 3)
├── scores.rs               # Attention score computation (Checkpoint 4)
└── output.rs               # Attention output projection (Checkpoint 5)
```

---

## File Responsibilities

### 1. mod.rs - Public API

**Purpose:** Orchestrate attention components

```rust
// src/layers/attention/mod.rs

mod qkv;
mod cache;
mod scores;
mod output;

pub use qkv::QKVProjection;
pub use cache::KVCache;
pub use scores::AttentionScores;
pub use output::AttentionOutput;

use ndarray::Array2;

/// Complete attention mechanism
pub struct Attention {
    qkv_proj: QKVProjection,
    cache: KVCache,
    scores: AttentionScores,
    output_proj: AttentionOutput,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        let head_dim = dim / n_heads;
        Self {
            qkv_proj: QKVProjection::new(dim, n_heads),
            cache: KVCache::new(n_heads, head_dim),
            scores: AttentionScores::new(head_dim),
            output_proj: AttentionOutput::new(dim),
            n_heads,
            head_dim,
        }
    }
    
    pub fn forward(
        &mut self,
        x: &Array2<f32>,
        start_pos: usize,
        mask: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        // 1. QKV Projection (Checkpoint 2)
        let (q, k, v) = self.qkv_proj.forward(x);
        
        // 2. Update cache (Checkpoint 3)
        let (k_cached, v_cached) = self.cache.update(k, v, start_pos);
        
        // 3. Compute attention scores (Checkpoint 4)
        let attn_weights = self.scores.compute(&q, &k_cached, mask);
        
        // 4. Apply attention and project output (Checkpoint 5)
        let output = self.output_proj.forward(&attn_weights, &v_cached);
        
        output
    }
}
```

**Validation:**
- [ ] Orchestrates all attention components
- [ ] Clean separation of concerns
- [ ] Each step maps to a checkpoint

---

### 2. qkv.rs - QKV Projection (Checkpoint 2)

**Purpose:** Project input to Query, Key, Value

```rust
// src/layers/attention/qkv.rs

use ndarray::{Array2, Array3};

/// QKV projection layer
pub struct QKVProjection {
    weight: Array2<f32>,  // [dim, 3 * dim]
    bias: Array1<f32>,    // [3 * dim]
    n_heads: usize,
    head_dim: usize,
    dim: usize,
}

impl QKVProjection {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        let head_dim = dim / n_heads;
        Self {
            weight: Array2::zeros((dim, 3 * dim)),
            bias: Array1::zeros(3 * dim),
            n_heads,
            head_dim,
            dim,
        }
    }
    
    /// Project input to Q, K, V
    /// 
    /// Input: [batch, seq_len, dim]
    /// Output: (Q, K, V) each [batch, seq_len, n_heads, head_dim]
    pub fn forward(&self, x: &Array2<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        // 1. Linear projection: x @ weight + bias
        let qkv = x.dot(&self.weight) + &self.bias;  // [batch, seq_len, 3*dim]
        
        // 2. Reshape to [batch, seq_len, 3, n_heads, head_dim]
        let shape = (x.shape()[0], x.shape()[1], 3, self.n_heads, self.head_dim);
        let qkv = qkv.into_shape(shape).unwrap();
        
        // 3. Split into Q, K, V
        let q = qkv.slice(s![.., .., 0, .., ..]).to_owned();
        let k = qkv.slice(s![.., .., 1, .., ..]).to_owned();
        let v = qkv.slice(s![.., .., 2, .., ..]).to_owned();
        
        (q, k, v)
    }
    
    pub fn load_weights(&mut self, weight: Array2<f32>, bias: Array1<f32>) {
        self.weight = weight;
        self.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qkv_shapes() {
        let qkv = QKVProjection::new(1024, 16);
        let input = Array2::zeros((1, 2, 1024));
        let (q, k, v) = qkv.forward(&input);
        
        assert_eq!(q.shape(), &[1, 2, 16, 64]);
        assert_eq!(k.shape(), &[1, 2, 16, 64]);
        assert_eq!(v.shape(), &[1, 2, 16, 64]);
    }
}
```

**Validation:**
- [ ] Checkpoint 2 tests pass
- [ ] Q, K, V shapes correct
- [ ] Values match reference

---

### 3. cache.rs - KV Cache (Checkpoint 3)

**Purpose:** Manage KV cache for autoregressive generation

```rust
// src/layers/attention/cache.rs

use ndarray::Array3;

/// KV cache for efficient autoregressive generation
pub struct KVCache {
    k_cache: Option<Array3<f32>>,  // [batch, max_seq_len, n_heads, head_dim]
    v_cache: Option<Array3<f32>>,  // [batch, max_seq_len, n_heads, head_dim]
    n_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl KVCache {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            n_heads,
            head_dim,
            max_seq_len: 2048,  // Default max context
        }
    }
    
    /// Update cache with new K, V tensors
    /// 
    /// Returns: (k_full, v_full) containing all cached tokens
    pub fn update(
        &mut self,
        k: Array3<f32>,
        v: Array3<f32>,
        start_pos: usize,
    ) -> (Array3<f32>, Array3<f32>) {
        let batch_size = k.shape()[0];
        let seq_len = k.shape()[1];
        
        // Initialize cache if needed
        if self.k_cache.is_none() {
            self.k_cache = Some(Array3::zeros((
                batch_size,
                self.max_seq_len,
                self.n_heads,
                self.head_dim,
            )));
            self.v_cache = Some(Array3::zeros((
                batch_size,
                self.max_seq_len,
                self.n_heads,
                self.head_dim,
            )));
        }
        
        // Update cache at [start_pos:start_pos+seq_len]
        let k_cache = self.k_cache.as_mut().unwrap();
        let v_cache = self.v_cache.as_mut().unwrap();
        
        k_cache
            .slice_mut(s![.., start_pos..start_pos + seq_len, .., ..])
            .assign(&k);
        v_cache
            .slice_mut(s![.., start_pos..start_pos + seq_len, .., ..])
            .assign(&v);
        
        // Return cached values up to current position
        let k_full = k_cache.slice(s![.., ..start_pos + seq_len, .., ..]).to_owned();
        let v_full = v_cache.slice(s![.., ..start_pos + seq_len, .., ..]).to_owned();
        
        (k_full, v_full)
    }
    
    pub fn clear(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_initialization() {
        let mut cache = KVCache::new(16, 64);
        let k = Array3::zeros((1, 2, 16, 64));
        let v = Array3::zeros((1, 2, 16, 64));
        
        let (k_full, v_full) = cache.update(k, v, 0);
        
        assert_eq!(k_full.shape(), &[1, 2, 16, 64]);
        assert_eq!(v_full.shape(), &[1, 2, 16, 64]);
    }
    
    #[test]
    fn test_cache_append() {
        let mut cache = KVCache::new(16, 64);
        
        // First update
        let k1 = Array3::ones((1, 2, 16, 64));
        let v1 = Array3::ones((1, 2, 16, 64));
        cache.update(k1, v1, 0);
        
        // Second update
        let k2 = Array3::ones((1, 1, 16, 64)) * 2.0;
        let v2 = Array3::ones((1, 1, 16, 64)) * 2.0;
        let (k_full, v_full) = cache.update(k2, v2, 2);
        
        // Should contain 3 tokens now
        assert_eq!(k_full.shape(), &[1, 3, 16, 64]);
        assert_eq!(v_full.shape(), &[1, 3, 16, 64]);
    }
}
```

**Validation:**
- [ ] Checkpoint 3 tests pass
- [ ] Cache initialization correct
- [ ] Cache update correct
- [ ] Cache retrieval correct

---

### 4. scores.rs - Attention Scores (Checkpoint 4)

**Purpose:** Compute scaled dot-product attention scores

```rust
// src/layers/attention/scores.rs

use ndarray::{Array3, Array4};

/// Attention score computation
pub struct AttentionScores {
    head_dim: usize,
    scale: f32,
}

impl AttentionScores {
    pub fn new(head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { head_dim, scale }
    }
    
    /// Compute attention scores: softmax(Q @ K^T / sqrt(d_k))
    /// 
    /// Q: [batch, seq_q, n_heads, head_dim]
    /// K: [batch, seq_k, n_heads, head_dim]
    /// mask: Optional [batch, seq_q, seq_k]
    /// 
    /// Returns: [batch, n_heads, seq_q, seq_k]
    pub fn compute(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Array4<f32> {
        // 1. Transpose Q, K to [batch, n_heads, seq, head_dim]
        let q = q.permuted_axes([0, 2, 1, 3]);
        let k = k.permuted_axes([0, 2, 1, 3]);
        
        // 2. Compute Q @ K^T
        // Q: [batch, n_heads, seq_q, head_dim]
        // K^T: [batch, n_heads, head_dim, seq_k]
        let k_t = k.permuted_axes([0, 1, 3, 2]);
        let scores = q.dot(&k_t);  // [batch, n_heads, seq_q, seq_k]
        
        // 3. Scale by sqrt(head_dim)
        let scores = scores * self.scale;
        
        // 4. Apply mask (set future positions to -inf)
        let scores = if let Some(mask) = mask {
            scores + mask.broadcast((scores.shape()[0], scores.shape()[1])).unwrap()
        } else {
            scores
        };
        
        // 5. Apply softmax
        let scores = softmax(&scores, -1);
        
        scores
    }
}

fn softmax(x: &Array4<f32>, axis: isize) -> Array4<f32> {
    // Subtract max for numerical stability
    let max = x.map_axis(Axis(axis as usize), |row| {
        row.fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    });
    let x = x - &max;
    
    // Compute exp
    let exp_x = x.mapv(f32::exp);
    
    // Compute sum
    let sum = exp_x.sum_axis(Axis(axis as usize));
    
    // Divide
    exp_x / &sum
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_scores_shape() {
        let scores_comp = AttentionScores::new(64);
        let q = Array3::zeros((1, 2, 16, 64));
        let k = Array3::zeros((1, 2, 16, 64));
        
        let scores = scores_comp.compute(&q, &k, None);
        
        assert_eq!(scores.shape(), &[1, 16, 2, 2]);
    }
    
    #[test]
    fn test_softmax_sums_to_one() {
        let scores_comp = AttentionScores::new(64);
        let q = Array3::ones((1, 2, 16, 64));
        let k = Array3::ones((1, 2, 16, 64));
        
        let scores = scores_comp.compute(&q, &k, None);
        
        // Each row should sum to 1.0
        let sum = scores.sum_axis(Axis(3));
        assert!((sum - 1.0).abs().sum() < 1e-5);
    }
}
```

**Validation:**
- [ ] Checkpoint 4 tests pass
- [ ] Scores shape correct
- [ ] Scale factor correct (1/sqrt(64) = 0.125)
- [ ] Softmax sums to 1.0
- [ ] Mask applied correctly

---

### 5. output.rs - Attention Output (Checkpoint 5)

**Purpose:** Apply attention weights and project output

```rust
// src/layers/attention/output.rs

use ndarray::{Array2, Array3, Array4};

/// Attention output projection
pub struct AttentionOutput {
    weight: Array2<f32>,  // [dim, dim]
    bias: Array1<f32>,    // [dim]
    dim: usize,
}

impl AttentionOutput {
    pub fn new(dim: usize) -> Self {
        Self {
            weight: Array2::zeros((dim, dim)),
            bias: Array1::zeros(dim),
            dim,
        }
    }
    
    /// Apply attention weights to values and project output
    /// 
    /// attn_weights: [batch, n_heads, seq_q, seq_k]
    /// v: [batch, seq_k, n_heads, head_dim]
    /// 
    /// Returns: [batch, seq_q, dim]
    pub fn forward(&self, attn_weights: &Array4<f32>, v: &Array3<f32>) -> Array2<f32> {
        // 1. Transpose V to [batch, n_heads, seq_k, head_dim]
        let v = v.permuted_axes([0, 2, 1, 3]);
        
        // 2. Apply attention: attn_weights @ V
        // [batch, n_heads, seq_q, seq_k] @ [batch, n_heads, seq_k, head_dim]
        // = [batch, n_heads, seq_q, head_dim]
        let output = attn_weights.dot(&v);
        
        // 3. Transpose back to [batch, seq_q, n_heads, head_dim]
        let output = output.permuted_axes([0, 2, 1, 3]);
        
        // 4. Reshape to [batch, seq_q, dim]
        let batch = output.shape()[0];
        let seq_q = output.shape()[1];
        let output = output.into_shape((batch, seq_q, self.dim)).unwrap();
        
        // 5. Output projection: output @ weight + bias
        let output = output.dot(&self.weight) + &self.bias;
        
        output
    }
    
    pub fn load_weights(&mut self, weight: Array2<f32>, bias: Array1<f32>) {
        self.weight = weight;
        self.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_output_shape() {
        let output_proj = AttentionOutput::new(1024);
        let attn_weights = Array4::zeros((1, 16, 2, 2));
        let v = Array3::zeros((1, 2, 16, 64));
        
        let output = output_proj.forward(&attn_weights, &v);
        
        assert_eq!(output.shape(), &[1, 2, 1024]);
    }
}
```

**Validation:**
- [ ] Checkpoint 5 tests pass
- [ ] Output shape correct
- [ ] Attention applied correctly
- [ ] Projection applied correctly

---

## Benefits of This Structure

### 1. Clear Separation of Concerns
- Each file has one responsibility
- Easy to understand and maintain
- Matches checkpoint structure

### 2. Incremental Validation
- Implement qkv.rs → Test Checkpoint 2
- Implement cache.rs → Test Checkpoint 3
- Implement scores.rs → Test Checkpoint 4
- Implement output.rs → Test Checkpoint 5
- Integrate in mod.rs → Test all together

### 3. Testability
- Each component can be tested independently
- Unit tests in each file
- Integration tests in mod.rs

### 4. Reusability
- Components can be used separately if needed
- Clear interfaces between components
- Easy to swap implementations

---

## Implementation Order

1. **Week 2, Day 1:** qkv.rs (Checkpoint 2)
   - Implement QKVProjection
   - Test shapes and values
   - Validate against reference

2. **Week 2, Day 2:** cache.rs (Checkpoint 3)
   - Implement KVCache
   - Test initialization and updates
   - Validate against reference

3. **Week 2, Day 3:** scores.rs (Checkpoint 4)
   - Implement AttentionScores
   - Test computation and softmax
   - Validate against reference

4. **Week 2, Day 4:** output.rs (Checkpoint 5)
   - Implement AttentionOutput
   - Test projection
   - Validate against reference

5. **Week 2, Day 5:** mod.rs (Integration)
   - Integrate all components
   - Test complete attention
   - Validate end-to-end

---

## Testing Strategy

### Unit Tests (In Each File)
```rust
// In qkv.rs
#[test]
fn test_qkv_projection_shapes() { ... }

#[test]
fn test_qkv_projection_values() { ... }
```

### Integration Tests (In tests/)
```rust
// tests/checkpoint_02_qkv.rs
#[test]
fn checkpoint_02_matches_tinygrad() {
    // Load reference
    // Run QKVProjection
    // Compare
}
```

### End-to-End Test
```rust
// tests/attention_integration.rs
#[test]
fn test_complete_attention() {
    // Test all components together
}
```

---

## Summary

**Old Structure:**
```
src/layers/
└── attention.rs  (300+ lines, 4 checkpoints mixed)
```

**New Structure:**
```
src/layers/attention/
├── mod.rs        (Orchestration, ~100 lines)
├── qkv.rs        (Checkpoint 2, ~80 lines)
├── cache.rs      (Checkpoint 3, ~100 lines)
├── scores.rs     (Checkpoint 4, ~80 lines)
└── output.rs     (Checkpoint 5, ~70 lines)
```

**Benefits:**
- ✅ Clear separation (1 file = 1 checkpoint)
- ✅ Easier to validate incrementally
- ✅ Better testability
- ✅ More maintainable

---

Built by TEAM CASCADE 🌊
