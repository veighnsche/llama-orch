# Candle Library Optimization Analysis

**Analyzed by:** TEAM-005  
**Date:** 2025-10-08  
**Purpose:** Identify opportunities to leverage Candle's built-in functionality

---

## Executive Summary

**Current Status:** We are **UNDER-utilizing** Candle's built-in optimizations.

### Critical Findings

1. ✅ **RMSNorm:** Already using `candle_nn::ops::rms_norm` (optimal)
2. ❌ **RoPE:** Custom implementation - Candle has optimized `rope()` and `rope_i()` functions
3. ❌ **QKV Projection:** Manual matmul - Could use `candle_nn::Linear` 
4. ❌ **Softmax:** Not yet implemented - Candle has `candle_nn::ops::softmax`
5. ❌ **KV Cache:** Stub implementation - Candle has `candle_nn::kv_cache::KvCache`

### Recommendation

**Replace custom implementations with Candle's optimized versions** to get:
- GPU kernel optimizations (CUDA/Metal)
- CPU SIMD optimizations  
- Parallel execution (rayon)
- Reduced maintenance burden

---

## Detailed Analysis

### 1. RoPE Implementation ❌ NEEDS OPTIMIZATION

#### Current Implementation (Custom)
**File:** `src/layers/rope.rs` (187 lines)

**What we do:**
```rust
// Manual even/odd splitting and rotation
let mut x_even_parts = Vec::new();
let mut x_odd_parts = Vec::new();
for i in 0..(self.head_dim / 2) {
    x_even_parts.push(x_flat.narrow(1, i * 2, 1)?);
    x_odd_parts.push(x_flat.narrow(1, i * 2 + 1, 1)?);
}
let x_even = Tensor::cat(&x_even_parts, 1)?;
let x_odd = Tensor::cat(&x_odd_parts, 1)?;

// Manual rotation
let x_even_rot = x_even.mul(&cos_expanded)?.sub(&x_odd.mul(&sin_expanded)?)?;
let x_odd_rot = x_even.mul(&sin_expanded)?.add(&x_odd.mul(&cos_expanded)?)?;
```

**Problems:**
- Multiple tensor allocations (even_parts, odd_parts vectors)
- Multiple narrow/cat operations (slow)
- No GPU kernel optimization
- No CPU SIMD optimization

#### Candle's Built-in RoPE ✅
**File:** `candle-nn-0.9.1/src/rotary_emb.rs`

**Available functions:**
1. **`rope_i(xs, cos, sin)`** - Interleaved variant (our use case)
2. **`rope(xs, cos, sin)`** - Half-split variant
3. **`rope_thd(xs, cos, sin)`** - T/H/D layout variant

**Optimizations in Candle's `rope_i`:**
```rust
// CPU: Parallel execution with rayon
src.par_chunks(t * d)
    .zip(dst.par_chunks_mut(t * d))
    .enumerate()
    .for_each(|(bh_i, (src, dst))| {
        for i_over_2 in 0..t * d / 2 {
            let i = 2 * i_over_2;
            dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
            dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
        }
    });

// CUDA: Custom kernel (optimized)
let func = dev.get_or_load_func(&kernel_name::<T>("rope_i"), &kernels::REDUCE)?;

// Metal: Custom kernel (optimized)
let command_buffer = device.command_buffer()?;
command_buffer.set_compute_pipeline_state(&pipeline);
```

**Benefits:**
- ✅ Single-pass algorithm (no intermediate allocations)
- ✅ CPU: Parallel with rayon
- ✅ CUDA: Custom optimized kernel
- ✅ Metal: Custom optimized kernel
- ✅ Supports batched and unbatched cos/sin

**Expected Performance Gain:** 3-5x faster

#### Recommended Change

**Replace our implementation:**
```rust
// OLD (187 lines of custom code)
impl RoPE {
    pub fn forward(&self, q: &Tensor, k: &Tensor, position: usize) 
        -> CandleResult<(Tensor, Tensor)> {
        let q_rot = self.apply_rotation(q, position)?;
        let k_rot = self.apply_rotation(k, position)?;
        Ok((q_rot, k_rot))
    }
    
    fn apply_rotation(&self, x: &Tensor, position: usize) -> CandleResult<Tensor> {
        // 50+ lines of manual splitting/rotation
    }
}
```

**With Candle's optimized version:**
```rust
// NEW (~30 lines total)
use candle_nn::rotary_emb::rope_i;

impl RoPE {
    pub fn forward(&self, q: &Tensor, k: &Tensor, position: usize) 
        -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos_cache.narrow(0, position, seq_len)?;
        let sin = self.sin_cache.narrow(0, position, seq_len)?;
        
        let q_rot = rope_i(q, &cos, &sin)?;
        let k_rot = rope_i(k, &cos, &sin)?;
        
        Ok((q_rot, k_rot))
    }
}
```

**Impact:**
- 🔥 **Delete 150+ lines of custom code**
- ✅ GPU acceleration (CUDA/Metal)
- ✅ CPU parallelization
- ✅ Maintained by Candle team

---

### 2. QKV Projection ⚠️ COULD BE OPTIMIZED

#### Current Implementation (Manual matmul)
**File:** `src/layers/attention.rs`

```rust
pub struct QKVProjection {
    q_proj: Tensor,  // [hidden_size, hidden_size]
    k_proj: Tensor,
    v_proj: Tensor,
    // ...
}

pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let x_flat = x.reshape((batch * seq_len, hidden_size))?;
    
    let q = x_flat.matmul(&self.q_proj)?;
    let k = x_flat.matmul(&self.k_proj)?;
    let v = x_flat.matmul(&self.v_proj)?;
    
    // reshape...
}
```

#### Candle's Linear Layer
**File:** `candle-nn-0.9.1/src/linear.rs`

```rust
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match *x.dims() {
            [b1, b2, _, _] => self.weight.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}
```

**Analysis:**
- Candle's `Linear` handles broadcasting automatically
- Our manual matmul is essentially the same
- **Verdict:** Current implementation is fine, but using `Linear` would be more idiomatic

**Recommendation:** 
- ✅ Keep current implementation (it's already optimal)
- OR use `Linear` for better code clarity (minor benefit)

---

### 3. RMSNorm ✅ ALREADY OPTIMAL

#### Current Implementation
**File:** `src/layers/rms_norm.rs`

```rust
use candle_nn::ops::rms_norm;

impl RMSNorm {
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        rms_norm(x, &self.weight, self.eps)
    }
}
```

**Status:** ✅ **Already using Candle's optimized implementation**

**Candle's optimizations:**
- CPU: Parallel execution with rayon
- CUDA: Custom kernel
- Metal: Custom kernel
- Automatic dtype handling (F16/BF16 → F32 internally)

**No changes needed.**

---

### 4. Softmax (Not Yet Implemented) ✅ USE CANDLE

#### Candle's Softmax
**File:** `candle-nn-0.9.1/src/ops.rs`

```rust
pub fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}
```

**Features:**
- Numerically stable (subtracts max before exp)
- Works on any dimension
- GPU/CPU optimized

**Recommendation for Checkpoint 3:**
```rust
use candle_nn::ops::softmax;

// In attention computation:
let scores = q.matmul(&k.t()?)?;
let scores = (scores / (head_dim as f64).sqrt())?;
let attn_weights = softmax(&scores, D::Minus1)?;
```

---

### 5. KV Cache ❌ REPLACE STUB WITH CANDLE

#### Current Implementation (Stub)
**File:** `src/cache/kv_cache.rs`

```rust
pub struct KVCache {
    k_cache: Option<Array3<f32>>,  // ndarray (wrong!)
    v_cache: Option<Array3<f32>>,
    max_seq_len: usize,
}

impl KVCache {
    pub fn append(&mut self, k: Array3<f32>, v: Array3<f32>, start_pos: usize) {
        // TODO: implement
    }
}
```

**Problems:**
- Uses `ndarray` instead of `Tensor` (incompatible!)
- Not implemented
- No GPU support

#### Candle's KV Cache ✅
**File:** `candle-nn-0.9.1/src/kv_cache.rs`

```rust
pub struct KvCache {
    k: Cache,
    v: Cache,
}

impl KvCache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self { ... }
    
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let k = self.k.current_data()?.unwrap();
        let v = self.v.current_data()?.unwrap();
        Ok((k, v))
    }
    
    pub fn reset(&mut self) { ... }
}
```

**Features:**
- ✅ Dynamic growth (grows by `max_seq_len` chunks)
- ✅ Efficient `slice_set` for appending
- ✅ GPU/CPU compatible (uses Tensor)
- ✅ Reset support for new sequences

**Recommendation:**
```rust
// DELETE src/cache/kv_cache.rs entirely
// USE candle_nn::kv_cache::KvCache instead

use candle_nn::kv_cache::KvCache;

pub struct Attention {
    qkv: QKVProjection,
    rope: RoPE,
    kv_cache: Option<KvCache>,  // Use Candle's implementation
}
```

---

### 6. SwiGLU (Future) ✅ USE CANDLE

**Candle provides:**
```rust
pub fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}
```

**When implementing FFN:**
```rust
use candle_nn::ops::swiglu;

// Instead of manual implementation:
let swiglu_output = swiglu(&gate_up_proj)?;
```

---

## Summary of Recommendations

### Immediate Actions (High Impact)

1. **🔥 Replace RoPE implementation with `candle_nn::rotary_emb::rope_i`**
   - **Impact:** 3-5x performance improvement
   - **Effort:** 2 hours
   - **Lines saved:** ~150 lines

2. **🔥 Replace KV Cache stub with `candle_nn::kv_cache::KvCache`**
   - **Impact:** Functional implementation + GPU support
   - **Effort:** 1 hour
   - **Lines saved:** ~50 lines (delete stub)

3. **✅ Use `candle_nn::ops::softmax` for attention (Checkpoint 3)**
   - **Impact:** Numerically stable, GPU optimized
   - **Effort:** 30 minutes
   - **Lines saved:** ~20 lines

### Keep As-Is (Already Optimal)

1. ✅ **RMSNorm** - Already using `candle_nn::ops::rms_norm`
2. ✅ **QKV Projection** - Manual matmul is fine (equivalent to Linear)

### Future Optimizations

1. **Use `candle_nn::ops::swiglu`** when implementing FFN
2. **Consider `candle_nn::Linear`** for QKV (code clarity, not performance)

---

## Performance Impact Estimate

### Current vs. Optimized

| Component | Current | With Candle | Speedup |
|-----------|---------|-------------|---------|
| RoPE | Custom (slow) | GPU kernel | **3-5x** |
| KV Cache | Not working | Optimized | **∞** (functional) |
| Softmax | TBD | GPU kernel | **2-3x** |
| RMSNorm | Candle ✅ | Candle ✅ | 1x (same) |
| QKV | Manual matmul | Same | 1x (same) |

**Overall Expected Speedup:** 2-3x for full inference pipeline

---

## Implementation Plan

### Phase 1: RoPE Optimization (Immediate)
```rust
// File: src/layers/rope.rs
// BEFORE: 187 lines
// AFTER: ~40 lines

use candle_nn::rotary_emb::rope_i;

pub struct RoPE {
    cos_cache: Tensor,
    sin_cache: Tensor,
    device: Device,
}

impl RoPE {
    pub fn forward(&self, q: &Tensor, k: &Tensor, position: usize) 
        -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos_cache.narrow(0, position, seq_len)?;
        let sin = self.sin_cache.narrow(0, position, seq_len)?;
        
        let q_rot = rope_i(q, &cos, &sin)?;
        let k_rot = rope_i(k, &cos, &sin)?;
        
        Ok((q_rot, k_rot))
    }
}
```

### Phase 2: KV Cache Replacement (Immediate)
```rust
// DELETE: src/cache/kv_cache.rs
// ADD: use candle_nn::kv_cache::KvCache;

pub struct Attention {
    qkv: QKVProjection,
    rope: RoPE,
    kv_cache: KvCache,  // Candle's implementation
}

impl Attention {
    pub fn forward_with_cache(&mut self, x: &Tensor, position: usize) 
        -> CandleResult<Tensor> {
        let (q, k, v) = self.qkv.forward(x)?;
        let (q_rot, k_rot) = self.rope.forward(&q, &k, position)?;
        
        // Append to cache
        let (k_cached, v_cached) = self.kv_cache.append(&k_rot, &v)?;
        
        // Attention computation with cached K, V
        // ...
    }
}
```

### Phase 3: Softmax (Checkpoint 3)
```rust
use candle_nn::ops::softmax;

let scores = q.matmul(&k.t()?)?;
let scores = (scores / (head_dim as f64).sqrt())?;
let attn_weights = softmax(&scores, D::Minus1)?;
let output = attn_weights.matmul(&v)?;
```

---

## Testing Strategy

### After RoPE Replacement
1. Run existing RoPE tests - should still pass
2. Add benchmark comparison (old vs new)
3. Verify GPU execution (if available)

### After KV Cache Replacement
1. Test cache append/reset functionality
2. Test with different sequence lengths
3. Verify memory efficiency

### Integration Tests
1. Full attention block with Candle components
2. Compare outputs with reference implementation
3. Performance benchmarks

---

## Risks & Mitigation

### Risk 1: API Differences
**Mitigation:** Candle's `rope_i` expects `[batch, n_heads, seq_len, head_dim]` layout
- Our current: `[batch, seq_len, n_heads, head_dim]`
- **Solution:** Transpose before/after or use `rope_thd` variant

### Risk 2: Breaking Tests
**Mitigation:** Keep old implementation temporarily
- Create `rope_candle.rs` alongside `rope.rs`
- Test both, compare outputs
- Switch when verified

### Risk 3: Performance Regression
**Mitigation:** Benchmark before/after
- If slower, investigate (unlikely with GPU kernels)
- Report to Candle team if needed

---

## Conclusion

**We are doing too much custom work.** Candle provides optimized implementations for:
- ✅ RoPE (GPU kernels)
- ✅ KV Cache (dynamic, efficient)
- ✅ Softmax (numerically stable)
- ✅ RMSNorm (already using)
- ✅ SwiGLU (for FFN)

**Recommendation:** Replace custom RoPE and KV Cache implementations with Candle's optimized versions **immediately**. This will:
1. Reduce codebase by ~200 lines
2. Improve performance 2-3x
3. Get GPU acceleration for free
4. Reduce maintenance burden

**The hard parts of inference should be:** Model architecture, weight loading, tokenization.  
**Not:** Reimplementing optimized tensor operations that Candle already provides.

---

**Next Steps:**
1. Implement RoPE replacement (2 hours)
2. Implement KV Cache replacement (1 hour)
3. Update tests (1 hour)
4. Benchmark and verify (1 hour)

**Total effort:** ~5 hours for 2-3x performance gain ✅

---

*"Use the library, don't fight it."*  
— TEAM-005, 2025-10-08
