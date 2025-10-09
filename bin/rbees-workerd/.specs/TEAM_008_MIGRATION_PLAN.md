# TEAM-008 MIGRATION & OPTIMIZATION PLAN

**Team:** TEAM-008  
**Date:** 2025-10-08T22:36:29+02:00  
**Mission:** Fix state fragmentation and complete backend implementation  
**Status:** 📋 PLANNING COMPLETE → 🚀 READY FOR EXECUTION

---

## Executive Summary

This plan addresses **PRIORITY 1** from the Outstanding Work Checklist: fixing state fragmentation to align with Candle's design patterns, followed by completing the backend implementation for functional inference.

### Key Objectives

1. **Phase 1: Unified Cache (5-6 hours)** - Fix state fragmentation
2. **Phase 2: Model Loading (6-8 hours)** - GGUF/SafeTensors support
3. **Phase 3: Generation Loop (4-6 hours)** - Token-by-token inference
4. **Phase 4: Streaming (3-4 hours)** - SSE/JSONL output
5. **Phase 5: Validation (2-3 hours)** - End-to-end testing

**Total Estimate:** 20-27 hours

---

## Phase 1: Unified Cache Architecture (5-6 hours)

### Problem Statement

**Current fragmentation:**
- `RoPE.cos_cache`, `RoPE.sin_cache` (in `layers/rope.rs`)
- `KvCache` (in `cache/kv_cache.rs`)
- `Attention.mask_cache` (in `layers/attention.rs`)

**Candle's pattern:**
```rust
pub struct Cache {
    kvs: Vec<Option<(Tensor, Tensor)>>,  // Per-layer KV
    cos: Tensor,                          // RoPE cos (shared)
    sin: Tensor,                          // RoPE sin (shared)
    masks: HashMap<usize, Tensor>,        // Causal masks (cached)
    device: Device,
}
```

### Implementation Steps

#### 1.1 Create Unified Cache Struct (1.5 hours)

**File:** `src/cache/unified_cache.rs`

```rust
//! Unified cache for all generation state
//!
//! Aligns with Candle's design pattern: single source of truth for:
//! - KV cache (per-layer)
//! - RoPE precomputed cos/sin
//! - Causal attention masks
//!
//! Created by: TEAM-008

use candle_core::{Tensor, Result as CandleResult, Device};
use candle_nn::kv_cache::KvCache;
use std::collections::HashMap;

/// Unified cache for all generation state
///
/// Single source of truth for:
/// - KV cache (per-layer, for attention)
/// - RoPE cos/sin (shared across layers)
/// - Causal masks (cached by sequence length)
pub struct Cache {
    /// Per-layer KV cache
    kv_caches: Vec<KvCache>,
    
    /// RoPE precomputed cosines [max_seq_len, head_dim/2]
    rope_cos: Tensor,
    
    /// RoPE precomputed sines [max_seq_len, head_dim/2]
    rope_sin: Tensor,
    
    /// Cached causal masks by sequence length
    causal_masks: HashMap<usize, Tensor>,
    
    /// Device for tensor operations
    device: Device,
    
    /// Configuration
    max_seq_len: usize,
    n_layers: usize,
}

impl Cache {
    /// Create new cache with precomputed RoPE values
    ///
    /// # Arguments
    /// * `n_layers` - Number of transformer layers
    /// * `head_dim` - Dimension per attention head
    /// * `max_seq_len` - Maximum sequence length
    /// * `rope_theta` - RoPE base frequency (10000.0 for Llama-2)
    /// * `device` - Device to place tensors on
    pub fn new(
        n_layers: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f32,
        device: &Device,
    ) -> CandleResult<Self> {
        // Initialize per-layer KV caches
        let kv_caches = (0..n_layers)
            .map(|_| KvCache::new(2, max_seq_len))
            .collect();
        
        // Precompute RoPE cos/sin
        let dim_pairs = head_dim / 2;
        let freqs: Vec<f32> = (0..dim_pairs)
            .map(|i| rope_theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
            .collect();
        
        let mut cos_values = Vec::with_capacity(max_seq_len * dim_pairs);
        let mut sin_values = Vec::with_capacity(max_seq_len * dim_pairs);
        
        for pos in 0..max_seq_len {
            for &freq in &freqs {
                let angle = (pos as f32) * freq;
                cos_values.push(angle.cos());
                sin_values.push(angle.sin());
            }
        }
        
        let rope_cos = Tensor::from_vec(cos_values, (max_seq_len, dim_pairs), device)?;
        let rope_sin = Tensor::from_vec(sin_values, (max_seq_len, dim_pairs), device)?;
        
        Ok(Self {
            kv_caches,
            rope_cos,
            rope_sin,
            causal_masks: HashMap::new(),
            device: device.clone(),
            max_seq_len,
            n_layers,
        })
    }
    
    /// Get KV cache for specific layer
    pub fn kv_cache(&mut self, layer_idx: usize) -> &mut KvCache {
        &mut self.kv_caches[layer_idx]
    }
    
    /// Get RoPE cos/sin for position range
    ///
    /// # Arguments
    /// * `position` - Starting position
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// * `(cos, sin)` - Tensors of shape [seq_len, head_dim/2]
    pub fn rope_values(&self, position: usize, seq_len: usize) -> CandleResult<(Tensor, Tensor)> {
        let cos = self.rope_cos.narrow(0, position, seq_len)?;
        let sin = self.rope_sin.narrow(0, position, seq_len)?;
        Ok((cos, sin))
    }
    
    /// Get cached causal mask for sequence length
    ///
    /// Creates and caches mask if not already present
    pub fn causal_mask(&mut self, seq_len: usize) -> CandleResult<&Tensor> {
        if !self.causal_masks.contains_key(&seq_len) {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &self.device)?;
            self.causal_masks.insert(seq_len, mask);
        }
        Ok(self.causal_masks.get(&seq_len).unwrap())
    }
    
    /// Reset all caches for new generation
    pub fn reset(&mut self) {
        for kv in &mut self.kv_caches {
            *kv = KvCache::new(2, self.max_seq_len);
        }
        // Keep RoPE and mask caches (they're reusable)
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_creation() -> CandleResult<()> {
        let device = Device::Cpu;
        let cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
        
        assert_eq!(cache.n_layers, 32);
        assert_eq!(cache.max_seq_len, 4096);
        assert_eq!(cache.rope_cos.dims(), &[4096, 64]); // head_dim/2 = 64
        assert_eq!(cache.rope_sin.dims(), &[4096, 64]);
        
        Ok(())
    }
    
    #[test]
    fn test_rope_values() -> CandleResult<()> {
        let device = Device::Cpu;
        let cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
        
        let (cos, sin) = cache.rope_values(0, 10)?;
        assert_eq!(cos.dims(), &[10, 64]);
        assert_eq!(sin.dims(), &[10, 64]);
        
        Ok(())
    }
    
    #[test]
    fn test_causal_mask_caching() -> CandleResult<()> {
        let device = Device::Cpu;
        let mut cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
        
        // First call creates mask
        let mask1 = cache.causal_mask(10)?;
        assert_eq!(mask1.dims(), &[10, 10]);
        
        // Second call returns cached mask
        let mask2 = cache.causal_mask(10)?;
        assert_eq!(mask2.dims(), &[10, 10]);
        
        Ok(())
    }
}
```

**Tasks:**
- [ ] Create `src/cache/unified_cache.rs`
- [ ] Implement `Cache::new()` with RoPE precomputation
- [ ] Implement `kv_cache()`, `rope_values()`, `causal_mask()` accessors
- [ ] Implement `reset()` method
- [ ] Add unit tests

#### 1.2 Update RoPE to Use Shared Cache (1 hour)

**File:** `src/layers/rope.rs`

**Changes:**
```rust
// REMOVE: cos_cache, sin_cache fields from RoPE struct
// REMOVE: RoPE::new() (no longer needed)

// ADD: Standalone function that uses Cache
/// Apply RoPE rotation using shared cache
///
/// # Arguments
/// * `q` - Query tensor [batch, seq_len, n_heads, head_dim]
/// * `k` - Key tensor [batch, seq_len, n_heads, head_dim]
/// * `position` - Starting position in sequence
/// * `cache` - Shared cache containing RoPE cos/sin
///
/// # Returns
/// * `(q_rotated, k_rotated)` - Rotated tensors
///
/// TEAM-008: Refactored to use unified cache
pub fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    position: usize,
    cache: &Cache,
) -> CandleResult<(Tensor, Tensor)> {
    let seq_len = q.dim(1)?;
    let (cos, sin) = cache.rope_values(position, seq_len)?;
    
    // Transpose to [batch, n_heads, seq_len, head_dim]
    let q_t = q.transpose(1, 2)?.contiguous()?;
    let k_t = k.transpose(1, 2)?.contiguous()?;
    
    // Apply RoPE using Candle's optimized implementation
    let q_rot = rope_i(&q_t, &cos, &sin)?;
    let k_rot = rope_i(&k_t, &cos, &sin)?;
    
    // Transpose back
    let q_rot = q_rot.transpose(1, 2)?.contiguous()?;
    let k_rot = k_rot.transpose(1, 2)?.contiguous()?;
    
    Ok((q_rot, k_rot))
}
```

**Tasks:**
- [ ] Remove `RoPE` struct
- [ ] Create standalone `apply_rope()` function
- [ ] Update tests to use `Cache`
- [ ] Verify all tests pass

#### 1.3 Update Attention to Use Shared Cache (1.5 hours)

**File:** `src/layers/attention.rs`

**Changes:**
```rust
// REMOVE: mask_cache field from Attention struct
// REMOVE: get_mask() method

// UPDATE: apply_causal_mask() to accept cache
pub fn apply_causal_mask(scores: &Tensor, cache: &mut Cache) -> CandleResult<Tensor> {
    let (_batch, _n_heads, seq_q, seq_k) = scores.dims4()?;
    
    if seq_q == seq_k {
        let mask = cache.causal_mask(seq_q)?;
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
        let mask = mask.broadcast_as(scores.shape())?;
        scores.broadcast_add(&mask)
    } else {
        Ok(scores.clone())
    }
}

// UPDATE: forward() to accept cache
pub fn forward(
    &self,  // Changed from &mut self
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    use_causal_mask: bool,
    cache: &mut Cache,
) -> CandleResult<Tensor> {
    let mut scores = self.compute_scores(q, k)?;
    
    if use_causal_mask {
        scores = Self::apply_causal_mask(&scores, cache)?;
    }
    
    // ... rest unchanged
}
```

**Tasks:**
- [ ] Remove `mask_cache` field
- [ ] Update `apply_causal_mask()` to use `Cache`
- [ ] Update `forward()` signature
- [ ] Update tests
- [ ] Verify all tests pass

#### 1.4 Update Cache Module Exports (0.5 hours)

**File:** `src/cache/mod.rs`

```rust
//! Cache module - Unified state management
//!
//! Aligns with Candle's design pattern: single source of truth for all generation state
//!
//! Created by: TEAM-000
//! Modified by: TEAM-008 (Unified cache architecture)

mod kv_cache;
mod unified_cache;

// Re-export unified cache (primary API)
pub use unified_cache::Cache;

// Re-export Candle's KV cache for internal use
pub use kv_cache::KvCache;
```

**Tasks:**
- [ ] Update `mod.rs` exports
- [ ] Update `lib.rs` exports
- [ ] Verify no broken imports

#### 1.5 Integration Testing (1.5 hours)

**File:** `tests/unified_cache_integration.rs`

```rust
//! Integration tests for unified cache architecture
//!
//! Validates that RoPE and Attention work correctly with shared cache
//!
//! Created by: TEAM-008

use llorch_candled::cache::Cache;
use llorch_candled::layers::{rope::apply_rope, attention::Attention};
use candle_core::{Tensor, Device};

#[test]
fn test_rope_with_unified_cache() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
    
    let q = Tensor::randn(0f32, 1.0, (1, 10, 32, 128), &device)?;
    let k = Tensor::randn(0f32, 1.0, (1, 10, 32, 128), &device)?;
    
    let (q_rot, k_rot) = apply_rope(&q, &k, 0, &cache)?;
    
    assert_eq!(q_rot.dims(), q.dims());
    assert_eq!(k_rot.dims(), k.dims());
    
    Ok(())
}

#[test]
fn test_attention_with_unified_cache() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let mut cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
    
    let hidden_size = 4096;
    let n_heads = 32;
    
    // Create attention layer
    let q_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device)?;
    
    let attention = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    // Forward pass with cache
    let input = Tensor::randn(0f32, 1.0, (1, 10, hidden_size), &device)?;
    let (q, k, v) = attention.qkv.forward(&input)?;
    
    let output = attention.forward(&q, &k, &v, true, &mut cache)?;
    
    assert_eq!(output.dims(), &[1, 10, hidden_size]);
    
    Ok(())
}

#[test]
fn test_cache_reset() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let mut cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
    
    // Use cache
    let _ = cache.causal_mask(10)?;
    
    // Reset should clear KV but keep RoPE/masks
    cache.reset();
    
    // Should still work
    let (cos, sin) = cache.rope_values(0, 10)?;
    assert_eq!(cos.dims(), &[10, 64]);
    
    Ok(())
}
```

**Tasks:**
- [ ] Create integration tests
- [ ] Test RoPE with unified cache
- [ ] Test Attention with unified cache
- [ ] Test cache reset behavior
- [ ] Run full test suite

---

## Phase 2: Integrate worker-crates (4-6 hours)

### 2.1 Use worker-gguf for Model Metadata (1 hour)

**File:** `src/backend/candle_backend.rs`

**Tasks:**
- [ ] Import `worker_gguf::GGUFMetadata`
- [ ] Parse model config from GGUF file
- [ ] Extract: vocab_size, hidden_dim, num_layers, num_heads, rope_freq_base
- [ ] Store config in backend
- [ ] Add unit tests

### 2.2 Use worker-models for Architecture Detection (1 hour)

**File:** `src/backend/candle_backend.rs`

**Tasks:**
- [ ] Import `worker_models::ModelFactory`
- [ ] Detect architecture from GGUF metadata
- [ ] Validate Llama-2 architecture
- [ ] Add error handling for unsupported architectures

### 2.3 Load Weights into Candle Tensors (2-3 hours)

**File:** `src/model/llama2.rs`

**Tasks:**
- [ ] Use `worker_gguf::GGUFMetadata::parse_tensors()` to get tensor metadata
- [ ] Load f32 tensors from GGUF into Candle `Tensor`
- [ ] Map GGUF tensor names to model layer names
- [ ] Initialize layers with loaded weights
- [ ] Initialize unified `Cache` with config from GGUF
- [ ] Add unit tests

### 2.4 Use worker-tokenizer (1 hour)

**File:** `src/backend/candle_backend.rs`

**Tasks:**
- [ ] Import `worker_tokenizer::Tokenizer`
- [ ] Initialize tokenizer from GGUF metadata
- [ ] Integrate encode/decode for generation
- [ ] Add unit tests

---

## Phase 3: Generation Loop (4-6 hours)

### 3.1 Sampling Implementation (2 hours)

**File:** `src/generation/sampling.rs`

**Tasks:**
- [ ] Implement temperature sampling
- [ ] Implement top-p (nucleus) sampling
- [ ] Implement top-k sampling
- [ ] Add greedy decoding
- [ ] Add unit tests

### 3.2 Generation Loop (2-3 hours)

**File:** `src/generation/generate.rs`

**Tasks:**
- [ ] Implement token-by-token generation
- [ ] KV cache management
- [ ] Stop conditions (EOS, max_tokens)
- [ ] Position tracking
- [ ] Add integration tests

### 3.3 Backend Integration (1 hour)

**File:** `src/backend.rs`

**Tasks:**
- [ ] Update `CandleInferenceBackend::load()` to use new loader
- [ ] Implement `generate()` method
- [ ] Wire up sampling config
- [ ] Add error handling

---

## Phase 4: Streaming (3-4 hours)

### 4.1 SSE Streaming (1.5 hours)

**File:** `src/streaming/sse.rs`

**Tasks:**
- [ ] Implement SSE event formatting
- [ ] Token-by-token streaming
- [ ] Error handling
- [ ] Add tests

### 4.2 JSONL Streaming (1 hour)

**File:** `src/streaming/jsonl.rs`

**Tasks:**
- [ ] Implement JSONL formatting
- [ ] Token-by-token streaming
- [ ] Add tests

### 4.3 HTTP Integration (0.5-1 hour)

**File:** Update `worker-http` integration

**Tasks:**
- [ ] Wire streaming to HTTP endpoints
- [ ] Add streaming tests
- [ ] Verify with real requests

---

## Phase 5: Validation (2-3 hours)

### 5.1 End-to-End Testing (1.5 hours)

**Tasks:**
- [ ] Download test model (TinyLlama or similar)
- [ ] Test full inference pipeline
- [ ] Verify output quality
- [ ] Test streaming
- [ ] Test different sampling configs

### 5.2 Performance Validation (0.5-1 hour)

**Tasks:**
- [ ] Benchmark token generation speed
- [ ] Measure memory usage
- [ ] Compare CPU vs CUDA (if available)
- [ ] Document baseline performance

### 5.3 Documentation (0.5-1 hour)

**Tasks:**
- [ ] Update README with usage examples
- [ ] Document unified cache architecture
- [ ] Add model loading guide
- [ ] Update handoff document

---

## Success Criteria

### Phase 1 (Unified Cache)
- ✅ All existing tests pass
- ✅ No performance regression
- ✅ State management cleaner and aligned with Candle
- ✅ Integration tests validate cache sharing

### Phase 2 (Model Loading)
- ✅ Can load GGUF files
- ✅ Can load SafeTensors files
- ✅ Weights correctly placed on device
- ✅ Model initializes without errors

### Phase 3 (Generation)
- ✅ Token-by-token generation works
- ✅ KV cache correctly updated
- ✅ Sampling produces reasonable outputs
- ✅ Stop conditions work correctly

### Phase 4 (Streaming)
- ✅ SSE streaming works
- ✅ JSONL streaming works
- ✅ HTTP endpoints return streamed responses
- ✅ No memory leaks

### Phase 5 (Validation)
- ✅ End-to-end inference produces coherent text
- ✅ Performance meets baseline expectations
- ✅ Documentation complete
- ✅ Ready for production testing

---

## Risk Mitigation

### Risk 1: Breaking Existing Tests
**Mitigation:** Run tests after each sub-phase, fix immediately

### Risk 2: Performance Regression
**Mitigation:** Benchmark before/after Phase 1, revert if >5% slower

### Risk 3: GGUF Quantization Complexity
**Mitigation:** Start with unquantized models, add quantization incrementally

### Risk 4: Memory Issues with Large Models
**Mitigation:** Test with small models first (TinyLlama 1.1B), scale up gradually

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Unified Cache | 5-6 hours | None |
| Phase 2: Model Loading | 6-8 hours | Phase 1 complete |
| Phase 3: Generation Loop | 4-6 hours | Phase 2 complete |
| Phase 4: Streaming | 3-4 hours | Phase 3 complete |
| Phase 5: Validation | 2-3 hours | Phase 4 complete |
| **Total** | **20-27 hours** | Sequential execution |

---

## Notes

### Why Phase 1 First?

TEAM-005 was correct: state fragmentation is a real issue. Fixing it first:
1. Aligns with Candle's design patterns
2. Makes subsequent phases easier
3. Prevents future refactoring pain
4. Is NOT a full refactor (just state management)

### Why Not Full Refactor?

TEAM-006 was correct: don't break what works. We're keeping:
- ✅ Modular file structure
- ✅ Existing layer implementations
- ✅ Test coverage
- ✅ Documentation

We're only changing:
- ❌ State ownership (unified cache)
- ❌ Function signatures (pass cache)

### Alignment with Memories

- ✅ Follows candled-rules.md (team signatures, no background jobs)
- ✅ Respects destructive-actions.md (v0.1.0 allows cleanup)
- ✅ Aligns with proof-bundle standard (will add in future)
- ✅ Follows spec-driven approach (README_LLM.md)

---

## TEAM-008 Commitment

I will:
1. ✅ Execute this plan sequentially (no skipping phases)
2. ✅ Add team signatures to all modified code
3. ✅ Run tests after each sub-phase
4. ✅ Document any deviations from plan
5. ✅ Create honest handoff for next team

I will NOT:
1. ❌ Skip Phase 1 (state fragmentation must be fixed)
2. ❌ Break existing tests
3. ❌ Create new infrastructure without completing core functionality
4. ❌ Optimize without measuring

---

**TEAM-008 signing on.**

*"Measure twice, cut once. Plan first, execute second."*  
— TEAM-008, 2025-10-08T22:36:29+02:00

**END PLAN**
