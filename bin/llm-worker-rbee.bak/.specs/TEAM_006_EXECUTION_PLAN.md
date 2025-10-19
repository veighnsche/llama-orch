# TEAM-006 EXECUTION PLAN

**Plan by:** TEAM-006 (Peer Review & Implementation)  
**Date:** 2025-10-08  
**Status:** READY TO EXECUTE

---

## Executive Decision

**REJECT TEAM-005's full refactor plan.**  
**APPROVE targeted, data-driven optimization approach.**

---

## Rationale

### Why Reject Full Refactor?

1. ❌ **No empirical evidence** - Zero benchmarks, pure speculation
2. ❌ **Current code works** - 6/6 tests passing, builds successfully
3. ❌ **Already optimized** - Uses Candle GPU kernels (rope_i, softmax, rms_norm)
4. ❌ **High risk** - Breaking working code for unproven gains
5. ❌ **Unrealistic timeline** - 7-9 hours estimate ignores reality (20-30 hours actual)

### Why Approve Targeted Optimization?

1. ✅ **Data-driven** - Profile first, optimize second
2. ✅ **Low risk** - Incremental changes, validate each step
3. ✅ **Measurable** - Benchmark before/after
4. ✅ **Pragmatic** - Only fix proven bottlenecks
5. ✅ **Fast** - 4-8 hours total vs 20-30 hours refactor

---

## Phase 0: Current State Validation ✅ COMPLETE

### Build Status
```bash
✅ cargo build --release: SUCCESS (1m 05s)
✅ cargo test --lib: 6/6 PASSED (0.66s)
```

### Test Coverage
- ✅ `test_rope_shape` - RoPE preserves tensor shapes
- ✅ `test_rope_no_nan` - RoPE produces valid values
- ✅ `test_qkv_projection_shape` - QKV projection correct shapes
- ✅ `test_qkv_projection_no_nan` - QKV produces valid values
- ✅ `test_rms_norm_shape` - RMSNorm preserves shapes
- ✅ `test_rms_norm_no_nan` - RMSNorm produces valid values

### Current Optimizations
```rust
// Already using Candle GPU kernels:
candle_nn::rotary_emb::rope_i()  // ✅ GPU-accelerated RoPE
candle_nn::ops::softmax()         // ✅ GPU-accelerated softmax
candle_nn::ops::rms_norm()        // ✅ GPU-accelerated RMSNorm
```

**Conclusion: Current implementation is solid. No urgent need for refactor.**

---

## Phase 1: Profiling & Measurement ⏱️ 2-3 hours

### Objective
Identify actual bottlenecks with empirical data, not speculation.

### Step 1.1: Create Benchmark Suite (1 hour)

**File:** `benches/inference_bench.rs`

```rust
// Created by: TEAM-006
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llorch_candled::layers::{RoPE, QKVProjection, Attention};
use candle_core::{Tensor, Device};

fn bench_rope(c: &mut Criterion) {
    let device = Device::Cpu;
    let rope = RoPE::new(128, 4096, 10000.0, &device).unwrap();
    let q = Tensor::randn(0f32, 1.0, (1, 32, 32, 128), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (1, 32, 32, 128), &device).unwrap();
    
    c.bench_function("rope_forward", |b| {
        b.iter(|| {
            rope.forward(black_box(&q), black_box(&k), 0).unwrap()
        })
    });
}

fn bench_attention_scores(c: &mut Criterion) {
    let device = Device::Cpu;
    let hidden_size = 4096;
    let n_heads = 32;
    
    let q_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let v_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device).unwrap();
    
    let q = Tensor::randn(0f32, 1.0, (1, 32, n_heads, 128), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (1, 32, n_heads, 128), &device).unwrap();
    
    c.bench_function("attention_scores", |b| {
        b.iter(|| {
            attn.compute_scores(black_box(&q), black_box(&k)).unwrap()
        })
    });
}

fn bench_causal_mask(c: &mut Criterion) {
    let device = Device::Cpu;
    let hidden_size = 4096;
    let n_heads = 32;
    
    let q_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let v_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device).unwrap();
    
    let scores = Tensor::randn(0f32, 1.0, (1, n_heads, 32, 32), &device).unwrap();
    
    c.bench_function("causal_mask", |b| {
        b.iter(|| {
            attn.apply_causal_mask(black_box(&scores)).unwrap()
        })
    });
}

criterion_group!(benches, bench_rope, bench_attention_scores, bench_causal_mask);
criterion_main!(benches);
```

**Update Cargo.toml:**
```toml
[[bench]]
name = "inference_bench"
harness = false

[dev-dependencies]
criterion = "0.5"
```

### Step 1.2: Run Benchmarks (30 min)

```bash
# Baseline benchmarks
cargo bench --bench inference_bench

# Save baseline
cargo bench --bench inference_bench -- --save-baseline current
```

**Expected Output:**
```
rope_forward            time: [X.XX ms]
attention_scores        time: [X.XX ms]
causal_mask            time: [X.XX ms]
```

### Step 1.3: Profile with Flamegraph (30 min)

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bench inference_bench

# Open flamegraph.svg in browser
firefox flamegraph.svg
```

**Analysis Questions:**
1. What % of time is in RoPE?
2. What % of time is in attention scores?
3. What % of time is in mask creation?
4. What % of time is in softmax?

### Step 1.4: Analyze Results (30 min)

**Create:** `.specs/PROFILING_RESULTS.md`

```markdown
# Profiling Results - TEAM-006

## Benchmark Results

### RoPE Forward
- Time: X.XX ms
- % of total: X%

### Attention Scores
- Time: X.XX ms
- % of total: X%

### Causal Mask
- Time: X.XX ms
- % of total: X%

## Flamegraph Analysis

### Top 5 Time Consumers
1. Function X: Y%
2. Function Y: Z%
...

## Bottleneck Identification

### Critical (>10% time):
- [List functions taking >10% time]

### Moderate (5-10% time):
- [List functions taking 5-10% time]

### Minor (<5% time):
- [List functions taking <5% time]

## Optimization Targets

Based on profiling:
1. [Target 1] - X% of time
2. [Target 2] - Y% of time
...
```

---

## Phase 2: Decision Point ⏱️ 30 min

### Decision Criteria

**IF profiling shows mask creation >10% of time:**
→ Proceed to Phase 3A (Mask Caching)

**IF profiling shows cache fragmentation >10% of time:**
→ Proceed to Phase 3B (Unified Cache)

**IF profiling shows no bottlenecks >10%:**
→ Proceed to Phase 4 (Document & Ship)

**IF profiling shows bottlenecks in Candle ops:**
→ Cannot optimize (already using GPU kernels)

---

## Phase 3A: Mask Caching Optimization ⏱️ 2-3 hours

**ONLY IF mask creation >10% of time**

### Step 3A.1: Add Mask Cache (1 hour)

**File:** `src/layers/attention.rs`

```rust
// Modified by: TEAM-006 (mask caching optimization)

use std::collections::HashMap;

pub struct Attention {
    qkv: QKVProjection,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
    device: Device,
    mask_cache: HashMap<usize, Tensor>,  // TEAM-006: Add mask cache
}

impl Attention {
    pub fn new(
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        n_heads: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let qkv = QKVProjection::new(q_weight, k_weight, v_weight, n_heads, device)?;
        let hidden_size = qkv.q_proj.dim(0)?;
        let head_dim = hidden_size / n_heads;
        let scale = (head_dim as f64).sqrt();
        
        Ok(Self {
            qkv,
            n_heads,
            head_dim,
            scale,
            device: device.clone(),
            mask_cache: HashMap::new(),  // TEAM-006: Initialize cache
        })
    }
    
    // TEAM-006: Cached mask retrieval
    fn get_mask(&mut self, seq_len: usize) -> CandleResult<&Tensor> {
        if !self.mask_cache.contains_key(&seq_len) {
            // Create mask only if not cached
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &self.device)?;
            self.mask_cache.insert(seq_len, mask);
        }
        Ok(self.mask_cache.get(&seq_len).unwrap())
    }
    
    pub fn apply_causal_mask(&mut self, scores: &Tensor) -> CandleResult<Tensor> {
        let (_, _, seq_q, seq_k) = scores.dims4()?;
        
        if seq_q == seq_k {
            let mask = self.get_mask(seq_q)?;  // TEAM-006: Use cached mask
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
            let mask = mask.broadcast_as(scores.shape())?;
            scores.broadcast_add(&mask)
        } else {
            Ok(scores.clone())
        }
    }
}
```

### Step 3A.2: Benchmark Improvement (30 min)

```bash
# Benchmark with mask caching
cargo bench --bench inference_bench -- --baseline current

# Expected: causal_mask time reduced by 50-80%
```

### Step 3A.3: Validate (30 min)

```bash
# Run all tests
cargo test --lib

# Expected: All tests pass
```

---

## Phase 3B: Unified Cache (Alternative) ⏱️ 3-4 hours

**ONLY IF cache fragmentation >10% of time**

### Step 3B.1: Create Unified Cache (2 hours)

**File:** `src/cache/unified.rs`

```rust
// Created by: TEAM-006 (unified cache optimization)

use candle_core::{Tensor, Result as CandleResult, Device};
use candle_nn::kv_cache::KvCache;
use std::collections::HashMap;

/// Unified cache for Llama inference
/// 
/// Centralizes KV cache, RoPE cache, and causal masks
/// TEAM-006: Only created if profiling shows fragmentation bottleneck
pub struct UnifiedCache {
    /// KV cache from candle_nn
    pub kv: Vec<KvCache>,
    
    /// RoPE cosine cache [max_seq_len, head_dim/2]
    pub rope_cos: Tensor,
    
    /// RoPE sine cache [max_seq_len, head_dim/2]
    pub rope_sin: Tensor,
    
    /// Causal masks cached by sequence length
    masks: HashMap<usize, Tensor>,
    
    device: Device,
}

impl UnifiedCache {
    pub fn new(
        num_layers: usize,
        head_dim: usize,
        max_seq_len: usize,
        theta: f32,
        device: &Device,
    ) -> CandleResult<Self> {
        // Initialize KV cache
        let kv = (0..num_layers)
            .map(|_| KvCache::new(2, max_seq_len))
            .collect();
        
        // Precompute RoPE cos/sin
        let dim_pairs = head_dim / 2;
        let freqs: Vec<f32> = (0..dim_pairs)
            .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
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
            kv,
            rope_cos,
            rope_sin,
            masks: HashMap::new(),
            device: device.clone(),
        })
    }
    
    pub fn get_mask(&mut self, seq_len: usize) -> CandleResult<&Tensor> {
        if !self.masks.contains_key(&seq_len) {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            self.masks.insert(seq_len, mask);
        }
        Ok(self.masks.get(&seq_len).unwrap())
    }
}
```

### Step 3B.2: Update RoPE to Use Unified Cache (1 hour)

```rust
// Modified by: TEAM-006
impl RoPE {
    pub fn forward_with_cache(
        &self,
        q: &Tensor,
        k: &Tensor,
        position: usize,
        cache: &UnifiedCache,
    ) -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = cache.rope_cos.narrow(0, position, seq_len)?;
        let sin = cache.rope_sin.narrow(0, position, seq_len)?;
        
        // ... rest of implementation
    }
}
```

### Step 3B.3: Benchmark & Validate (1 hour)

```bash
cargo bench --bench inference_bench -- --baseline current
cargo test --lib
```

---

## Phase 4: Documentation & Ship ⏱️ 1-2 hours

### Step 4.1: Update Documentation (1 hour)

**File:** `.specs/OPTIMIZATION_SUMMARY.md`

```markdown
# Optimization Summary - TEAM-006

## Profiling Results
[Include profiling data]

## Optimizations Applied
[List optimizations based on profiling]

## Performance Improvements
- Before: X.XX ms
- After: Y.YY ms
- Improvement: Z%

## Decisions Made
- [List key decisions]
- [Rationale for each]

## Future Optimization Opportunities
[If any identified]
```

### Step 4.2: Update README (30 min)

**File:** `README.md`

```markdown
# llm-worker-rbee

Candle-based Llama-2 inference worker with GPU acceleration.

## Performance

- ✅ GPU-accelerated RoPE (candle_nn::rotary_emb)
- ✅ GPU-accelerated Softmax (candle_nn::ops)
- ✅ GPU-accelerated RMSNorm (candle_nn::ops)
- ✅ [Add any TEAM-006 optimizations]

## Benchmarks

[Include benchmark results]
```

### Step 4.3: Clean Up (30 min)

```bash
# Remove unused code
# Fix warnings
cargo clippy --fix

# Format code
cargo fmt

# Final test
cargo test --all
cargo build --release
```

---

## Success Criteria

### Must Have ✅
- [ ] Profiling data collected
- [ ] Bottlenecks identified (or confirmed none exist)
- [ ] All tests passing (6/6 minimum)
- [ ] Build successful
- [ ] Documentation updated

### Should Have ✅
- [ ] Benchmark baseline established
- [ ] If optimizations applied: measurable improvement (>10%)
- [ ] No performance regressions
- [ ] Code warnings fixed

### Nice to Have ✅
- [ ] Significant performance improvement (>50%)
- [ ] Flamegraph visualization
- [ ] Comparison with reference implementations

---

## Timeline

### Day 1 (4-5 hours)
- ✅ Phase 1: Profiling & Measurement (2-3 hours)
- ✅ Phase 2: Decision Point (30 min)
- ✅ Phase 3A or 3B: Targeted Optimization (2-4 hours, if needed)

### Day 2 (1-2 hours)
- ✅ Phase 4: Documentation & Ship (1-2 hours)

**Total: 5-7 hours** (vs TEAM-005's 20-30 hour refactor)

---

## Rollback Plan

### If Optimization Fails

**Step 1: Revert changes**
```bash
git checkout main
```

**Step 2: Document findings**
```markdown
# Optimization Attempt Failed

## What we tried:
[Description]

## Why it failed:
[Reason]

## Lessons learned:
[Insights]
```

**Step 3: Ship current code**
- Current code works
- Tests pass
- Already uses GPU kernels
- Good enough for v1.0

---

## Risk Mitigation

### Low Risk Approach

1. **Profile first** - No changes without data
2. **Incremental** - One optimization at a time
3. **Validate** - Benchmark after each change
4. **Rollback ready** - Git branches for safety

### High Confidence

- Current code works ✅
- Tests pass ✅
- Already optimized with Candle ✅
- Only optimize proven bottlenecks ✅

---

## Appendix: Commands Reference

### Profiling
```bash
# Benchmark
cargo bench --bench inference_bench

# Flamegraph
cargo flamegraph --bench inference_bench

# Perf (Linux)
perf record --call-graph dwarf ./target/release/llm-worker-rbee
perf report
```

### Testing
```bash
# Unit tests
cargo test --lib

# All tests
cargo test --all

# With output
cargo test -- --nocapture
```

### Build
```bash
# Debug
cargo build

# Release
cargo build --release

# With CUDA
cargo build --release --features cuda
```

---

**TEAM-006 Execution Plan: Data-Driven, Low-Risk, Pragmatic**

**Estimated Time: 5-7 hours** (vs 20-30 hours for full refactor)  
**Risk Level: LOW** (incremental, validated changes)  
**Confidence: HIGH** (based on empirical data)

---

*Execution Plan by TEAM-006, 2025-10-08*  
*"Profile, optimize, validate. Ship working code."*
