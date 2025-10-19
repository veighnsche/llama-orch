# Architecture Analysis: Our Structure vs Candle's Design

**Analysis by:** TEAM-005  
**Date:** 2025-10-08  
**Question:** Is our folder structure fighting against Candle's design patterns?

---

## TL;DR: **We're Actually Aligned** ✅

Our structure is **compatible** with Candle's design. The key insight: **Candle provides ops, we provide architecture**. Our split is correct.

---

## Candle's Architecture Pattern

### How Candle Organizes Llama

**File:** `reference/candle/candle-transformers/src/models/llama.rs` (537 lines, single file)

```rust
// Everything in ONE file:
pub struct Cache { ... }           // KV cache + RoPE cache
struct CausalSelfAttention { ... } // Q/K/V proj + RoPE + attention
struct Mlp { ... }                 // SwiGLU FFN
struct Block { ... }               // RMSNorm + Attn + RMSNorm + MLP
pub struct Llama { ... }           // Embedding + Blocks + LM head
```

**Key Pattern:** Candle keeps the **entire model in one file** because:
1. Components are tightly coupled (share Cache, Config)
2. Forward pass is sequential (embedding → blocks → head)
3. Weight loading is hierarchical (VarBuilder pattern)

### What Candle Provides vs What It Expects You To Build

**Candle Provides (candle-nn):**
```rust
candle_nn::ops::rms_norm()        // Op
candle_nn::rotary_emb::rope()     // Op
candle_nn::ops::softmax()         // Op
candle_nn::ops::silu()            // Op
candle_nn::Linear                 // Layer (thin wrapper)
candle_nn::Embedding              // Layer (thin wrapper)
```

**Candle Expects You To Build:**
```rust
struct CausalSelfAttention { ... } // Architecture
struct Mlp { ... }                 // Architecture
struct Block { ... }               // Architecture
struct Llama { ... }               // Architecture
struct Cache { ... }               // State management
```

---

## Our Current Structure

```
src/
├── layers/
│   ├── rms_norm.rs      ✅ Thin wrapper around candle_nn::ops::rms_norm
│   ├── rope.rs          ✅ Thin wrapper around candle_nn::rotary_emb::rope_i
│   ├── attention.rs     ✅ QKVProjection + Attention (architecture)
│   ├── swiglu.rs        ⏳ Will wrap candle_nn::ops::swiglu
│   ├── transformer.rs   ⏳ Block (architecture)
│   └── mod.rs
├── cache/
│   └── kv_cache.rs      ✅ Re-export candle_nn::kv_cache
├── model/
│   └── llama2.rs        ⏳ Full Llama model (architecture)
└── lib.rs
```

### Analysis: Is This Wrong?

**No, it's actually fine.** Here's why:

1. **We're building the same architecture** - just split into files
2. **Candle's ops are stateless** - they don't care about file structure
3. **Our split is logical** - layers are reusable components
4. **Candle's single-file is for simplicity** - not a requirement

---

## The Real Question: Are We Splitting What Should Be Together?

### Potential Issue #1: Cache Split ❓

**Candle's Pattern:**
```rust
// Cache contains BOTH KV cache AND RoPE cos/sin cache
pub struct Cache {
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,  // KV cache
    cos: Tensor,                          // RoPE cache
    sin: Tensor,                          // RoPE cache
    masks: HashMap<usize, Tensor>,        // Causal masks
    device: Device,
}
```

**Our Pattern:**
```rust
// We split them:
// cache/kv_cache.rs - Just KV cache
pub use candle_nn::kv_cache::KvCache;

// layers/rope.rs - RoPE has its own cos/sin cache
pub struct RoPE {
    cos_cache: Tensor,
    sin_cache: Tensor,
    // ...
}

// layers/attention.rs - Causal mask created on-the-fly
pub fn apply_causal_mask(&self, scores: &Tensor) -> CandleResult<Tensor> {
    // Creates mask each time
}
```

**Is This A Problem?** 🤔

**Potentially YES** - Here's why:

1. **Cache Duplication:** RoPE cache is separate from KV cache
2. **Mask Recreation:** Causal mask created every forward pass
3. **State Fragmentation:** Cache state split across multiple structs
4. **Coupling:** Attention needs to know about RoPE's cache format

### Potential Issue #2: Attention Split ❓

**Candle's Pattern:**
```rust
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // ... attention logic in same struct
    
    fn apply_rotary_emb(&self, x: &Tensor, cache: &Cache) -> Result<Tensor> {
        // RoPE is a method of attention
    }
    
    fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        // QKV + RoPE + scores + softmax + output all together
    }
}
```

**Our Pattern:**
```rust
// We split into 3 structs:
pub struct QKVProjection { ... }  // Just projection
pub struct RoPE { ... }           // Just rotation
pub struct Attention { ... }      // Scores + softmax + output
```

**Is This A Problem?** 🤔

**Potentially YES** - Here's why:

1. **Pipeline Complexity:** User must wire QKV → RoPE → Attention manually
2. **State Management:** Each struct manages its own state
3. **Performance:** Multiple function calls instead of one integrated forward pass
4. **Coupling:** Changes to one affect the others

---

## Candle's Design Philosophy

### Key Insight from Candle's Code

Looking at `llama.rs`, Candle's philosophy is:

1. **Ops are primitive** - `rope()`, `rms_norm()`, `softmax()` are building blocks
2. **Architecture is integrated** - `CausalSelfAttention` combines all ops into one forward pass
3. **State is centralized** - `Cache` holds all state (KV, RoPE, masks)
4. **Loading is hierarchical** - `VarBuilder` pattern for weight loading

### What This Means For Us

**We should:**
1. ✅ Use Candle's ops (we do)
2. ❌ **Don't over-split the architecture** (we might be doing this)
3. ✅ Centralize state management (we should improve this)
4. ✅ Keep forward pass integrated (we should improve this)

---

## Recommended Architecture Changes

### Option 1: Candle-Style (Single File) 🎯

**Pros:**
- Matches Candle's proven pattern
- Centralized state (Cache)
- Integrated forward pass
- Easier to optimize

**Cons:**
- Larger files
- Less modular

**Structure:**
```rust
// src/model/llama2.rs (everything in one file)

pub struct Cache {
    kv_cache: candle_nn::kv_cache::KvCache,
    cos: Tensor,  // RoPE cache
    sin: Tensor,  // RoPE cache
    masks: HashMap<usize, Tensor>,
}

struct CausalSelfAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    
    fn apply_rope(&self, x: &Tensor, cache: &Cache) -> Result<Tensor> {
        candle_nn::rotary_emb::rope_i(x, &cache.cos, &cache.sin)
    }
    
    fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        // QKV projection
        let q = x.matmul(&self.q_proj)?;
        let k = x.matmul(&self.k_proj)?;
        let v = x.matmul(&self.v_proj)?;
        
        // RoPE
        let q = self.apply_rope(&q, cache)?;
        let k = self.apply_rope(&k, cache)?;
        
        // Attention scores
        let scores = q.matmul(&k.t()?)? / scale;
        
        // Causal mask
        let scores = scores + cache.get_mask(seq_len)?;
        
        // Softmax
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        
        // Output
        let out = attn.matmul(&v)?;
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = x.matmul(&self.gate_proj)?;
        let up = x.matmul(&self.up_proj)?;
        let swiglu = candle_nn::ops::swiglu(&Tensor::cat(&[gate, up], D::Minus1)?)?;
        swiglu.matmul(&self.down_proj)
    }
}

struct Block {
    rms_1: Tensor,  // Just the weight
    attn: CausalSelfAttention,
    rms_2: Tensor,  // Just the weight
    mlp: Mlp,
    
    fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let residual = x;
        let x = candle_nn::ops::rms_norm(x, &self.rms_1, 1e-5)?;
        let x = (self.attn.forward(&x, cache)? + residual)?;
        
        let residual = &x;
        let x = candle_nn::ops::rms_norm(&x, &self.rms_2, 1e-5)?;
        let x = (self.mlp.forward(&x)? + residual)?;
        Ok(x)
    }
}

pub struct Llama {
    embedding: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln_f: Tensor,
    lm_head: Tensor,
}
```

### Option 2: Keep Current (Modular) 🔧

**Pros:**
- More modular
- Easier to test individual components
- Clearer separation of concerns

**Cons:**
- More complex wiring
- Fragmented state
- Potential performance overhead

**Improvements Needed:**
```rust
// 1. Centralize cache
pub struct LlamaCache {
    kv: candle_nn::kv_cache::KvCache,
    rope_cos: Tensor,
    rope_sin: Tensor,
    causal_masks: HashMap<usize, Tensor>,
}

// 2. Integrate attention pipeline
pub struct CausalSelfAttention {
    qkv: QKVProjection,
    rope: RoPE,
    o_proj: Tensor,
    
    fn forward(&self, x: &Tensor, cache: &mut LlamaCache) -> Result<Tensor> {
        // Integrated pipeline
    }
}
```

### Option 3: Hybrid (Recommended) ⭐

**Best of both worlds:**

```rust
// Keep layers/ for reusable ops wrappers
// layers/ops.rs
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::rms_norm(x, weight, eps)
}

pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    candle_nn::rotary_emb::rope_i(x, cos, sin)
}

// Move architecture to model/
// model/llama2.rs - Integrated architecture
pub struct Cache { ... }
struct CausalSelfAttention { ... }
struct Mlp { ... }
struct Block { ... }
pub struct Llama { ... }
```

---

## The Core Issue: State Management

### Current Problem

**State is fragmented:**
```
RoPE.cos_cache ─┐
RoPE.sin_cache ─┤
                ├─ Should be together
KvCache.k ──────┤
KvCache.v ──────┤
                │
Attention (creates masks on-the-fly) ─┘
```

### Candle's Solution

**State is centralized:**
```
Cache {
    cos: Tensor,           // RoPE
    sin: Tensor,           // RoPE
    kvs: Vec<(K, V)>,      // KV cache
    masks: HashMap<...>,   // Causal masks (cached)
}
```

**Why This Matters:**
1. **Single source of truth** - All state in one place
2. **Efficient caching** - Masks computed once, reused
3. **Clear ownership** - Cache owns all state
4. **Easy to reset** - One struct to clear

---

## Recommendation

### Short-term (Keep Current Structure) ✅

**What to do:**
1. Create unified `LlamaCache` struct
2. Move RoPE cos/sin into cache
3. Cache causal masks
4. Keep modular layers for now

**Why:**
- Less disruptive
- Tests still work
- Can refactor later

### Long-term (Refactor to Candle Pattern) 🎯

**What to do:**
1. Move architecture to `model/llama2.rs`
2. Integrate attention pipeline
3. Centralize state in Cache
4. Keep `layers/` for op wrappers only

**Why:**
- Matches proven pattern
- Better performance
- Easier to maintain
- Aligns with Candle's design

---

## Action Items

### Immediate (Fix State Fragmentation)

1. **Create unified cache:**
```rust
// src/model/cache.rs
pub struct LlamaCache {
    pub kv: candle_nn::kv_cache::KvCache,
    pub rope_cos: Tensor,
    pub rope_sin: Tensor,
    causal_masks: HashMap<usize, Tensor>,
    device: Device,
}

impl LlamaCache {
    pub fn new(config: &Config, device: &Device) -> Result<Self> {
        // Initialize all caches
    }
    
    pub fn get_mask(&mut self, seq_len: usize) -> Result<&Tensor> {
        // Cache and return mask
    }
}
```

2. **Update RoPE to use shared cache:**
```rust
// layers/rope.rs
pub fn apply_rope(
    x: &Tensor, 
    position: usize, 
    cache: &LlamaCache
) -> Result<Tensor> {
    let seq_len = x.dim(1)?;
    let cos = cache.rope_cos.narrow(0, position, seq_len)?;
    let sin = cache.rope_sin.narrow(0, position, seq_len)?;
    candle_nn::rotary_emb::rope_i(x, &cos, &sin)
}
```

3. **Update Attention to use shared cache:**
```rust
// layers/attention.rs
impl Attention {
    pub fn forward(
        &self, 
        x: &Tensor, 
        cache: &mut LlamaCache
    ) -> Result<Tensor> {
        // Use cache.get_mask() instead of creating mask
    }
}
```

### Future (Refactor to Integrated Architecture)

1. Move to single-file model (like Candle)
2. Integrate forward passes
3. Optimize state management
4. Add VarBuilder for weight loading

---

## Conclusion

**Your intuition was correct** - we ARE splitting things that Candle treats as tightly coupled:

1. ❌ **Cache is fragmented** - RoPE cache separate from KV cache
2. ❌ **Masks recreated** - Should be cached
3. ❌ **Pipeline is split** - QKV → RoPE → Attention should be integrated

**But it's not a fundamental problem** - it's a state management issue, not a structural incompatibility.

**Recommended path:**
1. **Now:** Unify cache (1-2 hours)
2. **Later:** Consider refactoring to Candle's integrated pattern (when we have time)

**The good news:** We're using Candle's ops correctly. We just need to manage state better.

---

*Analysis by TEAM-005, 2025-10-08*
