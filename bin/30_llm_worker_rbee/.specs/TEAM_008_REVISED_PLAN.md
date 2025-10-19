# TEAM-008 REVISED PLAN - Use Candle's Full Power

**Team:** TEAM-008  
**Date:** 2025-10-08T22:47:18+02:00  
**Revision:** After discovering Candle's complete Llama implementation  
**Status:** 🚀 READY FOR EXECUTION

---

## Executive Summary

**STOP REINVENTING THE WHEEL!**

Candle already has:
- ✅ Complete Llama implementation (`candle-transformers/models/llama.rs`)
- ✅ Unified Cache with kvs/cos/sin/masks (lines 145-153)
- ✅ SafeTensors loading with `VarBuilder`
- ✅ GGUF support via `candle-transformers`
- ✅ Generation loop with sampling
- ✅ KV cache management

**Our job:** Use Candle's implementation, adapt for worker-http integration.

---

## What We Just Learned

### 1. Candle Has Unified Cache Already!

```rust
// candle-transformers/src/models/llama.rs:145-153
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,  // Per-layer KV
    cos: Tensor,                          // RoPE cos
    sin: Tensor,                          // RoPE sin
    device: Device,
}
```

**This is EXACTLY what TEAM-005 wanted and we just implemented!**

### 2. Candle Has VarBuilder for Model Loading

```rust
// candle-nn/src/var_builder.rs
VarBuilder::from_mmaped_safetensors()  // Memory-mapped loading
VarBuilder::from_varmap()               // For training
```

### 3. Candle Has Complete Llama Model

```rust
// candle-transformers/src/models/llama.rs
pub struct Llama {
    wte: Embedding,                    // Token embeddings
    blocks: Vec<LlamaBlock>,           // Transformer layers
    ln_f: RmsNorm,                     // Final norm
    lm_head: Linear,                   // Output projection
}
```

---

## Revised Strategy

### Phase 1: Replace Our Cache with Candle's (2 hours)

**What we did:** Created `src/cache/unified_cache.rs` (150 lines)  
**What we should do:** Use `candle_transformers::models::llama::Cache`

**Tasks:**
1. Delete `src/cache/unified_cache.rs`
2. Import `candle_transformers::models::llama::Cache`
3. Update RoPE to use Candle's Cache
4. Update Attention to use Candle's Cache
5. Run tests

**Why:** Don't maintain duplicate code when Candle has it.

### Phase 2: Use Candle's Llama Model (3-4 hours)

**What we have:** Partial layer implementations  
**What we should use:** `candle_transformers::models::llama::Llama`

**Tasks:**
1. Add `candle-transformers` dependency
2. Import Llama model and config
3. Wrap Candle's Llama in our `CandleInferenceBackend`
4. Load model using `VarBuilder::from_mmaped_safetensors()`
5. Test forward pass

**Benefits:**
- ✅ Complete implementation (all 32 layers)
- ✅ Tested by Candle team
- ✅ Optimized for GPU/CPU
- ✅ Supports GQA, RoPE scaling, etc.

### Phase 3: Integrate with worker-http (2-3 hours)

**Tasks:**
1. Implement `InferenceBackend` trait for Candle's Llama
2. Add generation loop using Candle's sampling
3. Wire up SSE streaming
4. Test end-to-end

### Phase 4: Add GGUF Support (2-3 hours)

**Option A:** Use `candle-transformers` GGUF support  
**Option B:** Use `worker-gguf` + convert to SafeTensors  
**Option C:** Load GGUF tensors directly into VarBuilder

**Recommendation:** Option A (use Candle's GGUF support if available)

---

## Comparison: Our Code vs Candle

### Our Unified Cache (150 lines)
```rust
// src/cache/unified_cache.rs
pub struct Cache {
    kv_caches: Vec<KvCache>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    causal_masks: HashMap<usize, Tensor>,
    device: Device,
    max_seq_len: usize,
    n_layers: usize,
}
```

### Candle's Cache (8 lines)
```rust
// candle-transformers/src/models/llama.rs
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}
```

**Verdict:** Nearly identical! Use Candle's.

---

## Dependencies Update

```toml
[dependencies]
# Candle (use transformers for models)
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"  # ADD THIS

# Optional CUDA
candle-kernels = { version = "0.9", optional = true }
cudarc = { version = "0.11", optional = true }

# Worker infrastructure
worker-http = { path = "../worker-crates/worker-http" }
worker-common = { path = "../worker-crates/worker-common" }

# For GGUF support (if not using Candle's)
worker-gguf = { path = "../worker-crates/worker-gguf" }  # OPTIONAL
```

---

## Implementation Plan

### Step 1: Add candle-transformers (10 min)

```bash
cd bin/llm-worker-rbee
# Add to Cargo.toml
```

### Step 2: Replace Cache (1 hour)

```rust
// BEFORE: src/cache/mod.rs
mod unified_cache;
pub use unified_cache::Cache;

// AFTER: src/cache/mod.rs
pub use candle_transformers::models::llama::Cache;
```

Update RoPE and Attention to use Candle's Cache API.

### Step 3: Use Candle's Llama (2 hours)

```rust
// src/model/llama2.rs
use candle_transformers::models::llama::{Llama, Config, Cache};
use candle_nn::VarBuilder;

pub struct Llama2Model {
    model: Llama,
    cache: Cache,
    config: Config,
}

impl Llama2Model {
    pub fn from_safetensors(path: &str, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?;
        let config = Config::config_7b_v2(false);  // Or load from config.json
        let model = Llama::load(vb, &config)?;
        let cache = Cache::new(true, DType::F32, &config, device)?;
        
        Ok(Self { model, cache, config })
    }
    
    pub fn forward(&mut self, tokens: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(tokens, pos, &mut self.cache)
    }
}
```

### Step 4: Implement Backend (1 hour)

```rust
// src/backend/candle_backend.rs
use worker_http::InferenceBackend;
use crate::model::Llama2Model;

#[async_trait]
impl InferenceBackend for CandleInferenceBackend {
    async fn load(model_path: String, device: Device) -> Result<Self> {
        let model = Llama2Model::from_safetensors(&model_path, &device)?;
        Ok(Self { model })
    }
    
    async fn generate(&mut self, prompt: &str, config: SamplingConfig) 
        -> Result<InferenceResult> {
        // Use Candle's generation utilities
        // Or implement simple sampling loop
    }
}
```

### Step 5: Test (1 hour)

```bash
# Download a small Llama model
# Test loading
# Test generation
# Test streaming
```

---

## What We Keep vs What We Replace

### KEEP (Our Work)
- ✅ Multi-backend infrastructure (CPU/CUDA/Accelerate)
- ✅ worker-http integration
- ✅ Device initialization
- ✅ Test infrastructure

### REPLACE (Use Candle)
- ❌ Our unified Cache → Use Candle's Cache
- ❌ Our layer implementations → Use Candle's Llama
- ❌ Our model loading → Use VarBuilder
- ❌ Manual RoPE/Attention → Use Candle's implementations

### MAYBE (Evaluate)
- ⚠️ worker-gguf → Check if Candle has better GGUF support
- ⚠️ worker-tokenizer → Check if Candle has tokenizer
- ⚠️ Generation loop → Use Candle's or implement simple one

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Add candle-transformers | 10 min | Pending |
| Replace Cache | 1 hour | Pending |
| Use Candle's Llama | 2 hours | Pending |
| Implement Backend | 1 hour | Pending |
| Test & Validate | 1 hour | Pending |
| **Total** | **5-6 hours** | **Ready** |

---

## Success Criteria

1. ✅ Using `candle-transformers::models::llama::Llama`
2. ✅ Using `candle-transformers::models::llama::Cache`
3. ✅ Model loads from SafeTensors
4. ✅ Forward pass works
5. ✅ Generation produces coherent text
6. ✅ All tests pass
7. ✅ No duplicate code (Cache, RoPE, etc.)

---

## Key Insight

**We were reinventing Candle's wheel!**

TEAM-005 was right about unified cache, but we should have checked if Candle already had it (it does!).

**New approach:** Use Candle's full `candle-transformers` library, not just the low-level ops.

---

**TEAM-008 signing on (revised).**

*"Don't build what already exists. Use the library properly."*  
— TEAM-008, 2025-10-08T22:47:18+02:00

**END REVISED PLAN**
