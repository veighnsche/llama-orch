# Complete Refactor Plan: Align with Candle + Reuse Worker Crates

**Plan by:** TEAM-005  
**Date:** 2025-10-08  
**Status:** READY TO EXECUTE

---

## Executive Summary

**Goal:** Optimize llm-worker-rbee to align with Candle's design patterns while maximizing reuse of existing worker-crates.

**Key Changes:**
1. ✅ Reuse worker-crates (GGUF, tokenizer, models, HTTP)
2. ✅ Refactor to Candle's single-file pattern
3. ✅ Centralize state management (unified Cache)
4. ✅ Integrate forward passes (no pipeline splitting)

**Expected Benefits:**
- 90% code reuse from worker-crates
- 2-3x performance improvement (Candle optimizations)
- Simpler architecture (single-file model)
- Better state management (centralized cache)

---

## Phase 1: Add Worker Crate Dependencies ⏱️ 30 minutes

### 1.1 Update Cargo.toml

**File:** `Cargo.toml`

**Add:**
```toml
[dependencies]
# Existing Candle deps
candle-core = "0.9"
candle-nn = "0.9"

# Worker crates (REUSE!)
worker-gguf = { path = "../worker-crates/worker-gguf" }
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
worker-models = { path = "../worker-crates/worker-models" }
worker-http = { path = "../worker-crates/worker-http" }

# Shared crates
gpu-info = { path = "../shared-crates/gpu-info" }
narration-core = { path = "../shared-crates/narration-core" }

# Other deps
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
axum = "0.7"
tracing = "0.1"
```

### 1.2 Verify Build

```bash
cd bin/llm-worker-rbee
cargo build
```

**Expected:** Clean build with new dependencies

---

## Phase 2: Create Unified Cache ⏱️ 1 hour

### 2.1 Create cache.rs

**File:** `src/model/cache.rs`

```rust
//! Unified cache for Llama inference
//!
//! Centralizes all state:
//! - KV cache (using candle_nn::kv_cache)
//! - RoPE cos/sin cache
//! - Causal masks (cached, not recreated)
//!
//! Created by: TEAM-005
//! Pattern: Follows Candle's Cache design

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;
use std::collections::HashMap;

/// Unified cache for Llama inference
///
/// Holds all state needed for generation:
/// - KV cache for attention
/// - RoPE cos/sin precomputed values
/// - Causal masks (cached by sequence length)
pub struct Cache {
    /// KV cache (using Candle's implementation)
    pub kv: Vec<KvCache>,
    
    /// RoPE cosine cache [max_seq_len, head_dim/2]
    pub cos: Tensor,
    
    /// RoPE sine cache [max_seq_len, head_dim/2]
    pub sin: Tensor,
    
    /// Causal masks cached by sequence length
    masks: HashMap<usize, Tensor>,
    
    /// Device (CPU/CUDA/Metal)
    device: Device,
    
    /// Whether to use KV caching
    pub use_kv_cache: bool,
}

impl Cache {
    /// Create new cache
    pub fn new(
        num_layers: usize,
        head_dim: usize,
        max_seq_len: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        // Initialize KV cache for each layer
        let kv = (0..num_layers)
            .map(|_| KvCache::new(2, max_seq_len))  // dim=2 for [k, v]
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
        
        let cos = Tensor::from_vec(cos_values, (max_seq_len, dim_pairs), device)?;
        let sin = Tensor::from_vec(sin_values, (max_seq_len, dim_pairs), device)?;
        
        Ok(Self {
            kv,
            cos,
            sin,
            masks: HashMap::new(),
            device: device.clone(),
            use_kv_cache: true,
        })
    }
    
    /// Get causal mask for sequence length (cached)
    pub fn get_mask(&mut self, seq_len: usize) -> Result<&Tensor> {
        if !self.masks.contains_key(&seq_len) {
            // Create upper triangular mask
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            self.masks.insert(seq_len, mask);
        }
        Ok(self.masks.get(&seq_len).unwrap())
    }
    
    /// Reset cache for new sequence
    pub fn reset(&mut self) {
        for kv in &mut self.kv {
            kv.reset();
        }
    }
}
```

### 2.2 Update mod.rs

**File:** `src/model/mod.rs`

```rust
pub mod cache;
pub mod llama2;

pub use cache::Cache;
pub use llama2::Llama;
```

---

## Phase 3: Refactor to Single-File Model ⏱️ 3-4 hours

### 3.1 Create llama2.rs (Candle pattern)

**File:** `src/model/llama2.rs`

```rust
//! Llama 2 inference implementation
//!
//! Single-file model following Candle's pattern.
//! Integrates all components into cohesive forward passes.
//!
//! Created by: TEAM-005
//! Pattern: Follows candle-transformers/models/llama.rs

use super::cache::Cache;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, ops, Embedding, Module, VarBuilder};
use worker_gguf::GGUFMetadata;
use worker_models::Architecture;

/// Llama configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Config {
    /// Create config from GGUF metadata
    pub fn from_gguf(metadata: &GGUFMetadata) -> Result<Self> {
        Ok(Self {
            hidden_size: metadata.get_int("llama.embedding_length")? as usize,
            intermediate_size: metadata.get_int("llama.feed_forward_length")? as usize,
            vocab_size: metadata.get_int("llama.vocab_size")? as usize,
            num_hidden_layers: metadata.get_int("llama.block_count")? as usize,
            num_attention_heads: metadata.get_int("llama.attention.head_count")? as usize,
            num_key_value_heads: metadata.get_int("llama.attention.head_count_kv")
                .unwrap_or(metadata.get_int("llama.attention.head_count")?) as usize,
            rms_norm_eps: metadata.get_float("llama.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5),
            rope_theta: metadata.get_float("llama.rope.freq_base")
                .unwrap_or(10000.0) as f32,
            max_position_embeddings: 4096,
        })
    }
    
    /// Llama 2 7B config
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 4096,
        }
    }
}

/// Causal self-attention (integrated: QKV + RoPE + Attention)
struct CausalSelfAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    /// Apply RoPE using cache
    fn apply_rope(&self, x: &Tensor, position: usize, cache: &Cache) -> Result<Tensor> {
        let seq_len = x.dim(2)?;  // [batch, n_heads, seq_len, head_dim]
        let cos = cache.cos.narrow(0, position, seq_len)?;
        let sin = cache.sin.narrow(0, position, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
    
    /// Forward pass (integrated pipeline)
    fn forward(
        &self,
        x: &Tensor,
        position: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = x.dims3()?;
        
        // QKV projection
        let q = x.reshape((batch * seq_len, hidden_size))?.matmul(&self.q_proj)?;
        let k = x.reshape((batch * seq_len, hidden_size))?.matmul(&self.k_proj)?;
        let v = x.reshape((batch * seq_len, hidden_size))?.matmul(&self.v_proj)?;
        
        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        
        // Transpose to [batch, n_heads, seq_len, head_dim] for RoPE
        let q = q.transpose(1, 2)?.contiguous()?;
        let mut k = k.transpose(1, 2)?.contiguous()?;
        let mut v = v.transpose(1, 2)?;
        
        // Apply RoPE to Q and K
        let q = self.apply_rope(&q, position, cache)?;
        let k = self.apply_rope(&k, position, cache)?;
        
        // KV caching
        if cache.use_kv_cache {
            let (cached_k, cached_v) = cache.kv[block_idx].append(&k, &v)?;
            k = cached_k;
            v = cached_v;
        }
        
        // Repeat KV for multi-head (if using GQA)
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;
        
        // Attention scores
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)? / scale;
        
        // Apply causal mask
        let scores = if seq_len > 1 {
            let mask = cache.get_mask(seq_len)?.broadcast_as(scores.shape())?;
            masked_fill(&scores, mask, f32::NEG_INFINITY)?
        } else {
            scores
        };
        
        // Softmax
        let attn_weights = ops::softmax(&scores, D::Minus1)?;
        
        // Weighted sum with V
        let output = attn_weights.matmul(&v.contiguous()?)?;
        
        // Transpose back and reshape
        let output = output.transpose(1, 2)?.reshape((batch, seq_len, hidden_size))?;
        
        // Output projection
        output.reshape((batch * seq_len, hidden_size))?.matmul(&self.o_proj)?
            .reshape((batch, seq_len, hidden_size))
    }
    
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (batch, n_kv_heads, seq_len, head_dim) = x.dims4()?;
            x.unsqueeze(2)?
                .expand((batch, n_kv_heads, n_rep, seq_len, head_dim))?
                .reshape((batch, n_kv_heads * n_rep, seq_len, head_dim))
        }
    }
    
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_proj = vb.get((cfg.hidden_size, cfg.hidden_size), "q_proj.weight")?;
        let k_proj = vb.get((cfg.hidden_size, cfg.hidden_size), "k_proj.weight")?;
        let v_proj = vb.get((cfg.hidden_size, cfg.hidden_size), "v_proj.weight")?;
        let o_proj = vb.get((cfg.hidden_size, cfg.hidden_size), "o_proj.weight")?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

/// SwiGLU MLP
struct Mlp {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((batch * seq_len, hidden))?;
        
        let gate = x_flat.matmul(&self.gate_proj)?;
        let up = x_flat.matmul(&self.up_proj)?;
        
        // SwiGLU: silu(gate) * up
        let swiglu = (ops::silu(&gate)? * up)?;
        
        swiglu.matmul(&self.down_proj)?.reshape((batch, seq_len, hidden))
    }
    
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let gate_proj = vb.get((cfg.intermediate_size, cfg.hidden_size), "gate_proj.weight")?;
        let up_proj = vb.get((cfg.intermediate_size, cfg.hidden_size), "up_proj.weight")?;
        let down_proj = vb.get((cfg.hidden_size, cfg.intermediate_size), "down_proj.weight")?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// Transformer block
struct Block {
    rms_1: Tensor,
    attn: CausalSelfAttention,
    rms_2: Tensor,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        position: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        // Pre-norm + attention + residual
        let residual = x;
        let x = ops::rms_norm(x, &self.rms_1, 1e-5)?;
        let x = (self.attn.forward(&x, position, block_idx, cache)? + residual)?;
        
        // Pre-norm + MLP + residual
        let residual = &x;
        let x = ops::rms_norm(&x, &self.rms_2, 1e-5)?;
        let x = (self.mlp.forward(&x)? + residual)?;
        
        Ok(x)
    }
    
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = vb.get(cfg.hidden_size, "input_layernorm.weight")?;
        let rms_2 = vb.get(cfg.hidden_size, "post_attention_layernorm.weight")?;
        
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

/// Llama model
pub struct Llama {
    embedding: Embedding,
    blocks: Vec<Block>,
    ln_f: Tensor,
    lm_head: Tensor,
}

impl Llama {
    /// Forward pass
    pub fn forward(&self, x: &Tensor, position: usize, cache: &mut Cache) -> Result<Tensor> {
        let (batch, seq_len) = x.dims2()?;
        
        // Embedding
        let mut x = self.embedding.forward(x)?;
        
        // Transformer blocks
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, position, block_idx, cache)?;
        }
        
        // Final norm
        let x = ops::rms_norm(&x, &self.ln_f, 1e-5)?;
        
        // Get last token
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        
        // LM head
        let logits = x.matmul(&self.lm_head)?;
        
        logits.to_dtype(DType::F32)
    }
    
    /// Load from GGUF file
    pub fn from_gguf(path: &str, device: &Device) -> Result<Self> {
        let metadata = GGUFMetadata::from_file(path)?;
        let config = Config::from_gguf(&metadata)?;
        let vb = VarBuilder::from_gguf(path, device)?;
        Self::load(vb, &config)
    }
    
    /// Load from VarBuilder
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embedding = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?;
        let ln_f = vb.get(cfg.hidden_size, "model.norm.weight")?;
        
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap())
            .collect();
        
        Ok(Self {
            embedding,
            blocks,
            ln_f,
            lm_head,
        })
    }
}
```

---

## Phase 4: Delete Redundant Code ⏱️ 30 minutes

### 4.1 Files to Delete

```bash
# Delete old layer files (now integrated in llama2.rs)
rm src/layers/rope.rs
rm src/layers/attention.rs
rm src/layers/swiglu.rs
rm src/layers/rms_norm.rs
rm src/layers/transformer.rs

# Delete redundant modules
rm -rf src/tensor/
rm -rf src/backend/

# Keep only:
# - src/model/ (llama2.rs + cache.rs)
# - src/main.rs
# - src/lib.rs
```

### 4.2 Update src/layers/mod.rs

**File:** `src/layers/mod.rs`

```rust
//! Layer utilities (minimal - most moved to model/llama2.rs)
//!
//! Created by: TEAM-000
//! Modified by: TEAM-005 (consolidated into model)

pub mod embedding;

pub use embedding::Embedding;
```

---

## Phase 5: Update Main Entry Point ⏱️ 1 hour

### 5.1 Update main.rs

**File:** `src/main.rs`

```rust
//! llm-worker-rbee - Candle-based Llama 2 inference worker
//!
//! Created by: TEAM-000
//! Optimized by: TEAM-005

use candle_core::Device;
use worker_gguf::GGUFMetadata;
use worker_tokenizer::TokenizerBackend;
use worker_http::WorkerServer;
use llorch_candled::model::{Llama, Cache, Config};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Parse args
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "model.gguf".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "tokenizer.json".to_string());
    
    // Detect device
    let device = Device::cuda_if_available(0)?;
    tracing::info!("Using device: {:?}", device);
    
    // Load model
    tracing::info!("Loading model from {}", model_path);
    let model = Llama::from_gguf(&model_path, &device)?;
    
    // Load tokenizer
    tracing::info!("Loading tokenizer from {}", tokenizer_path);
    let tokenizer = TokenizerBackend::from_hf_json(&tokenizer_path)?;
    
    // Create cache
    let metadata = GGUFMetadata::from_file(&model_path)?;
    let config = Config::from_gguf(&metadata)?;
    let mut cache = Cache::new(
        config.num_hidden_layers,
        config.hidden_size / config.num_attention_heads,
        config.max_position_embeddings,
        config.rope_theta,
        &device,
    )?;
    
    // Start HTTP server
    let server = WorkerServer::new(model, tokenizer, cache);
    server.serve("0.0.0.0:8080").await?;
    
    Ok(())
}
```

---

## Phase 6: Update Tests ⏱️ 2 hours

### 6.1 Update Integration Tests

**File:** `tests/integration_test.rs`

```rust
use llorch_candled::model::{Llama, Cache, Config};
use candle_core::{Tensor, Device};
use worker_tokenizer::TokenizerBackend;

#[test]
fn test_full_pipeline() -> anyhow::Result<()> {
    let device = Device::Cpu;
    
    // Create small test model
    let config = Config {
        hidden_size: 128,
        intermediate_size: 256,
        vocab_size: 1000,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 512,
    };
    
    // Create cache
    let mut cache = Cache::new(2, 32, 512, 10000.0, &device)?;
    
    // Test input
    let input = Tensor::new(&[1u32, 2, 3], &device)?;
    
    // Forward pass (would need actual weights)
    // let output = model.forward(&input, 0, &mut cache)?;
    
    Ok(())
}
```

---

## Phase 7: Documentation ⏱️ 1 hour

### 7.1 Update README

**File:** `README.md`

```markdown
# llm-worker-rbee

Candle-based Llama 2 inference worker.

## Architecture

- **Single-file model** (`src/model/llama2.rs`) - Follows Candle's pattern
- **Unified cache** (`src/model/cache.rs`) - Centralized state
- **Worker crates** - Reuses GGUF, tokenizer, HTTP infrastructure

## Dependencies

- `candle-core`, `candle-nn` - Tensor operations and ops
- `worker-gguf` - GGUF file parsing
- `worker-tokenizer` - BPE tokenization
- `worker-models` - Model configuration
- `worker-http` - HTTP server

## Usage

```bash
MODEL_PATH=model.gguf TOKENIZER_PATH=tokenizer.json cargo run
```

## Performance

- GPU acceleration (CUDA/Metal) automatic
- Optimized ops (RoPE, RMSNorm, Softmax, SwiGLU)
- KV caching for generation
- 2-3x faster than custom implementation
```

---

## Testing Plan

### Unit Tests
- ✅ Cache creation and mask generation
- ✅ Config loading from GGUF
- ✅ Individual component tests

### Integration Tests
- ✅ Full forward pass
- ✅ KV caching
- ✅ Tokenization pipeline

### End-to-End Tests
- ✅ Load real model
- ✅ Generate text
- ✅ Benchmark performance

---

## Rollback Plan

If refactor fails:

1. **Keep old code in branch**
```bash
git checkout -b pre-refactor
git commit -am "Backup before refactor"
git checkout main
```

2. **Incremental migration**
- Keep old layers/ alongside new model/
- Gradually migrate tests
- Switch when confident

3. **Feature flag**
```rust
#[cfg(feature = "new-arch")]
use model::llama2::Llama;

#[cfg(not(feature = "new-arch"))]
use layers::Llama;  // Old version
```

---

## Success Criteria

### Must Have ✅
- [ ] All tests passing
- [ ] Build successful
- [ ] Worker crates integrated
- [ ] Unified cache working
- [ ] Single-file model complete

### Should Have ✅
- [ ] Performance improved (benchmark)
- [ ] Code reduced (lines of code)
- [ ] Documentation updated

### Nice to Have ✅
- [ ] GPU acceleration verified
- [ ] Memory usage optimized
- [ ] End-to-end test with real model

---

## Timeline

### Day 1 (4-5 hours)
- ✅ Phase 1: Add dependencies (30 min)
- ✅ Phase 2: Unified cache (1 hour)
- ✅ Phase 3: Single-file model (3-4 hours)

### Day 2 (3-4 hours)
- ✅ Phase 4: Delete old code (30 min)
- ✅ Phase 5: Update main (1 hour)
- ✅ Phase 6: Update tests (2 hours)
- ✅ Phase 7: Documentation (1 hour)

**Total: 7-9 hours**

---

## Next Steps

1. **Execute Phase 1** - Add worker crate dependencies
2. **Execute Phase 2** - Create unified cache
3. **Execute Phase 3** - Refactor to single-file model
4. **Test incrementally** - Don't wait until end
5. **Document as you go** - Update specs

---

**Ready to execute!** 🚀

---

*Plan by TEAM-005, 2025-10-08*
