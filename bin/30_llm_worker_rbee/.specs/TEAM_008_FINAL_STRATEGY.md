# TEAM-008 FINAL STRATEGY - Use Candle Properly

**Team:** TEAM-008  
**Date:** 2025-10-08T22:47:18+02:00  
**Status:** ✅ VALIDATED - Ready to Execute

---

## Executive Summary

**We were using Candle wrong!**

- ❌ **What we were doing:** Using only `candle-core` and `candle-nn` for low-level ops
- ✅ **What we should do:** Use `candle-transformers` for complete model implementations

**Candle provides EVERYTHING:**
1. Complete Llama implementation
2. Unified Cache (kvs + cos/sin + masks)
3. GGUF loading via `gguf_file`
4. SafeTensors loading via `VarBuilder`
5. HuggingFace tokenizers integration
6. Generation utilities

---

## What Candle Actually Provides

### 1. Complete Llama Model

**File:** `candle-transformers/src/models/llama.rs`

```rust
pub struct Llama {
    wte: Embedding,              // Token embeddings
    blocks: Vec<LlamaBlock>,     // 32 transformer layers
    ln_f: RmsNorm,               // Final norm
    lm_head: Linear,             // Output projection
}

impl Llama {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self>
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor>
}
```

### 2. Unified Cache (EXACTLY what TEAM-005 wanted!)

**File:** `candle-transformers/src/models/llama.rs:145-153`

```rust
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,        // Causal masks (cached)
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,   // Per-layer KV
    cos: Tensor,                          // RoPE cos
    sin: Tensor,                          // RoPE sin
    device: Device,
}
```

**This is IDENTICAL to what we just implemented!**

### 3. GGUF Loading

**File:** `candle-transformers/src/models/quantized_llama.rs`

```rust
use candle::quantized::gguf_file;

// Load quantized model from GGUF
let mut file = std::fs::File::open(path)?;
let model = gguf_file::Content::read(&mut file)?;
```

### 4. SafeTensors Loading

**File:** `candle-nn/src/var_builder.rs`

```rust
// Memory-mapped loading (efficient for large models)
let vb = unsafe { 
    VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? 
};

// Load model
let model = Llama::load(vb, &config)?;
```

### 5. Tokenizers

**Uses HuggingFace tokenizers crate:**

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
```

### 6. Generation Utilities

**File:** `candle-examples/src/token_output_stream.rs`

```rust
use candle_examples::token_output_stream::TokenOutputStream;

let mut token_stream = TokenOutputStream::new(tokenizer);
// Handles UTF-8 decoding, streaming output
```

---

## Our Mistake: Reinventing the Wheel

### What We Implemented (Unnecessarily)

1. **Unified Cache** (150 lines) → Candle has it
2. **RoPE layer** (100 lines) → Candle has it
3. **Attention layer** (200 lines) → Candle has it
4. **Model loading** (300 lines) → Candle has it

### What We Should Have Done

1. Add `candle-transformers` dependency
2. Use `Llama::load()` 
3. Use `Cache::new()`
4. Done!

---

## Revised Implementation Plan

### Phase 1: Add Dependencies (10 min)

```toml
[dependencies]
# Candle - USE TRANSFORMERS!
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"  # THIS IS THE KEY!

# Optional backends
candle-kernels = { version = "0.9", optional = true }
cudarc = { version = "0.11", optional = true }

# HuggingFace tokenizers
tokenizers = "0.15"

# Worker infrastructure
worker-http = { path = "../worker-crates/worker-http" }
worker-common = { path = "../worker-crates/worker-common" }

# Async runtime
tokio = { version = "1", features = ["rt", "macros"] }
async-trait = "0.1"
```

### Phase 2: Replace Our Cache (30 min)

**Delete:**
- `src/cache/unified_cache.rs` (our implementation)

**Use:**
```rust
// src/cache/mod.rs
pub use candle_transformers::models::llama::Cache;
```

**Update RoPE and Attention:**
- They already use `Cache` (we just implemented it!)
- Just change import to Candle's Cache

### Phase 3: Use Candle's Llama Model (2 hours)

```rust
// src/model/llama2.rs
use candle_transformers::models::llama::{Llama, Config, Cache};
use candle_nn::VarBuilder;
use candle::{DType, Device, Result};

pub struct Llama2Model {
    model: Llama,
    cache: Cache,
    config: Config,
    device: Device,
}

impl Llama2Model {
    /// Load from SafeTensors files
    pub fn from_safetensors(
        paths: &[String], 
        config_path: &str,
        device: &Device
    ) -> Result<Self> {
        // Load config
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_path)?
        )?;
        
        // Load model weights
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(paths, DType::F32, device)? 
        };
        
        // Create model
        let model = Llama::load(vb, &config)?;
        
        // Create cache
        let cache = Cache::new(true, DType::F32, &config, device)?;
        
        Ok(Self {
            model,
            cache,
            config,
            device: device.clone(),
        })
    }
    
    /// Load from GGUF file
    pub fn from_gguf(path: &str, device: &Device) -> Result<Self> {
        // Use candle's quantized_llama
        use candle_transformers::models::quantized_llama;
        // Implementation here
        todo!("GGUF loading")
    }
    
    /// Forward pass
    pub fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Vec<f32>> {
        let tokens = candle::Tensor::new(tokens, &self.device)?
            .unsqueeze(0)?;  // Add batch dimension
        
        let logits = self.model.forward(&tokens, pos, &mut self.cache)?;
        
        // Get last token logits
        let logits = logits.i((0, tokens.dim(1)? - 1))?;
        logits.to_vec1()
    }
    
    /// Reset cache for new sequence
    pub fn reset_cache(&mut self) -> Result<()> {
        self.cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
        Ok(())
    }
}
```

### Phase 4: Implement Backend (1 hour)

```rust
// src/backend/candle_backend.rs
use worker_http::InferenceBackend;
use worker_common::{SamplingConfig, InferenceResult};
use crate::model::Llama2Model;
use tokenizers::Tokenizer;

pub struct CandleInferenceBackend {
    model: Llama2Model,
    tokenizer: Tokenizer,
}

#[async_trait]
impl InferenceBackend for CandleInferenceBackend {
    async fn load(model_path: String, device: Device) -> Result<Self> {
        // Load model
        let model = Llama2Model::from_safetensors(
            &[model_path.clone()],
            &format!("{}/config.json", model_path),
            &device,
        )?;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(
            format!("{}/tokenizer.json", model_path)
        )?;
        
        Ok(Self { model, tokenizer })
    }
    
    async fn generate(
        &mut self,
        prompt: &str,
        config: SamplingConfig,
    ) -> Result<InferenceResult> {
        // Encode prompt
        let tokens = self.tokenizer
            .encode(prompt, true)?
            .get_ids()
            .to_vec();
        
        // Reset cache
        self.model.reset_cache()?;
        
        // Generation loop
        let mut generated_tokens = Vec::new();
        let mut pos = 0;
        
        // Process prompt
        for &token in &tokens {
            self.model.forward(&[token], pos)?;
            pos += 1;
        }
        
        // Generate tokens
        for _ in 0..config.max_tokens {
            let logits = self.model.forward(&[tokens[pos - 1]], pos)?;
            
            // Sample next token
            let next_token = sample_token(&logits, config.temperature)?;
            
            // Check for EOS
            if next_token == self.tokenizer.token_to_id("
