# Worker Crates Analysis & Optimization Plan

**Analysis by:** TEAM-005  
**Date:** 2025-10-08  
**Purpose:** Identify reusable crates and Candle optimization opportunities

---

## TL;DR: Massive Reuse Opportunities ✅

**Key Findings:**
1. ✅ **worker-gguf** - GGUF parser (ready to use, no Candle needed)
2. ✅ **worker-tokenizer** - Full BPE tokenizer (ready to use)
3. ✅ **worker-models** - Model adapters (architecture-agnostic, no inference code)
4. ❌ **NO Candle usage** in worker-crates (they're format/config only)
5. ✅ **shared-crates** - Auth, logging, GPU info (ready to use)

**We can reuse ~90% of the infrastructure!**

---

## Worker Crates Inventory

### 1. worker-gguf ✅ **READY TO USE**

**Purpose:** GGUF file format parser

**What it does:**
- Parses GGUF metadata (architecture, vocab size, etc.)
- Extracts tensor metadata (names, shapes, offsets)
- Provides model configuration

**What it DOESN'T do:**
- No tensor operations
- No inference
- No dequantization (moved to CUDA kernels)

**Dependencies:**
```toml
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
byteorder = "1.5"
half = "2.4"
```

**Files:**
- `lib.rs` - Metadata extraction
- `parser.rs` - GGUF format parsing

**Can we use Candle?** ❌ No need - it's pure parsing

**Can we reuse?** ✅ **YES** - Exactly what we need for weight loading

**Integration:**
```rust
// In llorch-candled
use worker_gguf::GGUFMetadata;

let metadata = GGUFMetadata::from_file("model.gguf")?;
let arch = metadata.architecture()?;
let vocab_size = metadata.vocab_size()?;
```

---

### 2. worker-tokenizer ✅ **READY TO USE**

**Purpose:** BPE tokenization (HuggingFace compatible)

**What it does:**
- Encodes text → token IDs
- Decodes token IDs → text
- Supports streaming
- HuggingFace JSON format
- Vocab and merges handling

**Dependencies:**
```toml
tokenizers = "0.15"  # HuggingFace tokenizers
worker-gguf = { path = "../worker-gguf" }
```

**Files:**
- `encoder.rs` - Text → tokens
- `decoder.rs` - Tokens → text
- `streaming.rs` - Streaming support
- `hf_json.rs` - HuggingFace format
- `vocab.rs` - Vocabulary handling
- `merges.rs` - BPE merges

**Can we use Candle?** ❌ No need - it's text processing

**Can we reuse?** ✅ **YES** - Complete tokenization solution

**Integration:**
```rust
// In llorch-candled
use worker_tokenizer::{TokenizerBackend, EncoderConfig};

let tokenizer = TokenizerBackend::from_hf_json("tokenizer.json")?;
let tokens = tokenizer.encode("Hello, world!", &config)?;
let text = tokenizer.decode(&tokens)?;
```

---

### 3. worker-models ✅ **READY TO USE**

**Purpose:** Model adapters (Qwen, Phi-3, GPT, Llama)

**What it does:**
- Model configuration (vocab size, hidden size, etc.)
- Architecture detection
- Adapter pattern for model-agnostic code
- Forward pass configuration (NOT implementation)

**What it DOESN'T do:**
- No inference implementation
- No tensor operations
- Just config and interfaces

**Dependencies:**
```toml
worker-gguf = { path = "../worker-gguf" }
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
```

**Files:**
- `adapter.rs` - Unified model interface
- `factory.rs` - Model detection/creation
- `gpt.rs` - GPT config
- `phi3.rs` - Phi-3 config
- `qwen.rs` - Qwen config

**Can we use Candle?** ❌ No - it's just config structs

**Can we reuse?** ✅ **YES** - For config and architecture detection

**Integration:**
```rust
// In llorch-candled
use worker_models::{AdapterFactory, Architecture};

let arch = AdapterFactory::detect_from_gguf("model.gguf")?;
match arch {
    Architecture::Llama2 => { /* use Llama2 config */ },
    Architecture::Qwen => { /* use Qwen config */ },
    // ...
}
```

---

### 4. worker-common ✅ **READY TO USE**

**Purpose:** Common types and utilities

**What it does:**
- Shared types
- Error handling
- Utilities

**Can we reuse?** ✅ **YES** - For shared infrastructure

---

### 5. worker-http ✅ **READY TO USE**

**Purpose:** HTTP server for workers

**What it does:**
- Axum-based HTTP server
- Streaming responses
- Health checks
- Callback support

**Can we reuse?** ✅ **YES** - Already using in llorch-candled

---

### 6. worker-compute ⚠️ **EVALUATE**

**Purpose:** Compute abstractions

**Need to check:** What does it provide?

---

## Shared Crates Inventory

### Available in /bin/shared-crates:

1. ✅ **audit-logging** - Audit trail
2. ✅ **auth-min** - Minimal auth
3. ✅ **deadline-propagation** - Request deadlines
4. ✅ **gpu-info** - GPU detection/info
5. ✅ **input-validation** - Input validation
6. ✅ **narration-core** - Logging/tracing
7. ✅ **pool-registry-types** - Pool types
8. ✅ **secrets-management** - Secret handling

**All ready to use!**

---

## Candle Optimization Analysis

### Question: Can worker-crates use Candle?

**Answer: NO - and that's correct!** ✅

**Why:**
1. **worker-gguf** - Pure parsing (no tensors)
2. **worker-tokenizer** - Text processing (no tensors)
3. **worker-models** - Config only (no inference)

**These crates are infrastructure, not inference.**

### Where Candle Belongs

**Candle is ONLY for inference:**
- ✅ llorch-candled (our project)
- ✅ Tensor operations
- ✅ Model forward passes
- ✅ GPU acceleration

**NOT for:**
- ❌ File parsing (GGUF)
- ❌ Text processing (tokenizer)
- ❌ Configuration (models)

---

## Integration Plan

### Phase 1: Reuse Worker Crates ✅

**Add to llorch-candled/Cargo.toml:**
```toml
[dependencies]
# Worker crates (reuse existing)
worker-gguf = { path = "../worker-crates/worker-gguf" }
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
worker-models = { path = "../worker-crates/worker-models" }
worker-http = { path = "../worker-crates/worker-http" }

# Shared crates
gpu-info = { path = "../shared-crates/gpu-info" }
narration-core = { path = "../shared-crates/narration-core" }

# Candle (for inference only)
candle-core = "0.9"
candle-nn = "0.9"
```

### Phase 2: Refactor to Candle Pattern ✅

**Follow Candle's single-file pattern:**

```rust
// src/model/llama2.rs (single file, like Candle)

use worker_gguf::GGUFMetadata;
use worker_tokenizer::TokenizerBackend;
use worker_models::Architecture;
use candle_core::{Tensor, Device};
use candle_nn::{VarBuilder, embedding, ops};

// Centralized cache (like Candle)
pub struct Cache {
    kv: candle_nn::kv_cache::KvCache,
    cos: Tensor,  // RoPE cache
    sin: Tensor,  // RoPE cache
    masks: HashMap<usize, Tensor>,
    device: Device,
}

// Integrated attention (like Candle)
struct CausalSelfAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    
    fn apply_rope(&self, x: &Tensor, cache: &Cache) -> Result<Tensor> {
        candle_nn::rotary_emb::rope_i(x, &cache.cos, &cache.sin)
    }
    
    fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        // Integrated: QKV + RoPE + Attention
    }
}

// SwiGLU FFN (like Candle)
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

// Transformer block (like Candle)
struct Block {
    rms_1: Tensor,
    attn: CausalSelfAttention,
    rms_2: Tensor,
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

// Full model (like Candle)
pub struct Llama {
    embedding: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln_f: Tensor,
    lm_head: Tensor,
    
    // Load from GGUF using worker-gguf
    pub fn from_gguf(path: &str, device: &Device) -> Result<Self> {
        let metadata = GGUFMetadata::from_file(path)?;
        let vb = VarBuilder::from_gguf(path, device)?;
        Self::load(vb, &Config::from_metadata(&metadata)?)
    }
}
```

### Phase 3: Weight Loading with worker-gguf ✅

**Use worker-gguf for metadata, Candle for tensors:**

```rust
use worker_gguf::GGUFMetadata;
use candle_core::safetensors;

// 1. Parse metadata with worker-gguf
let metadata = GGUFMetadata::from_file("model.gguf")?;
let config = Config {
    hidden_size: metadata.get_int("llama.embedding_length")? as usize,
    num_layers: metadata.get_int("llama.block_count")? as usize,
    // ...
};

// 2. Load tensors with Candle
let vb = VarBuilder::from_gguf("model.gguf", &device)?;

// 3. Build model
let model = Llama::load(vb, &config)?;
```

### Phase 4: Tokenization with worker-tokenizer ✅

**Use worker-tokenizer for text processing:**

```rust
use worker_tokenizer::TokenizerBackend;

// Load tokenizer
let tokenizer = TokenizerBackend::from_hf_json("tokenizer.json")?;

// Encode
let tokens = tokenizer.encode("Hello, world!", &config)?;
let input = Tensor::from_vec(tokens, (1, tokens.len()), &device)?;

// Inference
let logits = model.forward(&input, 0, &mut cache)?;

// Decode
let next_token = logits.argmax(D::Minus1)?;
let text = tokenizer.decode(&[next_token])?;
```

---

## Recommended Architecture

### Final Structure

```
llorch-candled/
├── src/
│   ├── model/
│   │   ├── llama2.rs        ✅ Single file (Candle pattern)
│   │   │   ├── Cache        ✅ Centralized state
│   │   │   ├── CausalSelfAttention  ✅ Integrated
│   │   │   ├── Mlp          ✅ SwiGLU
│   │   │   ├── Block        ✅ Transformer block
│   │   │   └── Llama        ✅ Full model
│   │   └── mod.rs
│   ├── inference.rs         ✅ Inference pipeline
│   ├── main.rs              ✅ HTTP server (using worker-http)
│   └── lib.rs
└── Cargo.toml               ✅ Depends on worker-crates

Dependencies:
- worker-gguf      ✅ GGUF parsing
- worker-tokenizer ✅ Tokenization
- worker-models    ✅ Config/detection
- worker-http      ✅ HTTP server
- candle-core      ✅ Tensors
- candle-nn        ✅ Ops
```

### Delete/Consolidate

**Delete (redundant with worker-crates):**
- ❌ `src/tensor/` - Use Candle tensors
- ❌ `src/backend/` - Use Candle device
- ❌ `src/cache/` (separate) - Integrate into model

**Keep (Candle-specific):**
- ✅ `src/model/llama2.rs` - Inference implementation
- ✅ `src/main.rs` - Entry point

---

## Benefits of This Approach

### 1. Massive Code Reuse ✅
- GGUF parsing: **Ready** (worker-gguf)
- Tokenization: **Ready** (worker-tokenizer)
- Config: **Ready** (worker-models)
- HTTP: **Ready** (worker-http)

**We only implement inference!**

### 2. Candle-Aligned Architecture ✅
- Single-file model (like Candle)
- Centralized cache
- Integrated forward passes
- VarBuilder pattern

### 3. Best of Both Worlds ✅
- **Infrastructure:** Reuse worker-crates
- **Inference:** Use Candle optimizations
- **Clean separation:** Config vs compute

---

## Action Plan

### Immediate (1-2 hours)

1. **Add worker-crate dependencies**
```bash
# Update Cargo.toml
```

2. **Create unified Cache**
```rust
// src/model/cache.rs
pub struct Cache {
    kv: candle_nn::kv_cache::KvCache,
    cos: Tensor,
    sin: Tensor,
    masks: HashMap<usize, Tensor>,
}
```

3. **Test integration**
```rust
// Verify worker-gguf works
let metadata = GGUFMetadata::from_file("test.gguf")?;
```

### Short-term (1 day)

1. **Refactor to single-file model**
   - Move everything to `src/model/llama2.rs`
   - Integrate attention pipeline
   - Use centralized cache

2. **Implement weight loading**
   - Use worker-gguf for metadata
   - Use Candle VarBuilder for tensors

3. **Integrate tokenizer**
   - Use worker-tokenizer
   - Wire encode/decode

### Medium-term (2-3 days)

1. **Full inference pipeline**
   - Prefill + decode
   - KV caching
   - Sampling

2. **HTTP server**
   - Use worker-http
   - Streaming responses

3. **Testing**
   - End-to-end tests
   - Benchmark

---

## Summary

### Key Findings

1. ✅ **worker-crates are infrastructure** - Not for inference
2. ✅ **NO Candle needed in worker-crates** - They're format/config only
3. ✅ **Massive reuse opportunity** - GGUF, tokenizer, HTTP all ready
4. ✅ **Our structure CAN be optimized** - Follow Candle's single-file pattern
5. ✅ **Best approach:** Reuse infrastructure + Candle inference

### Recommended Changes

**Architecture:**
- ✅ Single-file model (`model/llama2.rs`)
- ✅ Centralized cache
- ✅ Integrated forward passes
- ✅ Reuse worker-crates for infrastructure

**Dependencies:**
- ✅ Add worker-gguf, worker-tokenizer, worker-models
- ✅ Keep Candle for inference only
- ✅ Use shared-crates for utilities

**Result:**
- 90% code reuse from worker-crates
- Candle-aligned architecture
- Clean separation: infrastructure vs inference

---

**Next Steps:** Implement unified cache, then refactor to single-file model pattern.

---

*Analysis by TEAM-005, 2025-10-08*
