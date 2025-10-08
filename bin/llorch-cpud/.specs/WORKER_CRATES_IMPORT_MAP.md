# Worker Crates Import Map for llorch-cpud

**Date:** 2025-10-08  
**Purpose:** Document which worker-crates are imported in which files  
**Status:** Reference guide

---

## Overview

llorch-cpud uses 4 worker-crates:
1. **worker-http** - HTTP server, InferenceBackend trait
2. **worker-common** - SamplingConfig, InferenceResult, startup callbacks
3. **worker-tokenizer** - Tokenization (BPE, GGUF, HuggingFace)
4. **worker-models** - GPTConfig, model architecture configs

---

## Import Map by File

### src/main.rs

**Purpose:** HTTP server entry point

**Imports:**
```rust
use worker_http::{create_router, HttpServer};
use worker_common::startup;
use llorch_cpud::backend::CpuInferenceBackend;
```

**Why:**
- `worker_http` - Provides HTTP server and routing
- `worker_common` - Provides `startup::callback_ready()` for pool-managerd
- `CpuInferenceBackend` - Our implementation (internal)

---

### src/backend/cpu_backend.rs

**Purpose:** CPU inference backend implementation

**Imports:**
```rust
use async_trait::async_trait;
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use worker_tokenizer::Tokenizer;  // After model implementation
use crate::model::GPT2Model;
use crate::cache::KVCache;
```

**Why:**
- `worker_http::InferenceBackend` - Trait we implement
- `worker_common::InferenceResult` - Return type for execute()
- `worker_common::SamplingConfig` - Parameter for execute()
- `worker_tokenizer::Tokenizer` - Tokenize prompts, decode outputs
- Internal imports for model and cache

---

### src/model/gpt2.rs

**Purpose:** GPT-2 model implementation

**Imports:**
```rust
use worker_models::GPTConfig;
use worker_common::SamplingConfig;
use crate::layers::{LayerNorm, Attention, FFN, Transformer};
use crate::cache::KVCache;
```

**Why:**
- `worker_models::GPTConfig` - Model configuration (vocab_size, hidden_dim, etc.)
- `worker_common::SamplingConfig` - Generation parameters
- Internal imports for layers

---

### src/model/config.rs

**Purpose:** Model configuration helpers

**Imports:**
```rust
use worker_models::GPTConfig;
```

**Why:**
- Re-export or extend GPTConfig if needed

---

### src/layers/attention/mod.rs

**Purpose:** Attention orchestration

**Imports:**
```rust
use crate::cache::KVCache;
use crate::layers::attention::{QKVProjection, AttentionScores, AttentionOutput};
```

**Why:**
- `KVCache` - Internal import (from src/cache/)
- No worker-crates needed (pure implementation)

---

### src/layers/*.rs (layer_norm, embedding, ffn, transformer)

**Purpose:** Layer implementations

**Imports:**
```rust
// No worker-crates imports!
// Only internal imports and ndarray
use ndarray::{Array2, Array3};
use crate::tensor::ops;
```

**Why:**
- Pure implementation
- No need for worker-crates
- Only tensor operations

---

### src/cache/kv_cache.rs

**Purpose:** KV cache implementation

**Imports:**
```rust
// No worker-crates imports!
use ndarray::Array3;
```

**Why:**
- Pure implementation
- No need for worker-crates

---

### src/tensor/ops.rs

**Purpose:** CPU tensor operations

**Imports:**
```rust
// No worker-crates imports!
use ndarray::{Array1, Array2, Array3, Array4};
```

**Why:**
- Pure implementation
- Just ndarray operations

---

## Summary by Worker Crate

### worker-http (HTTP server)

**Used in:**
- `src/main.rs` - Create router, start server
- `src/backend/cpu_backend.rs` - Implement InferenceBackend trait

**Provides:**
- `create_router()` - Create HTTP router
- `HttpServer` - HTTP server
- `InferenceBackend` trait - Platform abstraction

---

### worker-common (Common types)

**Used in:**
- `src/main.rs` - Startup callbacks
- `src/backend/cpu_backend.rs` - Result and config types
- `src/model/gpt2.rs` - Sampling config

**Provides:**
- `startup::callback_ready()` - Pool manager callback
- `InferenceResult` - Inference result type
- `SamplingConfig` - Sampling parameters
- `WorkerError` - Error types

---

### worker-tokenizer (Tokenization)

**Used in:**
- `src/backend/cpu_backend.rs` - Tokenize/decode

**Provides:**
- `Tokenizer` - BPE tokenizer
- `encode()` - Text → tokens
- `decode()` - Tokens → text
- GGUF and HuggingFace support

---

### worker-models (Model configs)

**Used in:**
- `src/model/gpt2.rs` - Model configuration
- `src/model/config.rs` - Config helpers

**Provides:**
- `GPTConfig` - GPT-2 configuration
- `gpt2_small()`, `gpt2_medium()` - Presets
- Model architecture configs

---

## Files That DON'T Import Worker Crates

These files are **pure implementation** (no worker-crates):

- ✅ `src/layers/layer_norm.rs`
- ✅ `src/layers/embedding.rs`
- ✅ `src/layers/attention/qkv.rs`
- ✅ `src/layers/attention/scores.rs`
- ✅ `src/layers/attention/output.rs`
- ✅ `src/layers/ffn.rs`
- ✅ `src/layers/transformer.rs`
- ✅ `src/cache/kv_cache.rs`
- ✅ `src/tensor/ops.rs`
- ✅ `src/error.rs`

**Why:** These are the core model implementation. They only need ndarray and internal imports.

---

## Dependency Flow

```
main.rs
  ↓ uses
worker-http (HTTP server)
worker-common (startup)
  ↓ creates
CpuInferenceBackend (src/backend/cpu_backend.rs)
  ↓ uses
worker-http (InferenceBackend trait)
worker-common (InferenceResult, SamplingConfig)
worker-tokenizer (Tokenizer)
  ↓ uses
GPT2Model (src/model/gpt2.rs)
  ↓ uses
worker-models (GPTConfig)
worker-common (SamplingConfig)
  ↓ uses
Layers (src/layers/*.rs)
  ↓ uses
KVCache (src/cache/kv_cache.rs)
  ↓ uses
Tensor ops (src/tensor/ops.rs)
  ↓ uses
ndarray (external)
```

---

## Import Guidelines

### When to Import Worker Crates

✅ **Import worker-crates when:**
- Implementing HTTP server (worker-http)
- Implementing InferenceBackend trait (worker-http)
- Using startup callbacks (worker-common)
- Using result/config types (worker-common)
- Tokenizing text (worker-tokenizer)
- Using model configs (worker-models)

❌ **Don't import worker-crates when:**
- Implementing layers (pure implementation)
- Implementing cache (pure implementation)
- Implementing tensor ops (pure implementation)
- Writing tests (use internal imports)

### Import Order

```rust
// 1. Standard library
use std::sync::Arc;

// 2. External crates
use ndarray::Array2;
use async_trait::async_trait;

// 3. Worker crates
use worker_http::InferenceBackend;
use worker_common::SamplingConfig;
use worker_tokenizer::Tokenizer;
use worker_models::GPTConfig;

// 4. Internal crates
use crate::model::GPT2Model;
use crate::cache::KVCache;
use crate::layers::LayerNorm;
```

---

## Validation Checklist

### ✓ main.rs
- [ ] Imports worker-http (create_router, HttpServer)
- [ ] Imports worker-common (startup)
- [ ] No unnecessary imports

### ✓ backend/cpu_backend.rs
- [ ] Imports worker-http (InferenceBackend)
- [ ] Imports worker-common (InferenceResult, SamplingConfig)
- [ ] Imports worker-tokenizer (Tokenizer)
- [ ] Imports internal types (GPT2Model, KVCache)

### ✓ model/gpt2.rs
- [ ] Imports worker-models (GPTConfig)
- [ ] Imports worker-common (SamplingConfig)
- [ ] Imports internal layers

### ✓ layers/*.rs
- [ ] NO worker-crates imports
- [ ] Only ndarray and internal imports

### ✓ cache/kv_cache.rs
- [ ] NO worker-crates imports
- [ ] Only ndarray

### ✓ tensor/ops.rs
- [ ] NO worker-crates imports
- [ ] Only ndarray

---

## Common Mistakes to Avoid

### ❌ WRONG: Importing worker-crates in layers
```rust
// src/layers/layer_norm.rs
use worker_common::SamplingConfig;  // ← Don't do this!
```

### ✅ CORRECT: Pure implementation
```rust
// src/layers/layer_norm.rs
use ndarray::Array2;  // ← Only ndarray
```

---

### ❌ WRONG: Importing HTTP in model
```rust
// src/model/gpt2.rs
use worker_http::InferenceBackend;  // ← Don't do this!
```

### ✅ CORRECT: Only config types
```rust
// src/model/gpt2.rs
use worker_models::GPTConfig;  // ← Only config
use worker_common::SamplingConfig;  // ← Only config
```

---

### ❌ WRONG: Importing tokenizer in layers
```rust
// src/layers/attention/mod.rs
use worker_tokenizer::Tokenizer;  // ← Don't do this!
```

### ✅ CORRECT: Internal imports only
```rust
// src/layers/attention/mod.rs
use crate::cache::KVCache;  // ← Internal only
```

---

## Summary

**Worker crates are used in 3 places:**
1. **main.rs** - HTTP server setup
2. **backend/cpu_backend.rs** - InferenceBackend implementation
3. **model/gpt2.rs** - Model configuration

**Everything else is pure implementation:**
- Layers (no worker-crates)
- Cache (no worker-crates)
- Tensor ops (no worker-crates)

**This keeps the model implementation clean and reusable!**

---

Built by TEAM CASCADE 🌊
