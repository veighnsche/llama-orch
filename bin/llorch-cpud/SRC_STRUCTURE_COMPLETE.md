# llorch-cpud Source Structure - COMPLETE

**Date:** 2025-10-08  
**Status:** ✅ Stub files created with code hints  
**Next:** Implement Checkpoint 1 (LayerNorm)

---

## Summary

Complete directory structure for `llorch-cpud/src` has been created with:
- ✅ All stub files with proper structure
- ✅ Code hints and TODOs for each component
- ✅ Import guidelines clearly documented
- ✅ Checkpoint alignment noted in each file
- ✅ Test stubs included
- ✅ Cargo.toml with all dependencies

---

## Files Created

### Core Files (5 files)
```
src/
├── main.rs              ✅ Entry point with worker-http integration
├── lib.rs               ✅ Library exports
├── error.rs             ✅ Error types
├── Cargo.toml           ✅ Dependencies and configuration
└── README.md            ✅ Documentation
```

### Backend (2 files)
```
src/backend/
├── mod.rs               ✅ Module exports
└── cpu_backend.rs       ✅ InferenceBackend implementation
```

### Cache (2 files)
```
src/cache/
├── mod.rs               ✅ Module exports
└── kv_cache.rs          ✅ KV Cache implementation (Checkpoint 3)
```

### Layers (9 files)
```
src/layers/
├── mod.rs               ✅ Module exports
├── layer_norm.rs        ✅ LayerNorm (Checkpoint 1)
├── embedding.rs         ✅ Token + Position embeddings
├── ffn.rs               ✅ Feedforward network (Checkpoint 6)
├── transformer.rs       ✅ Transformer block (Checkpoint 7)
└── attention/
    ├── mod.rs           ✅ Attention module exports
    ├── qkv.rs           ✅ QKV projection (Checkpoint 2)
    ├── scores.rs        ✅ Attention scores (Checkpoint 4)
    └── output.rs        ✅ Attention output (Checkpoint 5)
```

### Model (2 files)
```
src/model/
├── mod.rs               ✅ Module exports
└── gpt2.rs              ✅ GPT-2 model (Checkpoints 8-12)
```

### Tensor Operations (2 files)
```
src/tensor/
├── mod.rs               ✅ Module exports
└── ops.rs               ✅ CPU tensor operations
```

**Total: 22 files created**

---

## Code Hints Included

Each file includes:

### 1. Header Comments
- Purpose of the file
- Import guidelines (worker-crates or ndarray only)
- Checkpoint alignment

### 2. Struct Definitions
- Complete struct with fields
- Field types and dimensions documented
- Constructor methods

### 3. TODO Comments
- Step-by-step implementation guide
- Algorithm pseudocode
- Important notes (e.g., "Handle Conv1D transpose")

### 4. Test Stubs
- Basic shape tests
- Placeholder for checkpoint tests
- Test structure examples

### 5. Documentation
- Docstrings for public APIs
- Parameter descriptions
- Return value descriptions

---

## Import Guidelines (Enforced in Code)

### Files with worker-crates imports:
```rust
// src/main.rs
use worker_http::{create_router, HttpServer};
use worker_common::startup;

// src/backend/cpu_backend.rs
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use worker_tokenizer::Tokenizer;

// src/model/gpt2.rs
use worker_models::GPTConfig;
```

### Files with NO worker-crates (pure implementation):
```rust
// All files in src/layers/
use ndarray::{Array1, Array2, Array3, Array4};
// NO worker-crates imports

// All files in src/cache/
use ndarray::{Array3, Array4, s};
// NO worker-crates imports

// All files in src/tensor/
use ndarray::{Array1, Array2, Array3, Array4};
// NO worker-crates imports
```

### Files with internal crate imports:
```rust
// src/layers/transformer.rs
use crate::layers::{LayerNorm, Attention, FFN};
use crate::cache::KVCache;
// Internal imports only - NO worker-crates

// src/layers/attention/mod.rs
use crate::cache::KVCache;
// Internal imports only - NO worker-crates
```

---

## Checkpoint Alignment

Each file is tagged with its checkpoint:

| File | Checkpoint | Status |
|------|-----------|--------|
| `main.rs` | 0 (Foundation) | ✅ Stub |
| `backend/cpu_backend.rs` | 0 (Foundation) | ✅ Stub |
| `layers/layer_norm.rs` | 1 (LayerNorm) | ⬜ TODO |
| `layers/attention/qkv.rs` | 2 (QKV) | ⬜ TODO |
| `cache/kv_cache.rs` | 3 (Cache) | ✅ Stub |
| `layers/attention/scores.rs` | 4 (Scores) | ⬜ TODO |
| `layers/attention/output.rs` | 5 (Output) | ⬜ TODO |
| `layers/ffn.rs` | 6 (FFN) | ⬜ TODO |
| `layers/transformer.rs` | 7 (Block) | ⬜ TODO |
| `model/gpt2.rs` | 8-12 (Full Model) | ⬜ TODO |

---

## Key Features

### 1. Single-Threaded Architecture
```rust
// src/main.rs
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Single-threaded for optimal CPU performance
}
```

### 2. Worker-HTTP Integration
```rust
// src/backend/cpu_backend.rs
#[async_trait]
impl InferenceBackend for CpuInferenceBackend {
    async fn execute(&self, prompt: &str, config: &SamplingConfig) 
        -> Result<InferenceResult> {
        // 1. Tokenize (worker-tokenizer)
        // 2. Generate (YOUR implementation)
        // 3. Decode (worker-tokenizer)
        // 4. Return (worker-common)
    }
}
```

### 3. KV Cache (Top-Level)
```rust
// src/cache/kv_cache.rs
pub struct KVCache {
    cache: Option<Array4<f32>>,  // [2, batch, max_seq, n_heads, head_dim]
    // Top-level because:
    // - Used by all 24 layers
    // - Future optimization target
}
```

### 4. Attention Module (Split)
```rust
// src/layers/attention/mod.rs
pub struct Attention {
    qkv: QKVProjection,      // Checkpoint 2
    scores: AttentionScores, // Checkpoint 4
    output: AttentionOutput, // Checkpoint 5
}
```

### 5. Pre-Norm Transformer
```rust
// src/layers/transformer.rs
pub fn forward(&mut self, x: &Array2<f32>, ...) -> Array2<f32> {
    // Pre-norm architecture:
    // 1. h = x + attention(ln_1(x))
    // 2. h = h + ffn(ln_2(h))
}
```

---

## Dependencies (Cargo.toml)

### Worker Crates (70% code reuse)
- `worker-common` - Types, error handling
- `worker-http` - HTTP server, InferenceBackend trait
- `worker-tokenizer` - BPE tokenization
- `worker-models` - GPTConfig

### New Dependencies (30% new code)
- `ndarray` - CPU tensor operations
- `tokio` - Async runtime (single-threaded)
- `anyhow`, `thiserror` - Error handling
- `tracing` - Logging

---

## Next Steps

### Immediate (Week 1, Day 1)
1. ✅ Source structure created
2. ⬜ Verify Cargo.toml paths to worker-crates
3. ⬜ Run `cargo check` to verify compilation
4. ⬜ Fix any import errors

### Week 1, Day 2-4
1. ⬜ Read `CHECKPOINT_01_LAYER_NORM.md`
2. ⬜ Implement `layers/layer_norm.rs`
3. ⬜ Create `tests/checkpoint_01_layer_norm.rs`
4. ⬜ Extract reference output from tinygrad
5. ⬜ Run test until it passes

### Week 1, Day 5
1. ⬜ Implement `layers/embedding.rs`
2. ⬜ Test embeddings

### Week 2+
Follow checkpoint order (2-12)

---

## Code Quality

### Every file includes:
- ✅ Header comments with purpose and imports
- ✅ Struct definitions with field documentation
- ✅ TODO comments with implementation steps
- ✅ Test stubs with examples
- ✅ Docstrings for public APIs

### Consistent patterns:
- ✅ Single-threaded emphasis
- ✅ Import clarity (worker-crates vs ndarray)
- ✅ Checkpoint alignment
- ✅ Error handling with custom Error type
- ✅ Tracing for debugging

---

## Validation

### Structure Validation
```bash
# Check all files exist
ls -R src/

# Expected output:
# src/main.rs
# src/lib.rs
# src/error.rs
# src/backend/mod.rs
# src/backend/cpu_backend.rs
# src/cache/mod.rs
# src/cache/kv_cache.rs
# src/layers/mod.rs
# src/layers/layer_norm.rs
# src/layers/embedding.rs
# src/layers/ffn.rs
# src/layers/transformer.rs
# src/layers/attention/mod.rs
# src/layers/attention/qkv.rs
# src/layers/attention/scores.rs
# src/layers/attention/output.rs
# src/model/mod.rs
# src/model/gpt2.rs
# src/tensor/mod.rs
# src/tensor/ops.rs
```

### Compilation Validation
```bash
# Check compilation (will fail on missing worker-crates)
cargo check

# Expected: Compilation errors for unimplemented TODOs
# This is normal - we have stubs, not implementations
```

---

## Success Criteria

- ✅ All 22 files created
- ✅ All files have proper structure
- ✅ All files have code hints and TODOs
- ✅ Import guidelines documented
- ✅ Checkpoint alignment clear
- ✅ Test stubs included
- ✅ Cargo.toml configured
- ✅ README documentation complete

**Status: COMPLETE ✅**

---

## Resources

- **Checkpoint Specs**: `.specs/checkpoints/`
- **Implementation Roadmap**: `.specs/IMPLEMENTATION_ROADMAP.md`
- **Source README**: `src/README.md`
- **Worker Crates**: `../../worker-crates/`

---

Built by TEAM CASCADE 🌊
