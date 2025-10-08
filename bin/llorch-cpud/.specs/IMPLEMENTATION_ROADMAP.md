# llorch-cpud Implementation Roadmap

**Date:** 2025-10-08  
**Purpose:** Connect specs, checkpoints, and worker-crates for GPT-2 CPU implementation  
**Status:** Ready to implement

---

## Overview

This document connects three critical resources:
1. **Specification:** `01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md` (complete behavioral spec)
2. **Validation:** `checkpoints/` (12 validation checkpoints)
3. **Existing Code:** `worker-crates/worker-models/src/gpt.rs` (GPT model scaffold)

---

## The Foundation We Have

### 1. Complete Specification ✅

**File:** `01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md`

- ✅ 12 phases documented
- ✅ All MUST/SHOULD/COULD requirements
- ✅ Validated against tinygrad, Candle, Mistral.rs
- ✅ Framework comparison (Appendix A)
- ✅ Line-by-line verification complete

### 2. Validation Checkpoints ✅

**Directory:** `checkpoints/`

- ✅ 12 detailed checkpoint files
- ✅ Master checklist for tracking
- ✅ Reference locations for all frameworks
- ✅ Tolerances and success criteria
- ✅ Debug commands and failure modes

### 3. Worker Crates (VERIFIED REUSABLE) ✅

**Directory:** `../worker-crates/`  
**Audit:** `WORKER_CRATES_REUSABILITY_AUDIT.md`

**Reusable Crates (99% reusable, approximately 4,198 lines):**

- ✅ **worker-common** (approximately 1,100 lines) - 100% reusable
  - SamplingConfig, InferenceResult, WorkerError
  - Startup callbacks, error handling
  - Pure Rust, no platform dependencies

- ✅ **worker-http** (approximately 771 lines) - 100% reusable
  - HTTP server (Axum), routes, SSE streaming
  - InferenceBackend trait (platform abstraction)
  - Request validation, health endpoint

- ✅ **worker-tokenizer** (approximately 1,200 lines) - 100% reusable
  - Pure Rust BPE tokenization
  - GGUF and HuggingFace support
  - No CUDA, no FFI, works on CPU

- ✅ **worker-models** (approximately 800 lines) - 100% reusable
  - GPTConfig, ModelAdapter trait
  - Architecture configs (GPT-2, Llama, Phi-3, Qwen)
  - Factory pattern for model detection

**Code Reuse:** 70% of llorch-cpud is already implemented!

---

## What We Need to Build

### Code Breakdown

**Reusable from worker-crates (70%):**
- HTTP server, routes, SSE streaming (approximately 771 lines)
- Tokenization (approximately 1,200 lines)
- Sampling config, result types (approximately 1,100 lines)
- Model configs (approximately 800 lines)
- **Total reusable: approximately 3,871 lines**

**Need to implement (30%):**
- CPU tensor operations (approximately 500 lines)
- LayerNorm (approximately 50 lines) - Checkpoint 1
- Attention (approximately 300 lines) - Checkpoints 2-5
- FFN (approximately 100 lines) - Checkpoint 6
- Transformer blocks (approximately 200 lines) - Checkpoint 7
- Model forward pass (approximately 300 lines) - Checkpoints 8-9
- Generation loop (approximately 150 lines) - Checkpoint 12
- Sampling implementation (approximately 200 lines) - Checkpoints 10-11
- **Total to implement: approximately 1,800 lines**

**Total llorch-cpud: approximately 5,671 lines**

### Phase 1: CPU Compute Backend (Week 1)

**Goal:** Implement `ComputeBackend` trait for CPU

**Files to Create:**
```
llorch-cpud/
├── Cargo.toml                          # Dependencies
├── src/
│   ├── main.rs                         # HTTP server entry point
│   │                                   # IMPORTS: worker-http, worker-common
│   ├── lib.rs                          # Library exports
│   ├── backend/
│   │   ├── mod.rs
│   │   └── cpu_backend.rs              # NEW: CpuInferenceBackend
│   │                                   # IMPORTS: worker-http, worker-common, worker-tokenizer
│   ├── model/
│   │   ├── mod.rs
│   │   ├── gpt2.rs                     # NEW: GPT2Model
│   │   │                               # IMPORTS: worker-models (GPTConfig)
│   │   └── config.rs
│   ├── layers/
│   │   ├── mod.rs
│   │   ├── layer_norm.rs               # NEW: CHECKPOINT 1
│   │   ├── embedding.rs                # NEW: Phase 2
│   │   ├── attention/                  # NEW: CHECKPOINTS 2, 4, 5
│   │   │   ├── mod.rs                  # IMPORTS: crate::cache::KVCache
│   │   │   ├── qkv.rs                  # CHECKPOINT 2
│   │   │   ├── scores.rs               # CHECKPOINT 4
│   │   │   └── output.rs               # CHECKPOINT 5
│   │   ├── ffn.rs                      # NEW: CHECKPOINT 6
│   │   └── transformer.rs              # NEW: CHECKPOINT 7
│   ├── cache/                          # NEW: CHECKPOINT 3
│   │   ├── mod.rs
│   │   └── kv_cache.rs                 # KV cache implementation
│   ├── tensor/
│   │   ├── mod.rs
│   │   └── ops.rs                      # CPU tensor operations (ndarray)
│   └── error.rs
└── tests/
    ├── checkpoint_00_foundation.rs
    ├── checkpoint_01_layer_norm.rs
    ├── checkpoint_02_qkv.rs
    ├── ...
    └── checkpoint_12_e2e.rs
```

**Worker Crates Usage:**
- **main.rs** → worker-http, worker-common
- **cpu_backend.rs** → worker-http, worker-common, worker-tokenizer
- **gpt2.rs** → worker-models
- **All layers** → No worker-crates (pure implementation)

**Dependencies:**
```toml
[package]
name = "llorch-cpud"
version = "0.1.0"
edition = "2021"

[dependencies]
# VERIFIED REUSABLE: worker-crates (99% reusable, approximately 4,198 lines)
# Audit: WORKER_CRATES_REUSABILITY_AUDIT.md

# worker-common: 100% reusable (approximately 1,100 lines)
# Provides: SamplingConfig, InferenceResult, WorkerError, startup callbacks
worker-common = { path = "../worker-crates/worker-common" }

# worker-http: 100% reusable (approximately 771 lines)
# Provides: HTTP server, routes, SSE streaming, InferenceBackend trait
worker-http = { path = "../worker-crates/worker-http" }

# worker-tokenizer: 100% reusable (approximately 1,200 lines)
# Provides: Pure Rust BPE tokenization, GGUF and HuggingFace support
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }

# worker-models: 100% reusable (approximately 800 lines)
# Provides: GPTConfig, ModelAdapter trait, architecture configs
worker-models = { path = "../worker-crates/worker-models" }

# CPU tensor operations (NEW - approximately 500 lines to implement)
ndarray = "0.15"
ndarray-linalg = "0.16"

# Async runtime (already in worker-http)
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"

# Utilities
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
clap = { version = "4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
```

---

## HTTP API Integration (CRITICAL)

### Important Note

**The GPT-2 spec (01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md) covers the MODEL only, not the HTTP server!**

The HTTP server comes from **worker-http** (already extracted and ready to use).

### Two-Part Implementation

1. **HTTP Server** (Week 1, Day 1) - Use worker-http
2. **GPT-2 Model** (Week 1-5) - Follow checkpoints

### HTTP Integration Steps

**Step 1: Implement InferenceBackend Trait**

```rust
// src/cpu_backend.rs
use async_trait::async_trait;
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};

pub struct CpuInferenceBackend {
    model: GPT2Model,  // Implemented via checkpoints
    tokenizer: Tokenizer,  // From worker-tokenizer
}

#[async_trait]
impl InferenceBackend for CpuInferenceBackend {
    async fn execute(&self, prompt: &str, config: &SamplingConfig) 
        -> Result<InferenceResult> {
        // 1. Tokenize (worker-tokenizer)
        let tokens = self.tokenizer.encode(prompt)?;
        
        // 2. Generate (YOUR implementation via checkpoints)
        let output = self.model.generate(&tokens, config)?;
        
        // 3. Decode (worker-tokenizer)
        let text = self.tokenizer.decode(&output)?;
        
        // 4. Return (worker-common)
        Ok(InferenceResult::max_tokens(tokens, output, config.seed, 0))
    }
    
    async fn cancel(&self, _: &str) -> Result<()> { Ok(()) }
    fn vram_usage(&self) -> u64 { 0 }  // CPU worker
    fn is_healthy(&self) -> bool { true }
}
```

**Step 2: Use worker-http in main.rs**

```rust
// src/main.rs
use worker_http::{create_router, HttpServer};
use worker_common::startup;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Load model (YOUR implementation)
    let backend = CpuInferenceBackend::load(&model_path)?;
    
    // Callback to pool manager (worker-common)
    startup::callback_ready(&callback_url, &worker_id, 
                           backend.memory_bytes(), port).await?;
    
    // Start HTTP server (worker-http)
    let router = create_router(Arc::new(backend));
    let server = HttpServer::new(addr, router).await?;
    server.run().await?;
    
    Ok(())
}
```

**That's it!** HTTP server is done. Now focus on model implementation via checkpoints.

### API Endpoints (Provided by worker-http)

- **GET /health** - Returns health status
- **POST /execute** - Executes inference with SSE streaming

See `API_INTEGRATION.md` for complete details.

---

## Implementation Strategy

### The Golden Rule (from worker-orcd lessons)

**COMPARE WITH REFERENCE AT EVERY STEP**

```rust
#[test]
fn checkpoint_01_layer_norm() {
    // 1. Load reference output from tinygrad
    let reference = load_tinygrad_output("checkpoint_01.npy");
    
    // 2. Run our implementation
    let our_output = layer_norm.forward(&input);
    
    // 3. Compare within tolerance
    assert_tensors_close(&our_output, &reference, 1e-5);
    
    // 4. If fails, STOP and fix before proceeding
}
```

### Sequential Implementation

#### Week 1: Foundation + HTTP Integration + Checkpoint 1

**Day 1: HTTP Server Integration**
- [ ] Create llorch-cpud crate structure
- [ ] Add worker-http dependency
- [ ] Create stub CpuInferenceBackend implementing InferenceBackend trait
- [ ] Implement main.rs using worker-http (copy from API_INTEGRATION.md)
- [ ] Test GET /health endpoint works
- [ ] Test POST /execute endpoint returns stub response
- [ ] **Verify HTTP API works before model implementation**

**Day 2: CPU Tensor Setup**
- [ ] Set up CPU tensor operations (ndarray)
- [ ] Implement basic tensor ops (matmul, add, etc.)
- [ ] Create tensor wrapper types

**Day 3-4: LayerNorm (Checkpoint 1)**
- [ ] Implement LayerNorm
- [ ] Extract reference output from tinygrad
- [ ] Test until checkpoint 1 passes
- [ ] **DO NOT PROCEED until checkpoint 1 passes**

**Day 5: Embeddings (Phase 2)**
- [ ] Implement token embeddings
- [ ] Implement position embeddings
- [ ] Test against reference

#### Week 2: Attention (Checkpoints 2-5)

**Day 1: QKV Projection (Checkpoint 2)**
- [ ] Implement combined QKV linear layer
- [ ] Implement reshape and split
- [ ] Test until checkpoint 2 passes

**Day 2: KV Cache (Checkpoint 3)**
- [ ] Implement cache initialization
- [ ] Implement cache update
- [ ] Implement cache retrieval
- [ ] Test until checkpoint 3 passes

**Day 3: Attention Computation (Checkpoint 4)**
- [ ] Implement scaled dot-product attention
- [ ] Implement causal masking
- [ ] Test until checkpoint 4 passes

**Day 4: Attention Output (Checkpoint 5)**
- [ ] Implement output projection
- [ ] Test until checkpoint 5 passes

**Day 5: Integration**
- [ ] Complete attention module
- [ ] Test all attention checkpoints together

#### Week 3: FFN + Blocks (Checkpoints 6-7)

**Day 1-2: FFN (Checkpoint 6)**
- [ ] Implement up projection (c_fc)
- [ ] Implement GELU activation
- [ ] Implement down projection (c_proj)
- [ ] Test until checkpoint 6 passes

**Day 3-4: Transformer Block (Checkpoint 7)**
- [ ] Implement pre-norm architecture
- [ ] Implement residual connections
- [ ] Test until checkpoint 7 passes
- [ ] **This validates entire architecture!**

**Day 5: Multiple Blocks**
- [ ] Implement block iteration (24 layers)
- [ ] Test forward pass through all blocks

#### Week 4: Output + Sampling (Checkpoints 8-11)

**Day 1: LM Head (Checkpoints 8-9)**
- [ ] Implement final layer norm
- [ ] Implement lm_head projection
- [ ] Implement weight tying
- [ ] Implement logit selection
- [ ] Test checkpoints 8-9

**Day 2-3: Sampling (Checkpoints 10-11)**
- [ ] Implement temperature=0 (argmax)
- [ ] Implement temperature>0 (softmax)
- [ ] Test checkpoints 10-11

**Day 4-5: Generation Loop**
- [ ] Implement autoregressive generation
- [ ] Implement cache management across iterations
- [ ] Test with multiple prompts

#### Week 5: End-to-End (Checkpoint 12)

**Day 1-3: Integration**
- [ ] Complete model integration
- [ ] Load GPT-2 Medium weights
- [ ] Test checkpoint 12 with "Hello."
- [ ] **If passes: IMPLEMENTATION CORRECT!**

**Day 4-5: Additional Testing**
- [ ] Test multiple prompts
- [ ] Test different temperatures
- [ ] Test batch processing
- [ ] Performance profiling

---

## Checkpoint-Driven Development

### The Process

For each checkpoint:

1. **Read checkpoint file** (e.g., `CHECKPOINT_01_LAYER_NORM.md`)
2. **Study reference implementations** (tinygrad, Candle, Mistral.rs)
3. **Implement component** in llorch-cpud
4. **Extract reference output** from tinygrad
5. **Write test** comparing outputs
6. **Run test** - if fails, debug
7. **Fix until test passes** - no "partial fixes"
8. **Move to next checkpoint** only after pass

### Test Structure

```rust
// tests/checkpoint_01_layer_norm.rs
use llorch_cpud::layers::LayerNorm;
use ndarray::Array2;

#[test]
fn checkpoint_01_matches_tinygrad() {
    // Load reference data
    let input = load_npy("fixtures/checkpoint_01_input.npy");
    let expected = load_npy("fixtures/checkpoint_01_output.npy");
    
    // Run our implementation
    let layer_norm = LayerNorm::new(1024, 1e-5);
    let output = layer_norm.forward(&input);
    
    // Compare
    assert_tensors_close(&output, &expected, 1e-5);
}
```

---

## Reference Data Extraction

### From Tinygrad

```python
# extract_checkpoints.py
import numpy as np
from tinygrad import Tensor
# ... load model ...

# Checkpoint 1: LayerNorm output
ln1_out = self.ln_1(x)
np.save('checkpoint_01_input.npy', x.numpy())
np.save('checkpoint_01_output.npy', ln1_out.numpy())

# Checkpoint 2: QKV
xq, xk, xv = [xqkv[:, :, i, :, :] for i in range(3)]
np.save('checkpoint_02_q.npy', xq.numpy())
np.save('checkpoint_02_k.npy', xk.numpy())
np.save('checkpoint_02_v.npy', xv.numpy())

# ... etc for all checkpoints ...
```

### Test Fixtures Directory

```
llorch-cpud/
└── tests/
    └── fixtures/
        ├── checkpoint_01_input.npy
        ├── checkpoint_01_output.npy
        ├── checkpoint_02_input.npy
        ├── checkpoint_02_q.npy
        ├── checkpoint_02_k.npy
        ├── checkpoint_02_v.npy
        └── ...
```

---

## Integration with Worker Crates

### Using Existing Code

```rust
// llorch-cpud/src/lib.rs
use worker_models::{GPTConfig, ModelAdapter};
use worker_common::{SamplingConfig, InferenceResult};
use worker_tokenizer::Tokenizer;

pub struct LlorcCpuBackend {
    model: GPTModel,
    tokenizer: Tokenizer,
}

impl LlorcCpuBackend {
    pub fn new(config: GPTConfig) -> Result<Self> {
        // Use worker-models GPTConfig
        let model = GPTModel::new(config)?;
        
        // Use worker-tokenizer
        let tokenizer = Tokenizer::from_gguf(...)?;
        
        Ok(Self { model, tokenizer })
    }
    
    pub fn generate(&mut self, prompt: &str, config: SamplingConfig) -> Result<String> {
        // Use worker-common SamplingConfig
        // Implement generation using our CPU backend
        todo!()
    }
}
```

### Reusing 85% of Code

From `worker-crates/`:
- ✅ Tokenization (worker-tokenizer)
- ✅ Sampling config (worker-common)
- ✅ Error types (worker-common)
- ✅ Model adapter trait (worker-models)

Only implement:
- ❌ CPU tensor operations
- ❌ CPU layer implementations
- ❌ CPU forward pass

---

## Success Criteria

### Minimum Viable (Week 5)
- ✅ All 12 checkpoints pass
- ✅ Checkpoint 12 generates correct output
- ✅ Deterministic with temperature=0

### Production Ready (Week 6+)
- ✅ All checkpoints pass
- ✅ Multiple test cases pass
- ✅ Temperature>0 works
- ✅ Batch processing works
- ✅ Performance acceptable
- ✅ Integration with worker-crates complete

---

## Lessons from worker-orcd

### What NOT to Do

❌ "Mathematically correct but output wrong"  
❌ "Partial fix, still investigating"  
❌ "Fixed one component, model still broken"  
❌ "This looks like the root cause"  
❌ "Let's try fixing this and see"

### What TO Do

✅ "Matches reference? Yes or no."  
✅ Compare at every step  
✅ Fix until checkpoint passes  
✅ No forward progress without validation  
✅ Use existing reference implementations

---

## Timeline

| Week | Focus | Checkpoints | Status |
|------|-------|-------------|--------|
| 1 | Foundation + LayerNorm | 1 | ⬜ |
| 2 | Attention | 2-5 | ⬜ |
| 3 | FFN + Blocks | 6-7 | ⬜ |
| 4 | Output + Sampling | 8-11 | ⬜ |
| 5 | End-to-End | 12 | ⬜ |
| 6+ | Polish + Integration | All | ⬜ |

**Estimated completion:** 5-6 weeks with checkpoint-driven development

---

## Next Steps

1. **Read all checkpoint files** in `checkpoints/`
2. **Study reference implementations** (tinygrad, Candle, Mistral.rs)
3. **Set up llorch-cpud crate** structure
4. **Extract reference data** from tinygrad for all checkpoints
5. **Start with Checkpoint 1** (LayerNorm)
6. **Do not proceed** until each checkpoint passes

---

## Resources

- **Spec:** `01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md`
- **Checkpoints:** `checkpoints/MASTER_CHECKLIST.md`
- **Usage Guide:** `VALIDATION_CHECKPOINT_USAGE.md`
- **Reusability Audit:** `WORKER_CRATES_REUSABILITY_AUDIT.md` (NEW)
- **API Integration:** `API_INTEGRATION.md` (NEW)
- **Worker Crates:** `../worker-crates/README.md`
- **Lessons:** `../WORKER_ORCD_LESSONS_LEARNED.md`

---

## Summary

### What We Have
- ✅ Complete GPT-2 MODEL specification (validated)
- ✅ Complete HTTP SERVER (worker-http, ready to use)
- ✅ 12 validation checkpoints (with references)
- ✅ 3 reference implementations (tinygrad, Candle, Mistral.rs)
- ✅ 4 reusable worker crates (approximately 3,871 lines)
- ✅ Lessons from worker-orcd failure

### What We Need
- ⬜ Implement InferenceBackend trait (approximately 200 lines) - Week 1, Day 1
- ⬜ CPU tensor operations (approximately 500 lines) - Week 1, Day 2
- ⬜ GPT-2 model layers (approximately 1,300 lines) - Week 1-5 via checkpoints

### Implementation Split
1. **HTTP Server** (Day 1) - Use worker-http, implement trait
2. **Model** (Week 1-5) - Follow GPT-2 spec via checkpoints

### Success Criteria
- ✅ All 12 checkpoints pass
- ✅ Checkpoint 12 generates exact output
- ✅ Same API as worker-orcd (drop-in replacement)

**The difference between worker-orcd and llorch-cpud:**

**worker-orcd:** 85K lines, 40+ teams, 23 days, still broken  
**llorch-cpud:** 5.7K lines, 70% reused, checkpoint-validated, will succeed

**Key:** Checkpoints + Reference Comparison + Reusable Code = Success

---

Built by TEAM CASCADE 🌊
