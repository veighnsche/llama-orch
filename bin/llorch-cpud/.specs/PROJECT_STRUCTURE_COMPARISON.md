# Project Structure Comparison: Candle vs Mistral.rs vs llorch-cpud

**Date:** 2025-10-08  
**Purpose:** Understand architectural decisions across three Rust ML projects  
**Status:** Analysis

---

## Executive Summary

| Project | Purpose | Architecture | Key Insight |
|---------|---------|--------------|-------------|
| **Candle** | ML Framework (like PyTorch) | Library-first, multi-backend | Provides building blocks, not applications |
| **Mistral.rs** | Production Inference Server | Application-first, feature-rich | Complete inference solution with many models |
| **llorch-cpud** | Single-Model Worker Daemon | Worker-first, minimal | One model, one process, HTTP API |

---

## 1. Candle: The ML Framework

### Purpose
**"PyTorch for Rust"** - A machine learning framework, not an inference server.

### Structure
```
candle/
├── candle-core/              # Tensor operations (like torch.Tensor)
│   ├── backend.rs            # CPU/CUDA/Metal backends
│   ├── tensor.rs             # Core tensor type
│   ├── op.rs                 # Operations (matmul, conv, etc.)
│   └── device.rs             # Device abstraction
├── candle-nn/                # Neural network layers (like torch.nn)
│   ├── layer_norm.rs
│   ├── linear.rs
│   ├── conv.rs
│   └── ...
├── candle-transformers/      # Transformer models (like transformers library)
│   ├── models/               # 100+ model implementations
│   │   ├── llama.rs
│   │   ├── gpt2.rs
│   │   ├── mistral.rs
│   │   └── ...
│   └── generation/           # Text generation utilities
└── candle-examples/          # Example applications
    ├── llama/                # Example: Run Llama
    ├── gpt2/                 # Example: Run GPT-2
    └── ...
```

### Key Insights

**1. Library, Not Application**
- Candle is a **framework** like PyTorch
- You build applications **on top** of Candle
- No HTTP server, no worker daemon
- Just provides tensor operations and model implementations

**2. Multi-Backend from Day 1**
- `candle-core` abstracts CPU/CUDA/Metal
- Same code runs on any backend
- Backend is a runtime choice, not compile-time

**3. Model Implementations are Examples**
- `candle-transformers/models/` has 100+ models
- These are **reference implementations**
- Users copy/modify for their needs
- Not production-ready servers

**4. VarBuilder Pattern**
- Uses `VarBuilder` for weight loading
- `.pp("layer")` = "push prefix" (like cd into directory)
- Matches PyTorch's `state_dict` naming exactly
- Example: `vb.pp("transformer").pp("h").pp("0")` → `transformer.h.0.*`

**Why This Structure:**
- ✅ Reusable across many applications
- ✅ Easy to add new models
- ✅ Backend-agnostic
- ❌ Not a production server
- ❌ No HTTP API
- ❌ No worker management

---

## 2. Mistral.rs: The Production Server

### Purpose
**"Complete inference solution"** - Production-ready server with many features.

### Structure
```
mistral.rs/
├── mistralrs-core/           # Core inference engine
│   ├── models/               # 24 model families
│   │   ├── llama.rs
│   │   ├── mistral.rs
│   │   ├── phi3.rs
│   │   └── ...
│   ├── pipeline/             # Inference pipelines
│   ├── engine/               # Request scheduling
│   ├── kv_cache/             # KV cache management
│   ├── attention/            # Attention implementations
│   ├── layers.rs             # Common layers (90KB file!)
│   ├── sampler.rs            # Sampling strategies
│   └── scheduler/            # Request scheduling
├── mistralrs-vision/         # Vision model support
├── mistralrs-quant/          # Quantization (ISQ, GGUF, GPTQ, AWQ)
├── mistralrs-paged-attn/     # Paged attention (like vLLM)
├── mistralrs-server/         # HTTP server + CLI
│   ├── main.rs               # Server entry point
│   └── ...
├── mistralrs-pyo3/           # Python bindings
└── examples/                 # Usage examples
```

### Key Insights

**1. Application, Not Library**
- Mistral.rs is a **complete server**
- Has HTTP API, CLI, Python bindings
- Production-ready out of the box
- Users run it, not build on it

**2. Feature-Rich**
- Supports 24+ model families
- Quantization (GGUF, GPTQ, AWQ, FP8, HQQ)
- Paged attention (memory optimization)
- Vision models (multimodal)
- Diffusion models (image generation)
- Speech models
- LoRA/X-LoRA adapters
- Mixture of Experts (AnyMoE)
- Tool calling
- Streaming

**3. Monolithic Core**
- `layers.rs` is 90KB (2000+ lines)
- Everything in one place
- Not modular like Candle
- Optimized for completeness, not reusability

**4. Built on Candle**
- Uses `candle-core` for tensors
- Uses `candle-nn` for layers
- Adds production features on top
- VarBuilder pattern for weight loading

**5. Request Scheduling**
- Has `engine/` for request management
- Has `scheduler/` for batching
- Has `sequence.rs` for tracking requests
- Production-grade orchestration

**Why This Structure:**
- ✅ Complete solution
- ✅ Production-ready
- ✅ Many features
- ✅ Easy to deploy
- ❌ Hard to customize
- ❌ Monolithic
- ❌ Not a framework

---

## 3. llorch-cpud: The Minimal Worker

### Purpose
**"Single-model worker daemon"** - One model, one process, HTTP API, orchestrated by pool-managerd.

### Structure
```
llorch-cpud/
├── src/
│   ├── main.rs               # HTTP server (uses worker-http)
│   ├── backend/
│   │   └── cpu_backend.rs    # InferenceBackend impl
│   ├── model/
│   │   └── gpt2.rs           # ONE model (GPT-2)
│   ├── layers/               # Pure implementations
│   │   ├── layer_norm.rs
│   │   ├── embedding.rs
│   │   ├── attention/        # Split into focused files
│   │   │   ├── qkv.rs
│   │   │   ├── scores.rs
│   │   │   └── output.rs
│   │   ├── ffn.rs
│   │   └── transformer.rs
│   ├── cache/                # Top-level (room to grow)
│   │   └── kv_cache.rs
│   └── tensor/
│       └── ops.rs            # CPU tensor operations
└── tests/
    └── checkpoint_*.rs       # Validation tests
```

### Key Insights

**1. Worker, Not Server**
- llorch-cpud is a **worker daemon**
- Spawned by pool-managerd
- One model per process
- HTTP API for orchestration
- Not user-facing

**2. Minimal and Focused**
- ONE model (GPT-2)
- ONE backend (CPU)
- ONE purpose (inference)
- No quantization, no vision, no diffusion
- Just text generation

**3. Checkpoint-Driven Development**
- 13 validation checkpoints
- Each checkpoint validates one component
- Reference implementations (tinygrad, Candle, Mistral.rs)
- Incremental validation

**4. Modular Layers**
- Each layer in its own file
- Attention split into 4 files (qkv, cache, scores, output)
- Cache is top-level (room for optimization)
- Clean separation

**5. Worker-Crates Pattern**
- Reuses worker-http, worker-common, worker-tokenizer, worker-models
- HTTP server is external (worker-http)
- Model implementation is internal (pure)
- Clear boundaries

**6. Single-Threaded**
- Uses `tokio::main(flavor = "current_thread")`
- No thread pool
- Sequential request processing
- Optimal for CPU-bound inference

**Why This Structure:**
- ✅ Minimal and focused
- ✅ Easy to validate (checkpoints)
- ✅ Modular layers
- ✅ Reuses infrastructure (worker-crates)
- ✅ Orchestrated by pool-managerd
- ❌ Only one model
- ❌ Only CPU
- ❌ Not standalone (needs pool-managerd)

---

## Architectural Comparison

### Candle: Framework Architecture

```
Application (your code)
    ↓
candle-transformers (models)
    ↓
candle-nn (layers)
    ↓
candle-core (tensors)
    ↓
Backend (CPU/CUDA/Metal)
```

**Philosophy:** Provide building blocks, let users build applications.

---

### Mistral.rs: Monolithic Architecture

```
Client (HTTP/Python)
    ↓
mistralrs-server (API)
    ↓
mistralrs-core (engine + models + layers)
    ↓
candle-core (tensors)
    ↓
Backend (CPU/CUDA/Metal)
```

**Philosophy:** Complete solution, batteries included.

---

### llorch-cpud: Worker Architecture

```
orchestratord (brain)
    ↓
pool-managerd (control plane)
    ↓
llorch-cpud (worker)
    ↓ uses
worker-http (HTTP server)
worker-common (types)
worker-tokenizer (tokenization)
worker-models (configs)
    ↓
llorch-cpud/layers (pure implementation)
    ↓
ndarray (CPU tensors)
```

**Philosophy:** Minimal worker, orchestrated by system.

---

## Key Differences

### 1. Scope

| Aspect | Candle | Mistral.rs | llorch-cpud |
|--------|--------|------------|-------------|
| **Purpose** | Framework | Server | Worker |
| **Models** | 100+ (examples) | 24+ (built-in) | 1 (GPT-2) |
| **Backends** | CPU/CUDA/Metal | CPU/CUDA/Metal | CPU only |
| **HTTP API** | No | Yes | Yes (via worker-http) |
| **Standalone** | No | Yes | No (needs pool-managerd) |

### 2. Layer Organization

**Candle:**
```
candle-nn/
├── layer_norm.rs
├── linear.rs
├── conv.rs
└── ...
```
- One file per layer type
- Reusable across models
- Generic implementations

**Mistral.rs:**
```
mistralrs-core/src/
├── layers.rs (90KB!)
└── models/
    ├── llama.rs
    ├── mistral.rs
    └── ...
```
- Monolithic `layers.rs`
- Model-specific implementations
- Everything in one place

**llorch-cpud:**
```
src/layers/
├── layer_norm.rs
├── attention/
│   ├── qkv.rs
│   ├── scores.rs
│   └── output.rs
├── ffn.rs
└── transformer.rs
```
- One file per component
- Attention split into focused files
- Checkpoint-aligned

### 3. Cache Organization

**Candle:**
- No explicit cache module
- KV cache is part of model implementation
- Embedded in attention

**Mistral.rs:**
```
mistralrs-core/src/
├── kv_cache/
│   ├── normal.rs
│   ├── paged.rs
│   └── ...
└── paged_attention/
```
- Dedicated cache module
- Multiple cache strategies
- Paged attention support

**llorch-cpud:**
```
src/cache/
├── mod.rs
└── kv_cache.rs
```
- Top-level cache module
- Simple implementation for MVP
- Room to grow (paged attention later)

### 4. Weight Loading

**Candle:**
```rust
let vb = VarBuilder::from_safetensors(...);
let ln = layer_norm(size, vb.pp("ln_1"))?;
```
- VarBuilder pattern
- `.pp()` = push prefix
- Matches PyTorch naming

**Mistral.rs:**
```rust
let vb = ShardedVarBuilder::from_safetensors(...);
let ln = layer_norm(size, config, vb.pp("ln_1"))?;
```
- ShardedVarBuilder (for multi-GPU)
- Same pattern as Candle
- Adds sharding support

**llorch-cpud:**
```rust
// Load from GGUF
let weights = load_gguf(&model_path)?;
let ln = LayerNorm::new(weights.ln_1_weight, weights.ln_1_bias, 1e-5);
```
- Direct weight loading
- No VarBuilder (simpler)
- GGUF format only

---

## Why llorch-cpud is Different

### 1. Worker, Not Server

**Candle/Mistral.rs:**
- Standalone applications
- Users run them directly
- Self-contained

**llorch-cpud:**
- Worker daemon
- Spawned by pool-managerd
- Part of larger system

### 2. Single Model Focus

**Candle:**
- Framework for any model
- Users implement their models

**Mistral.rs:**
- 24+ models built-in
- Users choose at runtime

**llorch-cpud:**
- ONE model (GPT-2)
- Hardcoded at compile time
- Optimized for that model

### 3. Checkpoint-Driven

**Candle/Mistral.rs:**
- No explicit checkpoints
- Test by running inference
- Compare with reference

**llorch-cpud:**
- 13 validation checkpoints
- Each component validated separately
- Incremental development

### 4. Modular Layers

**Candle:**
- Layers in `candle-nn`
- Reusable across models

**Mistral.rs:**
- Monolithic `layers.rs`
- Everything in one file

**llorch-cpud:**
- Each layer in own file
- Attention split into 4 files
- Cache is top-level

### 5. Worker-Crates Pattern

**Candle/Mistral.rs:**
- Self-contained
- No external worker infrastructure

**llorch-cpud:**
- Uses worker-http for HTTP
- Uses worker-common for types
- Uses worker-tokenizer for tokenization
- Uses worker-models for configs
- Reuses infrastructure

---

## Lessons from Candle

### 1. VarBuilder Pattern
**What:** `.pp("layer")` pushes prefix, matches PyTorch naming

**Why:** Makes weight loading match PyTorch exactly

**llorch-cpud:** Doesn't use this (simpler, GGUF-only)

### 2. Backend Abstraction
**What:** `Device` enum (CPU/CUDA/Metal)

**Why:** Same code, multiple backends

**llorch-cpud:** CPU-only (no abstraction needed)

### 3. Modular Layers
**What:** `candle-nn` has reusable layers

**Why:** Build any model from components

**llorch-cpud:** Similar, but focused on GPT-2

---

## Lessons from Mistral.rs

### 1. Production Features
**What:** Paged attention, quantization, scheduling

**Why:** Production inference needs these

**llorch-cpud:** MVP first, add later

### 2. Monolithic Core
**What:** `layers.rs` is 90KB

**Why:** Everything in one place for performance

**llorch-cpud:** Opposite - split into files for clarity

### 3. Multiple Cache Strategies
**What:** `kv_cache/` has normal, paged, etc.

**Why:** Different use cases need different strategies

**llorch-cpud:** Simple cache now, room to grow

### 4. Request Scheduling
**What:** `engine/` and `scheduler/` for batching

**Why:** Production needs request management

**llorch-cpud:** Orchestratord handles this

---

## Why llorch-cpud's Structure Makes Sense

### 1. Worker-First Design
- Part of larger system (llama-orch)
- Orchestrated by pool-managerd
- Not standalone

### 2. Minimal and Focused
- ONE model (GPT-2)
- ONE backend (CPU)
- Easy to validate

### 3. Checkpoint-Driven
- Incremental validation
- Catch errors early
- Reference implementations

### 4. Modular Layers
- Easy to understand
- Easy to test
- Easy to validate

### 5. Reuses Infrastructure
- worker-http for HTTP
- worker-common for types
- worker-tokenizer for tokenization
- Focus on model, not infrastructure

### 6. Room to Grow
- Cache is top-level (can add paged attention)
- Layers are modular (can optimize)
- Single-threaded (can add multi-model later)

---

## Summary Table

| Aspect | Candle | Mistral.rs | llorch-cpud |
|--------|--------|------------|-------------|
| **Type** | Framework | Server | Worker |
| **Purpose** | Build ML apps | Run inference | Execute one model |
| **Models** | 100+ examples | 24+ built-in | 1 hardcoded |
| **Backends** | CPU/CUDA/Metal | CPU/CUDA/Metal | CPU only |
| **HTTP API** | No | Yes | Yes (worker-http) |
| **Standalone** | No | Yes | No |
| **Layer Org** | Modular | Monolithic | Modular |
| **Cache** | Embedded | Dedicated module | Top-level module |
| **Weight Loading** | VarBuilder | ShardedVarBuilder | Direct GGUF |
| **Validation** | Manual | Manual | 13 checkpoints |
| **Threading** | Multi | Multi | Single |
| **Orchestration** | None | Built-in | External (pool-managerd) |

---

## Conclusion

**Candle:** Framework for building ML applications  
**Mistral.rs:** Complete production inference server  
**llorch-cpud:** Minimal worker daemon for orchestrated system

**llorch-cpud's structure is correct because:**
- ✅ It's a worker, not a server
- ✅ It's minimal and focused
- ✅ It's checkpoint-driven
- ✅ It reuses infrastructure
- ✅ It's part of larger system

**Different purposes require different structures!**

---

Built by TEAM CASCADE 🌊
