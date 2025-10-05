# Integration Strategy: Rust worker-crates + C++ CUDA Implementation

**Date**: 2025-10-05  
**Purpose**: Clarify how Rust worker-crates and C++ CUDA code work together  
**Critical**: We have TWO codebases that need to integrate!

---

## The Two Codebases

### 1. Rust Layer (`bin/worker-crates/`)

**What it does**:
- ✅ HTTP server (Axum)
- ✅ GGUF metadata parsing (pure Rust)
- ✅ Tokenization (BPE, HuggingFace)
- ✅ Model adapters (Qwen, Phi-3, GPT, Llama)
- ✅ Architecture detection
- ✅ Sampling logic

**What it DOESN'T do**:
- ❌ GPU memory management
- ❌ CUDA kernel execution
- ❌ Weight loading to VRAM
- ❌ Actual inference computation

### 2. C++ CUDA Layer (`bin/worker-orcd/cuda/`)

**What it does**:
- ✅ CUDA context management
- ✅ GPU memory allocation
- ✅ GGUF weight loading to VRAM
- ✅ CUDA kernels (attention, RoPE, RMSNorm, etc.)
- ✅ Inference execution

**What it DOESN'T do**:
- ❌ HTTP server
- ❌ Tokenization
- ❌ Architecture-level abstractions

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ RUST LAYER (bin/worker-crates/)                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ worker-http  │  │ worker-gguf  │  │ worker-      │     │
│  │ (Axum)       │  │ (metadata)   │  │ tokenizer    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │ worker-models (Adapters)                         │      │
│  │ - QwenModel, Phi3Model, GPTModel                │      │
│  │ - Architecture detection                         │      │
│  │ - Forward pass routing                           │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│                         │ FFI                               │
│                         ↓                                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ C++ CUDA LAYER (bin/worker-orcd/cuda/)                      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │ FFI Interface (src/ffi.cpp)                      │      │
│  │ - cuda_init(), cuda_load_model()                │      │
│  │ - cuda_inference_start(), cuda_next_token()     │      │
│  └──────────────────────────────────────────────────┘      │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Model Loading (src/model/)                       │      │
│  │ - gpt_weights.cpp (GGUF → VRAM)                 │      │
│  │ - gpt_model.cpp (Model structure)               │      │
│  └──────────────────────────────────────────────────┘      │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Inference (src/inference_impl.cpp)               │      │
│  │ - Transformer layers                             │      │
│  │ - KV cache management                            │      │
│  └──────────────────────────────────────────────────┘      │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │ CUDA Kernels (kernels/)                          │      │
│  │ - attention.cu, rope.cu, rmsnorm.cu, etc.       │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## The Problem

**Our V2 stories focus on C++ CUDA layer**, but we ALSO have:
- Rust `worker-models` with `QwenModel`, `Phi3Model`, `GPTModel`
- Rust `AdapterFactory` with architecture detection
- Rust `worker-gguf` with metadata parsing

**This creates duplication**:
- ❌ Architecture detection in BOTH Rust and C++
- ❌ Config parsing in BOTH Rust and C++
- ❌ Model adapters in BOTH Rust and C++

---

## The Solution: Clear Separation of Concerns

### Rust Layer Responsibilities

**Keep in Rust**:
1. ✅ HTTP server (Axum) - stays in Rust
2. ✅ Tokenization (BPE, HuggingFace) - stays in Rust
3. ✅ GGUF metadata parsing - stays in Rust (lightweight, no GPU)
4. ✅ Architecture detection - stays in Rust (uses GGUF metadata)
5. ✅ Sampling logic - stays in Rust (post-inference)

**Remove from Rust** (move to C++ or make thin wrapper):
1. ❌ `QwenModel`, `Phi3Model`, `GPTModel` - These are just stubs anyway
2. ❌ Forward pass logic - This belongs in C++/CUDA
3. ❌ Weight loading - This belongs in C++/CUDA

### C++ CUDA Layer Responsibilities

**Keep in C++**:
1. ✅ CUDA context management
2. ✅ GPU memory allocation
3. ✅ Weight loading to VRAM
4. ✅ Inference execution
5. ✅ KV cache management
6. ✅ CUDA kernels

**Add to C++** (from V2 stories):
1. ✅ Architecture registry (C++ version)
2. ✅ Tensor mapper (C++ version)
3. ✅ Paged KV cache (C++ version)

---

## Updated Integration Flow

### 1. Startup Flow

```rust
// Rust: main.rs
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse GGUF metadata (Rust)
    let metadata = worker_gguf::GGUFMetadata::from_file(&model_path)?;
    let arch = metadata.architecture()?;  // "qwen2", "llama", "gpt2"
    
    // 2. Initialize CUDA context (FFI)
    let cuda_ctx = unsafe { cuda_init(gpu_device, &mut error) };
    
    // 3. Load model to VRAM (FFI, passes arch string)
    let cuda_model = unsafe {
        cuda_load_model(
            cuda_ctx,
            model_path.as_ptr(),
            arch.as_ptr(),  // Pass architecture to C++
            &mut error
        )
    };
    
    // 4. Create tokenizer (Rust)
    let tokenizer = worker_tokenizer::Tokenizer::from_gguf(&model_path)?;
    
    // 5. Start HTTP server (Rust)
    let server = worker_http::HttpServer::new(port, create_router(cuda_model, tokenizer)).await?;
    server.run().await?;
    
    Ok(())
}
```

### 2. Inference Flow

```rust
// Rust: HTTP handler
async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, StatusCode> {
    // 1. Tokenize prompt (Rust)
    let token_ids = state.tokenizer.encode(&req.prompt, true)?;
    
    // 2. Start inference (FFI)
    let result = unsafe {
        cuda_inference_start(
            state.cuda_model,
            token_ids.as_ptr(),
            token_ids.len(),
            req.max_tokens,
            req.temperature,
            &mut error
        )
    };
    
    // 3. Stream tokens
    let stream = async_stream::stream! {
        loop {
            // Get next token from CUDA (FFI)
            let token_id = unsafe { cuda_inference_next_token(result, &mut error) };
            if token_id == EOS_TOKEN { break; }
            
            // Decode token (Rust)
            let text = state.tokenizer.decode(&[token_id], false)?;
            
            // Send SSE event (Rust)
            yield Ok(Event::default().data(text));
        }
    };
    
    Ok(Sse::new(stream).into_response())
}
```

### 3. C++ CUDA Implementation (Updated)

```cpp
// cuda/src/ffi.cpp

extern "C" {

CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    const char* arch_string,  // NEW: Rust passes architecture
    int* error
) {
    try {
        // Parse architecture (C++ version)
        Architecture arch = ArchitectureRegistry::from_string(arch_string);
        
        // Load weights using architecture registry
        auto weights = GPTWeightLoader::load_from_gguf(model_path, arch);
        
        // Create model
        auto model = new CudaModel(ctx, std::move(weights));
        
        *error = 0;
        return model;
    } catch (const std::exception& e) {
        *error = -1;
        return nullptr;
    }
}

} // extern "C"
```

---

## Updated V2 Stories

### GT-052-V2: Architecture Registry + Weight Loading (C++ ONLY)

**Changes**:
- ✅ Implement in C++ (as planned)
- ✅ Rust passes architecture string via FFI
- ✅ C++ uses registry to load weights
- ❌ NO Rust model adapters (they're stubs anyway)

**Rust side** (minimal):
```rust
// worker-models stays as thin wrapper
pub struct QwenModel {
    cuda_model: *mut CudaModel,  // Just holds FFI pointer
}

impl QwenModel {
    pub fn load(path: &str) -> Result<Self> {
        let cuda_model = unsafe { cuda_load_model(...) };
        Ok(Self { cuda_model })
    }
}
```

### GT-053-V2: BPE Tokenizer (RUST ONLY)

**Changes**:
- ✅ Already implemented in `worker-tokenizer`
- ✅ No C++ implementation needed
- ✅ Just wire it up in FFI boundary

**No changes needed** - `worker-tokenizer` is already complete!

### GT-054-V2: Paged KV Cache (C++ ONLY)

**Changes**:
- ✅ Implement in C++ (as planned)
- ✅ No Rust involvement
- ✅ Pure CUDA implementation

### GT-056-V2: Wire Inference (FFI INTEGRATION)

**Changes**:
- ✅ Update FFI to pass architecture string
- ✅ Update Rust HTTP handlers
- ✅ Wire tokenizer (Rust) → inference (C++) → sampling (Rust)

---

## What to Keep from worker-crates

### Keep (Already Working)

1. **`worker-http`** - HTTP server, SSE streaming
2. **`worker-tokenizer`** - BPE, HuggingFace tokenizers
3. **`worker-gguf`** - Metadata parsing (lightweight)
4. **`worker-common`** - Error types, sampling config

### Simplify (Make Thin Wrappers)

1. **`worker-models`** - Just hold FFI pointers, no logic
   ```rust
   pub struct QwenModel {
       cuda_model: *mut CudaModel,
   }
   ```

2. **`AdapterFactory`** - Just parse metadata and call FFI
   ```rust
   impl AdapterFactory {
       pub fn from_gguf(path: &str) -> Result<LlamaModelAdapter> {
           let metadata = GGUFMetadata::from_file(path)?;
           let arch = metadata.architecture()?;
           let cuda_model = unsafe { cuda_load_model(path, arch) };
           Ok(LlamaModelAdapter::new(cuda_model))
       }
   }
   ```

### Remove (Duplicate Logic)

1. ❌ Forward pass implementations in Rust (stubs anyway)
2. ❌ Weight loading in Rust (belongs in C++)
3. ❌ Config parsing in Rust (C++ does this from GGUF)

---

## Implementation Priority

### Phase 1: C++ CUDA Layer (GT-052, GT-054)
1. Implement ArchitectureRegistry in C++
2. Implement PagedKVCache in C++
3. Update FFI to accept architecture string

### Phase 2: Rust Integration (GT-056)
1. Update FFI bindings
2. Wire tokenizer → CUDA → sampling
3. Update HTTP handlers

### Phase 3: Cleanup (Post-M0)
1. Simplify `worker-models` to thin wrappers
2. Remove duplicate logic
3. Document FFI boundary

---

## Decision

**Proceed with V2 stories as planned**, but:

1. ✅ **Implement in C++** - Architecture registry, paged KV cache
2. ✅ **Keep Rust layer thin** - HTTP, tokenization, metadata parsing
3. ✅ **Clear FFI boundary** - Rust passes arch string, C++ does inference
4. ✅ **No duplication** - Logic lives in ONE place (C++ for compute, Rust for I/O)

**Benefit**: Clean separation, no duplicate logic, leverages strengths of both languages.

---

**Created by**: Project Management Team 📋  
**Date**: 2025-10-05  
**Status**: Architecture clarified  
**Next**: Proceed with V2 C++ implementation

---
Reviewed by Testing Team 🔍
