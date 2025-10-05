# Roadmap: Get Haiku Test Working with Real Inference

**Date**: 2025-10-05  
**Goal**: Make the haiku test pass with actual GPU inference  
**Current Status**: GT-051-REFACTOR ✅ COMPLETE

---

## The Haiku Test

**What it does**:
```rust
#[tokio::test]
async fn test_haiku_generation() {
    let response = client
        .post("/v1/generate")
        .json(&json!({
            "prompt": "Write a haiku about",
            "max_tokens": 20,
            "temperature": 0.7
        }))
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), 200);
    // Should generate actual text, not stub
}
```

**What it needs**:
1. ✅ Parse GGUF metadata (Rust) - **DONE!**
2. ⬜ Load weights to VRAM (C++)
3. ⬜ Tokenize prompt (Rust)
4. ⬜ Run inference (C++)
5. ⬜ Decode tokens (Rust)
6. ⬜ Stream response (Rust)

---

## Current State

### ✅ What Works

1. **GGUF Parsing** (Rust)
   - ✅ Parse Qwen2.5-0.5B metadata
   - ✅ Extract config (vocab_size, hidden_dim, etc.)
   - ✅ Architecture detection

2. **HTTP Server** (Rust)
   - ✅ Axum server running
   - ✅ `/v1/generate` endpoint
   - ✅ SSE streaming

3. **Tokenizer** (Rust)
   - ✅ BPE implementation complete
   - ✅ GGUF vocab parsing
   - ✅ Streaming decoder

4. **CUDA Kernels** (C++)
   - ✅ RoPE, RMSNorm, GQA attention
   - ✅ SwiGLU, embedding, sampling
   - ✅ All kernels implemented

### ❌ What's Missing

1. **Weight Loading to VRAM** (C++)
   - ❌ Load tensors from GGUF to GPU
   - ❌ Allocate GPU memory
   - ❌ Copy weights to VRAM

2. **Transformer Execution** (C++)
   - ❌ Wire transformer layers
   - ❌ KV cache management
   - ❌ Forward pass

3. **FFI Integration** (Rust ↔ C++)
   - ❌ Pass config from Rust to C++
   - ❌ Call C++ inference from Rust
   - ❌ Return tokens to Rust

---

## Stories Needed (In Order)

### 1. GT-052-SIMPLIFIED: Weight Loading (4-6 hours)

**What**: Load GGUF weights to GPU VRAM

**Tasks**:
- [ ] C++ receives config from Rust via FFI
- [ ] Open GGUF file (simple mmap or read)
- [ ] Find tensors by name
- [ ] Allocate GPU memory for each tensor
- [ ] Copy tensor data to VRAM
- [ ] Return model handle to Rust

**FFI**:
```rust
// Rust side
let config = worker_gguf::GGUFMetadata::from_file(&path)?;
let cuda_model = unsafe {
    cuda_load_model(
        ctx,
        path.as_ptr(),
        config.vocab_size() as u32,
        config.hidden_dim() as u32,
        config.num_layers() as u32,
        // ... more config
    )
};
```

```cpp
// C++ side
extern "C" {
    CudaModel* cuda_load_model(
        CudaContext* ctx,
        const char* path,
        uint32_t vocab_size,
        uint32_t hidden_dim,
        uint32_t num_layers,
        // ... more config
    ) {
        // Just load weights to VRAM
        auto model = load_weights_to_vram(path, config);
        return model;
    }
}
```

**Output**: Model weights loaded in VRAM ✅

---

### 2. GT-053: Tokenizer Integration (1-2 hours)

**What**: Wire up existing Rust tokenizer

**Tasks**:
- [ ] Load tokenizer from GGUF metadata
- [ ] Create `Tokenizer::from_gguf()` method
- [ ] Encode prompt → token IDs
- [ ] Decode token IDs → text
- [ ] Handle special tokens (BOS, EOS)

**Code**:
```rust
// Already mostly done in worker-tokenizer!
let tokenizer = Tokenizer::from_gguf(&model_path)?;
let token_ids = tokenizer.encode("Write a haiku about", true)?;
// [123, 456, 789, ...]
```

**Output**: Tokenization works ✅

---

### 3. GT-054-SIMPLIFIED: Basic Transformer (4-6 hours)

**What**: Wire transformer layers with simple KV cache

**Tasks**:
- [ ] Create `GPTTransformerLayer` class
- [ ] Wire: Embedding → RMSNorm → GQA → Residual → RMSNorm → SwiGLU → Residual
- [ ] Simple contiguous KV cache (not paged yet)
- [ ] Forward pass for single token
- [ ] LM head projection

**Code**:
```cpp
class GPTTransformerLayer {
    void forward(
        const half* input,      // [batch, seq_len, hidden_dim]
        half* output,           // [batch, seq_len, hidden_dim]
        half* k_cache,          // [batch, cache_len, hidden_dim]
        half* v_cache,          // [batch, cache_len, hidden_dim]
        int cache_len
    ) {
        // 1. RMSNorm
        cuda_rmsnorm(input, attn_norm, ...);
        
        // 2. GQA Attention
        cuda_gqa_attention(normed, k_cache, v_cache, ...);
        
        // 3. Residual
        cuda_residual_add(input, attn_out, ...);
        
        // 4. FFN (RMSNorm + SwiGLU)
        cuda_rmsnorm(residual, ffn_norm, ...);
        cuda_swiglu(normed, ffn_gate, ffn_up, ...);
        
        // 5. Residual
        cuda_residual_add(residual, ffn_out, output);
    }
};
```

**Output**: Transformer forward pass works ✅

---

### 4. GT-055: LM Head + Sampling (2-3 hours)

**What**: Project to vocab and sample next token

**Tasks**:
- [ ] LM head projection (cuBLAS GEMM)
- [ ] Softmax + temperature scaling
- [ ] Top-k/top-p sampling
- [ ] Return token ID

**Code**:
```cpp
int sample_next_token(
    const half* hidden_state,  // [hidden_dim]
    const half* lm_head,       // [vocab_size, hidden_dim]
    float temperature
) {
    // 1. Project to vocab: logits = hidden @ lm_head^T
    cuda_gemm(hidden_state, lm_head, logits, ...);
    
    // 2. Sample
    int token_id = cuda_sample_token(logits, temperature, ...);
    
    return token_id;
}
```

**Output**: Token sampling works ✅

---

### 5. GT-056: Wire Full Inference (3-4 hours)

**What**: Connect all pieces end-to-end

**Tasks**:
- [ ] Prefill phase (process full prompt)
- [ ] Decode phase (generate tokens one by one)
- [ ] KV cache management
- [ ] Stop on EOS token
- [ ] Return token stream to Rust

**FFI**:
```rust
// Rust side
let mut result = CudaBackend::inference_start(
    &model,
    &token_ids,
    max_tokens,
    temperature,
    seed
)?;

while let Some(token_id) = CudaBackend::inference_next_token(&mut result)? {
    let text = tokenizer.decode(&[token_id])?;
    yield text;
}
```

```cpp
// C++ side
extern "C" {
    InferenceResult* cuda_inference_start(
        CudaModel* model,
        const uint32_t* token_ids,
        size_t num_tokens,
        size_t max_tokens,
        float temperature,
        uint64_t seed
    ) {
        // Prefill: process all input tokens
        auto result = new InferenceResult();
        result->prefill(token_ids, num_tokens);
        return result;
    }
    
    int cuda_inference_next_token(InferenceResult* result) {
        // Decode: generate one token
        int token_id = result->decode_one_token();
        return token_id;
    }
}
```

**Output**: Full inference works ✅

---

### 6. GT-057: Test & Polish (1-2 hours)

**What**: Make haiku test pass

**Tasks**:
- [ ] Run haiku test
- [ ] Fix any bugs
- [ ] Verify output quality
- [ ] Add logging
- [ ] Performance check

**Output**: Haiku test passes ✅

---

## Timeline to Haiku Test

| Story | Hours | What |
|-------|-------|------|
| GT-051-REFACTOR | ✅ DONE | GGUF parser (Rust) |
| GT-052-SIMPLIFIED | 4-6h | Weight loading (C++) |
| GT-053 | 1-2h | Tokenizer (Rust) |
| GT-054-SIMPLIFIED | 4-6h | Transformer (C++) |
| GT-055 | 2-3h | LM head + sampling (C++) |
| GT-056 | 3-4h | Wire inference (FFI) |
| GT-057 | 1-2h | Test & polish |
| **TOTAL** | **15-23h** | **2-3 days** |

---

## Simplified vs Original Plan

### Original V2 Plan
- Architecture registry (C++)
- Paged KV cache (C++)
- Complex tensor mapping
- **Total**: 28-38 hours

### Simplified Plan (For Haiku)
- Simple weight loading
- Simple contiguous KV cache
- Direct tensor loading
- **Total**: 15-23 hours

**Difference**: -13 to -15 hours saved!

**Why simpler**:
- ✅ Rust does GGUF parsing (no C++ registry needed)
- ✅ Contiguous KV cache (paged cache later)
- ✅ Direct tensor loading (no complex mapping)
- ✅ Single model (Qwen2.5-0.5B only)

---

## What Haiku Test Actually Needs

### Minimum Viable Inference

1. **Load Model** (GT-052)
   - Read GGUF tensors
   - Copy to VRAM
   - ~500 MB for Qwen2.5-0.5B

2. **Tokenize** (GT-053)
   - "Write a haiku about" → `[123, 456, 789]`
   - Already implemented in Rust!

3. **Prefill** (GT-054)
   - Process input tokens
   - Build KV cache
   - Get hidden state

4. **Decode** (GT-055)
   - Generate 20 tokens
   - Sample from logits
   - Update KV cache

5. **Detokenize** (GT-053)
   - `[999, 888, 777]` → "code\nin\nthe"
   - Already implemented in Rust!

6. **Stream** (GT-056)
   - SSE events
   - Already implemented in Rust!

---

## Critical Path

```
GT-051-REFACTOR (✅ DONE)
    ↓
GT-052-SIMPLIFIED (Weight Loading)
    ↓
GT-053 (Tokenizer) ← Can do in parallel
    ↓
GT-054-SIMPLIFIED (Transformer)
    ↓
GT-055 (LM Head)
    ↓
GT-056 (Wire Inference)
    ↓
GT-057 (Test)
    ↓
🎉 HAIKU TEST PASSES
```

---

## What Can We Skip (For Now)

### Not Needed for Haiku Test

1. ❌ **Paged KV Cache** - Contiguous is fine for single request
2. ❌ **Architecture Registry** - Rust already has it
3. ❌ **Batch Inference** - Single request only
4. ❌ **Multiple Models** - Just Qwen2.5-0.5B
5. ❌ **Prefix Caching** - Not needed yet
6. ❌ **Quantization Kernels** - GGUF already quantized

### Can Add Later (M1+)

- Paged KV cache (for batching)
- Multiple model support
- Phi-3, GPT-OSS-20B support
- Performance optimizations
- Prefix caching

---

## Decision Point

### Option A: Simplified Path (RECOMMENDED)

**Goal**: Get haiku test passing ASAP

**Approach**:
- Simple weight loading
- Contiguous KV cache
- Single model (Qwen2.5-0.5B)
- No fancy features

**Time**: 15-23 hours (2-3 days)

**Pros**:
- ✅ Fast to M0
- ✅ Proves inference works
- ✅ Can refactor later

**Cons**:
- ⚠️ Not production-ready
- ⚠️ No batching
- ⚠️ Some refactor needed for M1

### Option B: Full V2 Path

**Goal**: Production-ready from day 1

**Approach**:
- Architecture registry (C++)
- Paged KV cache
- Full tensor mapping
- Batch-ready

**Time**: 28-38 hours (4-5 days)

**Pros**:
- ✅ Production-ready
- ✅ No refactor needed
- ✅ Supports batching

**Cons**:
- ⚠️ Takes longer
- ⚠️ More complex
- ⚠️ Haiku test delayed

---

## Recommendation

**Go with Option A: Simplified Path**

**Rationale**:
1. ✅ **Prove it works** - Get haiku test passing quickly
2. ✅ **Iterate** - Can add features incrementally
3. ✅ **Learn** - Understand performance before optimizing
4. ✅ **Ship M0** - Meet deadline
5. ✅ **Refactor M1** - Add paged cache, batching, etc.

**Next Story**: GT-052-SIMPLIFIED (4-6 hours)

---

**Ready to start GT-052-SIMPLIFIED?** Let's get weights loaded to VRAM! 🚀

---
Created by Project Management Team 📋
