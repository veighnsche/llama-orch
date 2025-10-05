# M0 Worker Inference Pipeline: Complete Analysis

**Date**: 2025-10-03  
**Scope**: Deep architectural analysis of inference execution flow  
**Focus**: End-to-end pipeline for all three target models

---

## Executive Summary

### Architectural Status: ✅ SOUND

The M0 worker spec **correctly addresses** the architectural gaps identified in the gap analysis. The ModelAdapter pattern provides clean separation between Llama-style (Qwen/Phi-3) and GPT-style (GPT-OSS-20B) execution paths.

**Key Finding**: No critical architectural gaps remain. The spec is implementation-ready with clear execution flows for all three models.

---

## 1. Complete Inference Pipeline Steps

### 1.1 High-Level Flow (All Models)

```
HTTP Request → Tokenization → VRAM Allocation → Forward Pass → Sampling → Detokenization → SSE Stream
```

**Detailed Breakdown**:

1. **HTTP Request Handling** (Rust layer)
2. **Request Validation** (Rust layer)
3. **Tokenization** (Rust layer, architecture-specific backend)
4. **VRAM Allocation** (CUDA layer, KV cache + buffers)
5. **Forward Pass** (CUDA layer, architecture-specific adapter)
6. **Sampling** (CUDA→CPU, temperature-based)
7. **Detokenization** (Rust layer)
8. **SSE Streaming** (Rust layer)
9. **Cleanup** (CUDA layer, free buffers)

---

## 2. Model-Specific Execution Paths

### 2.1 Qwen2.5-0.5B-Instruct (Llama-style)

#### Architecture Detection
```cpp
// From GGUF metadata
general.architecture = "llama"
→ Create LlamaModelAdapter
```

#### Tokenization
```rust
// GGUF byte-BPE backend
let tokenizer = GgufBpeTokenizer::from_gguf_metadata(gguf);
let token_ids = tokenizer.encode(prompt);
// BOS/EOS handling from GGUF metadata
```

#### Forward Pass (LlamaModelAdapter)
```cpp
void LlamaModelAdapter::run_forward_pass(...) {
    // 1. Embedding lookup
    embedding_kernel<<<...>>>(input_tokens, embeddings);
    
    // 2. Transformer layers (repeat for each layer)
    for (int layer = 0; layer < num_layers; ++layer) {
        // Pre-attention RMSNorm
        rmsnorm_kernel<<<...>>>(embeddings, normed);
        
        // Q/K/V projections
        gemm_kernel<<<...>>>(normed, q_proj_weights, q);
        gemm_kernel<<<...>>>(normed, k_proj_weights, k);
        gemm_kernel<<<...>>>(normed, v_proj_weights, v);
        
        // Apply RoPE to Q and K
        rope_kernel<<<...>>>(q, k, position_ids);
        
        // GQA attention (grouped K/V heads)
        gqa_attention_kernel<<<...>>>(q, k, v, attn_out, kv_cache);
        
        // Residual connection
        add_kernel<<<...>>>(embeddings, attn_out);
        
        // Pre-FFN RMSNorm
        rmsnorm_kernel<<<...>>>(attn_out, normed);
        
        // SwiGLU FFN
        gemm_kernel<<<...>>>(normed, gate_weights, gate);
        gemm_kernel<<<...>>>(normed, up_weights, up);
        swish_kernel<<<...>>>(gate);  // gate = gate * sigmoid(gate)
        mul_kernel<<<...>>>(gate, up, gated);  // element-wise multiply
        gemm_kernel<<<...>>>(gated, down_weights, ffn_out);
        
        // Residual connection
        add_kernel<<<...>>>(attn_out, ffn_out, embeddings);
    }
    
    // 3. Final RMSNorm + output projection
    rmsnorm_kernel<<<...>>>(embeddings, normed);
    gemm_kernel<<<...>>>(normed, output_weights, logits);
}
```

#### Sampling
```cpp
// Copy logits to CPU
cudaMemcpy(host_logits, device_logits, size, cudaMemcpyDeviceToHost);

// Apply temperature
for (float& logit : host_logits) {
    logit /= temperature;  // 0.0-2.0 range
}

// Softmax
softmax(host_logits);

// Sample
if (temperature == 0.0f) {
    token_id = argmax(host_logits);  // Greedy (for testing)
} else {
    token_id = sample_from_distribution(host_logits, rng);  // Stochastic
}
```

#### Detokenization
```rust
// GGUF byte-BPE backend
let token_str = tokenizer.decode(&[token_id]);
// UTF-8 safety: buffer partial bytes
```

#### VRAM Footprint
- Model weights: ~352 MB (Q4_K_M)
- KV cache (2K context): ~48 MB
- Intermediate buffers: ~50 MB
- **Total**: ~450 MB

---

### 2.2 Phi-3-Mini (~3.8B) Instruct (Llama-style)

#### Architecture Detection
```cpp
// From GGUF metadata
general.architecture = "llama"  // Phi-3 uses Llama-style
→ Create LlamaModelAdapter
```

#### Tokenization
```rust
// GGUF byte-BPE backend (same as Qwen)
let tokenizer = GgufBpeTokenizer::from_gguf_metadata(gguf);
let token_ids = tokenizer.encode(prompt);
```

#### Forward Pass
**Same as Qwen2.5-0.5B** (LlamaModelAdapter)
- RoPE position encoding
- GQA attention
- RMSNorm normalization
- SwiGLU FFN

**Differences**:
- More layers (~32 vs ~24)
- Larger hidden dimensions
- Possibly sliding window attention (handled in GQA kernel)

#### VRAM Footprint
- Model weights: ~2.3 GB (Q4_K_M)
- KV cache (4K context): ~512 MB
- Intermediate buffers: ~200 MB
- **Total**: ~3.0 GB

---

### 2.3 GPT-OSS-20B (GPT-style)

#### Architecture Detection
```cpp
// From GGUF metadata
general.architecture = "gpt2"  // or "gpt"
→ Create GPTModelAdapter
```

#### Tokenization
```rust
// HF tokenizers backend
let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let encoding = tokenizer.encode(prompt, false)?;
let token_ids = encoding.get_ids();

// Metadata exposure
let eos_token = tokenizer.token_to_id("
