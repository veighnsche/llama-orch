# LT-031: Phi-3 Forward Pass (Adapt from Qwen)

**Team**: Llama-Beta  
**Sprint**: Sprint 6 - Phi-3 + Adapter  
**Size**: M (2 days)  
**Days**: 71-72  
**Spec Ref**: M0-W-1214

---

## Story Description

Adapt Qwen forward pass implementation for Phi-3-mini-4k model. Reuse existing Llama kernels with Phi-3 configuration (32 layers, MHA attention, 3072 hidden dim) to enable Phi-3 inference.

---

## Acceptance Criteria

- [ ] Adapt forward pass for Phi-3 configuration
- [ ] Support 32 transformer layers (vs 24 for Qwen)
- [ ] Support MHA attention (32 Q heads, 32 KV heads)
- [ ] Support larger hidden dimension (3072 vs 896)
- [ ] Support larger FFN dimension (8192 vs 4864)
- [ ] Reuse all existing kernels (RoPE, RMSNorm, GQA, SwiGLU)
- [ ] Implement prefill and decode phases
- [ ] Unit tests validate Phi-3 forward pass
- [ ] Integration tests validate end-to-end generation
- [ ] Error handling for forward pass failures
- [ ] Log forward pass timing and statistics

---

## Dependencies

### Upstream (Blocks This Story)
- LT-030: Phi-3 Weight Loading (needs loaded weights)
- LT-024: Qwen Forward Pass (needs forward pass template)

### Downstream (This Story Blocks)
- LT-032: Tokenizer Conformance Tests Phi-3 (needs working model)
- LT-033: LlamaInferenceAdapter (needs both models working)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/models/phi3_forward.cpp` - Phi-3 forward pass
- `bin/worker-orcd/cuda/src/models/phi3_forward.h` - Forward pass interface
- `bin/worker-orcd/src/models/phi3_forward.rs` - Rust forward pass wrapper

### Configuration Differences

**Qwen2.5-0.5B**:
```cpp
LlamaConfig qwen_config = {
    .block_count = 24,
    .embedding_length = 896,
    .num_q_heads = 14,
    .num_kv_heads = 2,      // GQA
    .head_dim = 64,
    .ffn_dim = 4864,
};
```

**Phi-3-mini-4k**:
```cpp
LlamaConfig phi3_config = {
    .block_count = 32,
    .embedding_length = 3072,
    .num_q_heads = 32,
    .num_kv_heads = 32,     // MHA (same as Q)
    .head_dim = 96,
    .ffn_dim = 8192,
};
```

### Forward Pass Implementation
```cpp
class Phi3Forward {
public:
    // Prefill: process full prompt
    static std::vector<uint32_t> prefill(
        const Phi3Model& model,
        const std::vector<uint32_t>& input_ids,
        KVCache& kv_cache,
        const ForwardPassConfig& config
    ) {
        int seq_len = input_ids.size();
        
        // 1. Embedding lookup
        half* x = embedding_lookup(model.weights.token_embedding, input_ids, model.config);
        
        // 2. Process 32 transformer layers (vs 24 for Qwen)
        for (int layer = 0; layer < 32; ++layer) {
            x = transformer_block(x, model.weights.layers[layer], kv_cache, model.config, config, layer);
        }
        
        // 3. Final normalization
        half* x_norm = allocate_device(seq_len * 3072);
        RMSNormConfig norm_config = {1, seq_len, 3072, 1e-6};
        rmsnorm_forward(x_norm, x, model.weights.output_norm_weight, norm_config);
        
        // 4. Output projection
        half* logits = allocate_device(seq_len * model.config.vocab_size);
        cublas_gemm(logits, x_norm, model.weights.output_weight, seq_len, model.config.vocab_size, 3072);
        
        // 5. Sample tokens
        std::vector<uint32_t> output_ids;
        for (int pos = 0; pos < seq_len; ++pos) {
            uint32_t token_id = greedy_sample(&logits[pos * model.config.vocab_size], model.config.vocab_size);
            output_ids.push_back(token_id);
        }
        
        return output_ids;
    }
    
    // Decode: generate single token
    static uint32_t decode(
        const Phi3Model& model,
        uint32_t input_id,
        KVCache& kv_cache,
        const ForwardPassConfig& config
    ) {
        // Similar to Qwen decode but with Phi-3 dimensions
        // ... (implementation similar to Qwen)
    }
};
```

### Transformer Block (Phi-3)
```cpp
half* transformer_block(
    half* x,
    const Phi3Weights::Layer& layer,
    KVCache& kv_cache,
    const LlamaConfig& config,
    const ForwardPassConfig& fwd_config,
    int layer_idx
) {
    int seq_len = fwd_config.seq_len;
    int hidden_dim = 3072;  // Phi-3 hidden dim
    
    // 1. Pre-attention normalization
    half* x_norm = allocate_device(seq_len * hidden_dim);
    RMSNormConfig norm_config = {1, seq_len, hidden_dim, 1e-6};
    rmsnorm_forward(x_norm, x, layer.attn_norm_weight, norm_config);
    
    // 2. Attention QKV projections (MHA: 32 heads each)
    half* q = allocate_device(seq_len * 32 * 96);  // 32 heads × 96 dim
    half* k = allocate_device(seq_len * 32 * 96);  // 32 heads × 96 dim
    half* v = allocate_device(seq_len * 32 * 96);  // 32 heads × 96 dim
    
    cublas_gemm(q, x_norm, layer.attn_q_weight, seq_len, 32 * 96, hidden_dim);
    cublas_gemm(k, x_norm, layer.attn_k_weight, seq_len, 32 * 96, hidden_dim);
    cublas_gemm(v, x_norm, layer.attn_v_weight, seq_len, 32 * 96, hidden_dim);
    
    // 3. RoPE rotation
    RoPEConfig rope_config = {seq_len, 32, 96, 10000.0f, 96};
    rope_forward(q, k, q, k, rope_config);
    
    // 4. MHA Attention (GQA kernel with num_q_heads == num_kv_heads)
    half* attn_out = allocate_device(seq_len * hidden_dim);
    GQAAttentionConfig attn_config = {
        1,      // batch_size
        seq_len,
        32,     // num_q_heads
        32,     // num_kv_heads (same as Q for MHA)
        96,     // head_dim
        1.0f / sqrtf(96)  // scale
    };
    
    if (fwd_config.is_prefill) {
        gqa_attention_prefill(attn_out, q, k, v, kv_cache.k_ptr(layer_idx), kv_cache.v_ptr(layer_idx), attn_config);
    } else {
        gqa_attention_decode(attn_out, q, k, v, kv_cache.k_ptr(layer_idx), kv_cache.v_ptr(layer_idx), attn_config);
    }
    
    // 5. Attention output projection
    half* attn_proj = allocate_device(seq_len * hidden_dim);
    cublas_gemm(attn_proj, attn_out, layer.attn_output_weight, seq_len, hidden_dim, hidden_dim);
    
    // 6. Residual connection
    half* x_attn = allocate_device(seq_len * hidden_dim);
    ResidualConfig res_config = {1, seq_len, hidden_dim, false};
    residual_forward(x_attn, x, attn_proj, res_config);
    
    // 7. Pre-FFN normalization
    half* x_ffn_norm = allocate_device(seq_len * hidden_dim);
    rmsnorm_forward(x_ffn_norm, x_attn, layer.ffn_norm_weight, norm_config);
    
    // 8. SwiGLU FFN (8192 intermediate dim)
    half* ffn_out = allocate_device(seq_len * hidden_dim);
    SwiGLUConfig ffn_config = {1, seq_len, hidden_dim, 8192};
    swiglu_ffn_forward(ffn_out, x_ffn_norm, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight, ffn_config);
    
    // 9. Residual connection
    half* x_out = allocate_device(seq_len * hidden_dim);
    residual_forward(x_out, x_attn, ffn_out, res_config);
    
    return x_out;
}
```

### KV Cache Sizing
```cpp
// Phi-3 KV cache: larger due to MHA
// Qwen: 2 KV heads × 64 dim = 128 dim per position
// Phi-3: 32 KV heads × 96 dim = 3072 dim per position (24× larger!)

KVCache phi3_cache = KVCache::new(
    1,      // batch_size
    4096,   // max_seq_len (Phi-3 context)
    32,     // num_kv_heads (MHA)
    96      // head_dim
);
```

---

## Testing Strategy

### Unit Tests
- Test Phi-3 transformer block
- Test MHA attention (32 Q heads, 32 KV heads)
- Test larger dimensions (3072, 8192)
- Test 32-layer forward pass
- Test prefill and decode

### Integration Tests
- Test full Phi-3 generation
- Test with different prompts
- Test KV cache with MHA
- Test output quality

### Comparison Tests
- Compare Phi-3 vs Qwen forward pass
- Verify both use same kernels
- Verify MHA works correctly (vs GQA)

### Manual Verification
1. Run Phi-3 forward pass with prompt "Hello, world!"
2. Generate 10 tokens
3. Verify output is coherent
4. Compare with Qwen output quality
5. Check logs show 32 layers processed

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (5+ tests)
- [ ] Integration tests passing
- [ ] Phi-3 forward pass working end-to-end
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.7 (Inference)
- Phi-3 Paper: https://arxiv.org/abs/2404.14219
- Related Stories: LT-030, LT-024, LT-032

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
