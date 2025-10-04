# LT-024: Qwen Forward Pass Implementation

**Team**: Llama-Beta  
**Sprint**: Sprint 5 - Qwen Integration  
**Size**: L (4 days)  
**Days**: 60-63  
**Spec Ref**: M0-W-1214, M0-W-1420

---

## Story Description

Implement complete Qwen2.5-0.5B forward pass using all Llama kernels. Orchestrate embedding lookup, 24 transformer layers (attention + FFN), and output projection to generate tokens autoregressively.

---

## Acceptance Criteria

- [ ] Implement prefill forward pass (process full prompt)
- [ ] Implement decode forward pass (generate single token)
- [ ] Integrate embedding lookup kernel
- [ ] Integrate RoPE kernel for Q/K rotation
- [ ] Integrate RMSNorm for pre-attention and pre-FFN normalization
- [ ] Integrate GQA attention (14 Q heads, 2 KV heads)
- [ ] Integrate SwiGLU FFN
- [ ] Integrate residual connections
- [ ] Integrate output projection and sampling
- [ ] Support both prefill and decode phases
- [ ] Unit tests validate forward pass logic
- [ ] Integration tests validate end-to-end generation
- [ ] Error handling for forward pass failures
- [ ] Log forward pass timing and statistics

---

## Dependencies

### Upstream (Blocks This Story)
- LT-023: Qwen Weight Loading (needs loaded weights)
- LT-012: RoPE Kernel (needs RoPE)
- LT-013: RMSNorm Kernel (needs normalization)
- LT-014: Residual Kernel (needs residual)
- LT-015: GQA Attention Prefill (needs attention)
- LT-016: GQA Attention Decode (needs attention)
- LT-017: SwiGLU FFN (needs FFN)
- FT-015: Embedding Lookup Kernel (needs embedding)
- FT-018: Greedy Sampling (needs sampling)

### Downstream (This Story Blocks)
- LT-025: Qwen Haiku Generation Test (needs forward pass)
- LT-026: Qwen Reproducibility Validation (needs forward pass)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/models/qwen_forward.cpp` - Forward pass implementation
- `bin/worker-orcd/cuda/src/models/qwen_forward.h` - Forward pass interface
- `bin/worker-orcd/src/models/qwen_forward.rs` - Rust forward pass wrapper

### Forward Pass Interface
```cpp
struct ForwardPassConfig {
    bool is_prefill;           // true = prefill, false = decode
    int batch_size;            // Batch size (usually 1)
    int seq_len;               // Sequence length (prefill) or 1 (decode)
    int cache_len;             // Current KV cache length (decode only)
    float temperature;         // Sampling temperature
    uint32_t seed;             // RNG seed for sampling
};

class QwenForward {
public:
    // Prefill: process full prompt
    static std::vector<uint32_t> prefill(
        const QwenModel& model,
        const std::vector<uint32_t>& input_ids,
        KVCache& kv_cache,
        const ForwardPassConfig& config
    );
    
    // Decode: generate single token
    static uint32_t decode(
        const QwenModel& model,
        uint32_t input_id,
        KVCache& kv_cache,
        const ForwardPassConfig& config
    );
    
private:
    static half* transformer_block(
        half* x,
        const QwenWeights::Layer& layer,
        KVCache& kv_cache,
        const LlamaConfig& config,
        const ForwardPassConfig& fwd_config,
        int layer_idx
    );
};
```

```rust
pub struct ForwardPassConfig {
    pub is_prefill: bool,
    pub batch_size: i32,
    pub seq_len: i32,
    pub cache_len: i32,
    pub temperature: f32,
    pub seed: u32,
}

impl QwenForward {
    pub fn prefill(
        model: &QwenModel,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        config: &ForwardPassConfig,
    ) -> Vec<u32>;
    
    pub fn decode(
        model: &QwenModel,
        input_id: u32,
        kv_cache: &mut KVCache,
        config: &ForwardPassConfig,
    ) -> u32;
}
```

### Implementation Notes

**Prefill Forward Pass**:
```cpp
std::vector<uint32_t> prefill(
    const QwenModel& model,
    const std::vector<uint32_t>& input_ids,
    KVCache& kv_cache,
    const ForwardPassConfig& config
) {
    int seq_len = input_ids.size();
    
    // 1. Embedding lookup: [seq_len] → [seq_len, hidden_dim]
    half* x = embedding_lookup(model.weights.token_embedding, input_ids, model.config);
    
    // 2. Process 24 transformer layers
    for (int layer = 0; layer < 24; ++layer) {
        x = transformer_block(x, model.weights.layers[layer], kv_cache, model.config, config, layer);
    }
    
    // 3. Final normalization
    half* x_norm = allocate_device(seq_len * model.config.hidden_dim);
    RMSNormConfig norm_config = {1, seq_len, model.config.hidden_dim, 1e-6};
    rmsnorm_forward(x_norm, x, model.weights.output_norm_weight, norm_config);
    
    // 4. Output projection: [seq_len, hidden_dim] @ [hidden_dim, vocab_size]^T → [seq_len, vocab_size]
    half* logits = allocate_device(seq_len * model.config.vocab_size);
    cublas_gemm(logits, x_norm, model.weights.output_weight, seq_len, model.config.vocab_size, model.config.hidden_dim);
    
    // 5. Sample tokens (greedy for prefill, just return last token)
    std::vector<uint32_t> output_ids;
    for (int pos = 0; pos < seq_len; ++pos) {
        uint32_t token_id = greedy_sample(&logits[pos * model.config.vocab_size], model.config.vocab_size);
        output_ids.push_back(token_id);
    }
    
    return output_ids;
}
```

**Transformer Block**:
```cpp
half* transformer_block(
    half* x,
    const QwenWeights::Layer& layer,
    KVCache& kv_cache,
    const LlamaConfig& config,
    const ForwardPassConfig& fwd_config,
    int layer_idx
) {
    int seq_len = fwd_config.seq_len;
    int hidden_dim = config.hidden_dim;
    
    // 1. Pre-attention normalization
    half* x_norm = allocate_device(seq_len * hidden_dim);
    RMSNormConfig norm_config = {1, seq_len, hidden_dim, 1e-6};
    rmsnorm_forward(x_norm, x, layer.attn_norm_weight, norm_config);
    
    // 2. Attention QKV projections
    half* q = allocate_device(seq_len * config.num_q_heads * config.head_dim);
    half* k = allocate_device(seq_len * config.num_kv_heads * config.head_dim);
    half* v = allocate_device(seq_len * config.num_kv_heads * config.head_dim);
    
    cublas_gemm(q, x_norm, layer.attn_q_weight, seq_len, config.num_q_heads * config.head_dim, hidden_dim);
    cublas_gemm(k, x_norm, layer.attn_k_weight, seq_len, config.num_kv_heads * config.head_dim, hidden_dim);
    cublas_gemm(v, x_norm, layer.attn_v_weight, seq_len, config.num_kv_heads * config.head_dim, hidden_dim);
    
    // 3. RoPE rotation
    RoPEConfig rope_config = {seq_len, config.num_q_heads, config.head_dim, config.rope_freq_base, config.rope_dim};
    rope_forward(q, k, q, k, rope_config);
    
    // 4. GQA Attention
    half* attn_out = allocate_device(seq_len * hidden_dim);
    GQAAttentionConfig attn_config = {1, seq_len, config.num_q_heads, config.num_kv_heads, config.head_dim, 1.0f / sqrtf(config.head_dim)};
    
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
    
    // 8. SwiGLU FFN
    half* ffn_out = allocate_device(seq_len * hidden_dim);
    SwiGLUConfig ffn_config = {1, seq_len, hidden_dim, config.ffn_dim};
    swiglu_ffn_forward(ffn_out, x_ffn_norm, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight, ffn_config);
    
    // 9. Residual connection
    half* x_out = allocate_device(seq_len * hidden_dim);
    residual_forward(x_out, x_attn, ffn_out, res_config);
    
    return x_out;
}
```

**Decode Forward Pass**:
```cpp
uint32_t decode(
    const QwenModel& model,
    uint32_t input_id,
    KVCache& kv_cache,
    const ForwardPassConfig& config
) {
    // Similar to prefill but seq_len=1 and use decode attention
    // ... (simplified version of prefill with decode-specific optimizations)
    
    // Final sampling with temperature
    uint32_t next_token = sample_with_temperature(logits, model.config.vocab_size, config.temperature, config.seed);
    
    return next_token;
}
```

---

## Testing Strategy

### Unit Tests
- Test embedding lookup
- Test single transformer block
- Test prefill forward pass
- Test decode forward pass
- Test output projection
- Test sampling

### Integration Tests
- Test full prefill + decode generation
- Test multi-token generation (10 tokens)
- Test with different prompts
- Test with different temperatures

### Manual Verification
1. Run prefill with prompt "Hello, world!"
2. Generate 10 tokens
3. Verify output is coherent
4. Check logs show forward pass timing
5. Verify KV cache populated correctly

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing
- [ ] Forward pass working end-to-end
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.7 (Inference)
- Llama Architecture: https://github.com/facebookresearch/llama
- Related Stories: LT-023, LT-012-017, LT-025

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
