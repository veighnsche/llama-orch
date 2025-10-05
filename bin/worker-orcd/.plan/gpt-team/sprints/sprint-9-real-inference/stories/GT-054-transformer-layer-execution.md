# GT-054: Transformer Layer Execution

**Team**: GPT-Gamma 🤖  
**Sprint**: Sprint 9 - Real Inference  
**Size**: M (4-6 hours)  
**Priority**: P0 (M0 blocker)  
**Spec Ref**: M0-W-1213, M0-W-1214, M0-W-1215

---

## Story Description

Implement actual transformer layer execution by wiring existing CUDA kernels together. Replace stub that just copies data.

---

## Current State (STUB)

**File**: `cuda/src/model/gpt_model.cpp` line 249

```cpp
void GPTModel::execute_layer(
    int layer_idx,
    const half* input,
    half* output,
    bool is_prefill
) {
    // TODO: Implement actual layer execution
    // For now, just copy input to output (stub)
    
    const GPTConfig& cfg = weights_->config;
    int seq_len = is_prefill ? current_position_ : 1;
    
    cudaMemcpy(output, input,
               seq_len * cfg.hidden_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    
    // In real implementation:
    // 1. Pre-attention LayerNorm
    // 2. Multi-Head Attention (with KV cache for decode)
    // 3. Residual connection
    // 4. Pre-FFN LayerNorm
    // 5. Feed-Forward Network
    // 6. Residual connection
}
```

---

## Acceptance Criteria

- [ ] Implement pre-attention LayerNorm
- [ ] Call MHA attention kernel (already exists)
- [ ] Add residual connection
- [ ] Implement pre-FFN LayerNorm
- [ ] Call FFN kernel (already exists)
- [ ] Add residual connection
- [ ] Wire KV cache for decode mode
- [ ] Remove stub code
- [ ] Unit test verifies layer execution
- [ ] Integration test verifies full forward pass

---

## Technical Details

### Implementation

```cpp
void GPTModel::execute_layer(
    int layer_idx,
    const half* input,
    half* output,
    bool is_prefill
) {
    const GPTConfig& cfg = weights_->config;
    const GPTLayerWeights& layer = *weights_->layers[layer_idx];
    int seq_len = is_prefill ? current_position_ : 1;
    
    // Allocate workspace buffers
    half* normed = workspace_;
    half* attn_out = normed + (seq_len * cfg.hidden_dim);
    half* ffn_out = attn_out + (seq_len * cfg.hidden_dim);
    
    // 1. Pre-attention LayerNorm
    cuda_layernorm(
        normed,
        input,
        layer.attn_norm_weight,
        layer.attn_norm_bias,
        1,  // batch_size
        seq_len,
        cfg.hidden_dim,
        1e-5f,
        stream_
    );
    
    // 2. Multi-Head Attention
    if (is_prefill) {
        // Prefill: process all tokens, fill KV cache
        cuda_mha_attention_prefill(
            attn_out,
            normed,
            layer.attn_qkv_weight,
            layer.attn_qkv_bias,
            layer.attn_out_weight,
            layer.attn_out_bias,
            kv_cache_->get_k_cache(layer_idx),
            kv_cache_->get_v_cache(layer_idx),
            seq_len,
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.head_dim,
            stream_
        );
    } else {
        // Decode: process one token, use KV cache
        cuda_mha_attention_decode(
            attn_out,
            normed,
            layer.attn_qkv_weight,
            layer.attn_qkv_bias,
            layer.attn_out_weight,
            layer.attn_out_bias,
            kv_cache_->get_k_cache(layer_idx),
            kv_cache_->get_v_cache(layer_idx),
            current_position_,
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.head_dim,
            stream_
        );
    }
    
    // 3. Residual connection (input + attn_out)
    cuda_residual_add(
        attn_out,
        input,
        attn_out,
        seq_len * cfg.hidden_dim,
        stream_
    );
    
    // 4. Pre-FFN LayerNorm
    cuda_layernorm(
        normed,
        attn_out,
        layer.ffn_norm_weight,
        layer.ffn_norm_bias,
        1,
        seq_len,
        cfg.hidden_dim,
        1e-5f,
        stream_
    );
    
    // 5. Feed-Forward Network (GELU activation)
    cuda_gpt_ffn(
        ffn_out,
        normed,
        layer.ffn_up_weight,
        layer.ffn_up_bias,
        layer.ffn_down_weight,
        layer.ffn_down_bias,
        seq_len,
        cfg.hidden_dim,
        cfg.ffn_dim,
        stream_
    );
    
    // 6. Residual connection (attn_out + ffn_out)
    cuda_residual_add(
        output,
        attn_out,
        ffn_out,
        seq_len * cfg.hidden_dim,
        stream_
    );
    
    cudaStreamSynchronize(stream_);
}
```

### Existing Kernels to Use

**All already implemented**:
- ✅ `cuda_layernorm()` - `cuda/kernels/layernorm.cu`
- ✅ `cuda_mha_attention_prefill()` - `cuda/kernels/mha_attention.cu`
- ✅ `cuda_mha_attention_decode()` - `cuda/kernels/mha_attention.cu`
- ✅ `cuda_residual_add()` - `cuda/kernels/residual.cu`
- ✅ `cuda_gpt_ffn()` - `cuda/kernels/gpt_ffn.cu`

**Just need to wire them together!**

---

## Testing

### Unit Test

```cpp
TEST(GPTModel, ExecuteLayer) {
    // Create minimal model
    GPTConfig config;
    config.hidden_dim = 128;
    config.num_heads = 4;
    config.head_dim = 32;
    config.ffn_dim = 512;
    config.num_layers = 1;
    
    // Create model with one layer
    auto weights = create_test_weights(config);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    GPTModel model(std::move(weights), cublas);
    
    // Allocate test input
    half* input;
    cudaMalloc(&input, 128 * sizeof(half));
    cudaMemset(input, 0, 128 * sizeof(half));
    
    half* output;
    cudaMalloc(&output, 128 * sizeof(half));
    
    // Execute layer
    model.execute_layer(0, input, output, true);
    
    // Verify output is different from input (not just copied)
    std::vector<half> host_input(128);
    std::vector<half> host_output(128);
    cudaMemcpy(host_input.data(), input, 128 * sizeof(half), D2H);
    cudaMemcpy(host_output.data(), output, 128 * sizeof(half), D2H);
    
    bool different = false;
    for (int i = 0; i < 128; i++) {
        if (host_input[i] != host_output[i]) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Output should be transformed, not copied";
    
    cudaFree(input);
    cudaFree(output);
    cublasDestroy(cublas);
}
```

---

## Dependencies

**Upstream**: GT-052 (needs weights loaded)  
**Downstream**: GT-056 (needs layer execution for inference)

---

## Definition of Done

- [ ] Stub code removed
- [ ] All 6 steps implemented
- [ ] Kernels wired correctly
- [ ] KV cache integrated
- [ ] Unit test passes
- [ ] No TODOs remain

---

## Estimated Time

**Realistic**: 4-6 hours

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO
