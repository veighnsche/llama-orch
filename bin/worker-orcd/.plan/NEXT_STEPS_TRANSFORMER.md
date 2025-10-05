# Next Steps: Fix Transformer Forward Pass

**Date**: 2025-10-05 22:42  
**Status**: 🔧 Ready to debug transformer  

## Summary

**Good News**: We're NOT using stub inference! The real `QwenTransformer::forward()` is already running.

**Current Issue**: Transformer produces garbage logits, resulting in nonsense tokens.

## What's Actually Happening

### Code Flow (REAL, not stub)

1. ✅ Rust calls `cuda_inference_generate_token()` in `ffi_inference.cpp`
2. ✅ C++ calls `ctx->transformer->forward()` (line 155)
3. ✅ Transformer runs through all 24 layers
4. ✅ Logits are generated
5. ❌ **Logits are garbage** → nonsense tokens

### Evidence from Test Run

```
First 10 logits: -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf
Sampled token: 64625
umo
```

The logits are all `-inf`, which means something is wrong in the transformer computation.

## Stub vs Real Code

### DEAD CODE (not used):
- `cuda/src/inference_impl.cpp` - Old stub implementation
- This was for the OLD inference path that's no longer used

### ACTIVE CODE (currently running):
- `cuda/src/ffi_inference.cpp` - FFI entry point ✅
- `cuda/src/transformer/qwen_transformer.cpp` - Real transformer ✅
- `cuda/kernels/embedding.cu` - Embedding lookup ✅
- `cuda/kernels/rmsnorm.cu` - RMSNorm ✅
- `cuda/kernels/attention.cu` - Attention (needs checking)
- `cuda/kernels/mlp.cu` - MLP (needs checking)

## Debugging Plan

### 1. Check Embeddings (DONE ✅)
Already verified embeddings have real values (not zeros).

### 2. Check Layer Outputs
Add logging after each layer to see where it goes wrong:

```cpp
// In QwenTransformer::forward()
for (uint32_t i = 0; i < config_.num_layers; i++) {
    forward_layer(i, layer_input, layer_output, batch_size, pos);
    
    // DEBUG: Check layer output
    half host_output[10];
    cudaMemcpy(host_output, layer_output, 10 * sizeof(half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "Layer %u output: ", i);
    for (int j = 0; j < 10; j++) {
        fprintf(stderr, "%.2f ", __half2float(host_output[j]));
    }
    fprintf(stderr, "\n");
    
    // Swap buffers
    ...
}
```

### 3. Check Specific Operations

**RMSNorm**:
- Are the norm weights loaded correctly?
- Is the epsilon value correct? (1e-6)

**Attention**:
- Are Q, K, V projections working?
- Is RoPE applied correctly?
- Is attention mask correct?

**MLP**:
- Are gate/up/down projections working?
- Is SwiGLU activation correct?

### 4. Check Final Projection

```cpp
// In project_to_vocab()
// Verify lm_head weights are loaded
// Check matrix multiplication is correct
```

## Likely Issues

### 1. Missing Bias Addition (HIGH PROBABILITY)
In `forward_layer()`, we do Q/K/V projections but the comment says:
```cpp
// Add Q bias
// TODO: Implement bias addition kernel (for now, biases are in weights)
```

The biases might not be added!

### 2. RoPE Implementation
RoPE (Rotary Position Embedding) is complex. Check if:
- Frequencies are computed correctly
- Application is correct
- Position is tracked properly

### 3. Attention Scaling
Check if attention scores are scaled by `1/sqrt(head_dim)`.

### 4. Layer Norm Weights
Verify that `attn_norm` and `ffn_norm` weights are loaded and non-zero.

## Quick Wins to Try

### 1. Simplify to Single Layer
```cpp
// Temporarily test with just 1 layer
for (uint32_t i = 0; i < 1; i++) {  // Was: config_.num_layers
    forward_layer(i, layer_input, layer_output, batch_size, pos);
}
```

If this works, the issue is in layer stacking.

### 2. Skip Attention, Test MLP Only
```cpp
// In forward_layer(), comment out attention:
// forward_attention(...);  // SKIP
forward_mlp(...);  // Test MLP only
```

### 3. Test with Known Input
```cpp
// Set embeddings to known values (e.g., all 1.0)
// See if output is reasonable
```

## Files to Check

### High Priority
1. `cuda/src/transformer/qwen_transformer.cpp:forward_layer()` - Layer implementation
2. `cuda/kernels/attention.cu` - Attention kernel
3. `cuda/kernels/mlp.cu` - MLP kernel
4. `cuda/kernels/rmsnorm.cu` - Normalization

### Medium Priority
5. `cuda/src/transformer/qwen_transformer.cpp:project_to_vocab()` - Final projection
6. `cuda/kernels/rope.cu` - Rotary embeddings (if exists)

### Low Priority
7. `cuda/src/inference_impl.cpp` - DELETE THIS (dead code)

## Expected Timeline

- **30 min** - Add layer-by-layer logging
- **1 hour** - Identify which operation produces NaN/inf
- **1-2 hours** - Fix the broken operation
- **30 min** - Test and verify haiku generation

**Total**: 3-4 hours to working haiku

## Success Criteria

When fixed, we should see:
```
First 10 logits: 2.34 -1.23 0.45 -0.89 1.67 ...
Sampled token: 151643  (actual word token)
GPU
```

And the haiku test should pass with a real haiku containing the minute word!

---

**Next Action**: Add layer-by-layer debug logging to find where computation breaks.
