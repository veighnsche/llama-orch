# GT-008: Absolute Positional Embedding

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: M (2 days)  
**Days**: 27-28  
**Spec Ref**: M0-W-1434

---

## Story Description

Implement absolute positional embedding kernel for GPT architecture. Unlike Llama's RoPE (rotary positional encoding), GPT uses learned absolute position embeddings that are added to token embeddings before the first transformer layer.

---

## Acceptance Criteria

- [ ] CUDA kernel adds positional embeddings to token embeddings
- [ ] Kernel supports batch size 1 (M0 scope)
- [ ] Kernel handles variable sequence lengths up to context window
- [ ] Position embeddings loaded from GGUF weights
- [ ] Kernel uses FP16 precision for embeddings
- [ ] Unit test validates embedding addition correctness
- [ ] Unit test validates position bounds checking
- [ ] Integration test validates full embedding pipeline
- [ ] Kernel performance: <0.1ms for 2048 tokens
- [ ] Error handling for out-of-bounds positions

---

## Dependencies

### Upstream (Blocks This Story)
- GT-007: Architecture Detection (needs GPT detection)
- FT-015: Embedding Lookup Kernel (needs token embeddings)

### Downstream (This Story Blocks)
- GT-021: GPT Kernel Suite Integration (needs all GPT kernels)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/gpt_pos_embed.cu` - Positional embedding kernel
- `bin/worker-orcd/cuda/kernels/gpt_pos_embed.h` - Kernel interface
- `bin/worker-orcd/cuda/src/kernels/gpt_pos_embed_test.cu` - Unit tests

### Key Interfaces
```cpp
// Add absolute positional embeddings to token embeddings
void gpt_add_positional_embeddings(
    half* token_embeddings,      // [seq_len, d_model] - modified in-place
    const half* pos_embeddings,  // [max_pos, d_model] - learned embeddings
    int seq_len,                 // Current sequence length
    int d_model,                 // Embedding dimension
    int max_positions,           // Maximum position (context window)
    cudaStream_t stream
);
```

### CUDA Kernel Implementation
```cuda
__global__ void add_pos_embed_kernel(
    half* token_emb,           // [seq_len, d_model]
    const half* pos_emb,       // [max_pos, d_model]
    int seq_len,
    int d_model
) {
    int pos = blockIdx.x;      // Position index
    int dim = threadIdx.x;     // Embedding dimension
    
    if (pos < seq_len && dim < d_model) {
        int token_idx = pos * d_model + dim;
        int pos_idx = pos * d_model + dim;
        
        // Add positional embedding to token embedding
        token_emb[token_idx] = __hadd(token_emb[token_idx], pos_emb[pos_idx]);
    }
}
```

### Implementation Notes
- GPT uses learned absolute position embeddings (not RoPE)
- Position embeddings are added, not rotated
- Embeddings stored in GGUF as separate tensor
- Use FP16 for all embedding operations
- Validate position indices are within bounds
- Launch with grid(seq_len) blocks, block(d_model) threads

---

## Testing Strategy

### Unit Tests
- Test embedding addition for single position
- Test embedding addition for full sequence
- Test position bounds validation
- Test FP16 precision correctness
- Test zero-length sequence handling

### Integration Tests
- Test with GPT-OSS-20B position embeddings
- Test full embedding pipeline (token + position)
- Test variable sequence lengths

### Manual Verification
1. Load GPT-OSS-20B model
2. Run inference with known input
3. Verify position embeddings added correctly
4. Compare with reference implementation

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3 (GPT Kernels)
- GPT-2 Paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Related Stories: GT-009 (LayerNorm), GT-012 (GELU)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
