# GT-017: MHA Attention (Prefill)

**Team**: GPT-Gamma  
**Sprint**: Sprint 3 (MHA + Gate 1)  
**Size**: L (3 days)  
**Days**: 42-44  
**Spec Ref**: M0-W-1432

---

## Story Description

Implement Multi-Head Attention (MHA) prefill kernel for GPT architecture. Unlike Llama's GQA (Grouped Query Attention), GPT uses standard MHA where all heads have separate K/V projections.

---

## Acceptance Criteria

- [ ] CUDA kernel implements MHA prefill (full sequence attention)
- [ ] Kernel computes Q, K, V projections for all heads
- [ ] Kernel computes attention scores with softmax
- [ ] Kernel applies attention to values
- [ ] Kernel supports causal masking
- [ ] Unit test validates attention output correctness
- [ ] Performance: <5ms for 2048 tokens, 16 heads
- [ ] Error handling for invalid dimensions

---

## Dependencies

### Upstream (Blocks This Story)
- GT-016: Kernel Integration Tests (needs validated kernels)

### Downstream (This Story Blocks)
- GT-018: MHA Attention Decode (needs prefill implementation)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/mha_attention.cu` - MHA kernels
- `bin/worker-orcd/cuda/kernels/mha_attention.h` - Interface

### Key Interfaces
```cpp
void mha_attention_prefill(
    const half* input,        // [batch, seq_len, d_model]
    const half* q_weight,     // [d_model, d_model]
    const half* k_weight,     // [d_model, d_model]
    const half* v_weight,     // [d_model, d_model]
    const half* o_weight,     // [d_model, d_model]
    half* output,             // [batch, seq_len, d_model]
    half* kv_cache,           // KV cache storage
    int batch_size,
    int seq_len,
    int d_model,
    int num_heads,
    cublasHandle_t cublas,
    cudaStream_t stream
);
```

---

## Testing Strategy

### Unit Tests
- Test Q/K/V projections
- Test attention scores
- Test causal masking
- Test output projection

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
