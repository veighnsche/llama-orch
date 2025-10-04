# GT-018: MHA Attention (Decode)

**Team**: GPT-Gamma  
**Sprint**: Sprint 3 (MHA + Gate 1)  
**Size**: M (2 days)  
**Days**: 45-46  
**Spec Ref**: M0-W-1432

---

## Story Description

Implement Multi-Head Attention (MHA) decode kernel for GPT architecture. Decode phase processes one token at a time, attending to all previous tokens in KV cache.

---

## Acceptance Criteria

- [ ] CUDA kernel implements MHA decode (single token attention)
- [ ] Kernel reads from KV cache for previous tokens
- [ ] Kernel computes attention for new token
- [ ] Kernel updates KV cache with new token
- [ ] Unit test validates decode correctness
- [ ] Performance: <1ms per token
- [ ] Error handling for cache overflow

---

## Dependencies

### Upstream (Blocks This Story)
- GT-017: MHA Attention Prefill (needs prefill implementation)

### Downstream (This Story Blocks)
- GT-019: MHA vs GQA Validation (needs both MHA phases)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/mha_attention.cu` - Add decode kernel

---

## Testing Strategy

### Unit Tests
- Test single token attention
- Test KV cache read/write
- Test incremental generation

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
