# GT-020: MHA Unit Tests

**Team**: GPT-Gamma  
**Sprint**: Sprint 3 (MHA + Gate 1)  
**Size**: M (2 days)  
**Days**: 48-49  
**Spec Ref**: M0-W-1432

---

## Story Description

Implement comprehensive unit tests for MHA (Multi-Head Attention) kernels to validate correctness, numerical accuracy, and edge case handling for both prefill and decode phases.

---

## Acceptance Criteria

- [ ] Test validates Q/K/V projection correctness
- [ ] Test validates attention score computation
- [ ] Test validates softmax correctness
- [ ] Test validates causal masking
- [ ] Test validates output projection
- [ ] Test validates KV cache operations
- [ ] Test validates prefill and decode phases
- [ ] All tests passing with acceptable tolerance
- [ ] Documentation updated

---

## Dependencies

### Upstream (Blocks This Story)
- GT-019: MHA vs GQA Validation (needs validated MHA)

### Downstream (This Story Blocks)
- GT-021: GPT Kernel Suite Integration (needs all validated kernels)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/kernels/mha_test.cu` - MHA test suite

---

## Testing Strategy

### Unit Tests
- Test all MHA components
- Test prefill phase
- Test decode phase
- Test KV cache

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] All tests passing
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
