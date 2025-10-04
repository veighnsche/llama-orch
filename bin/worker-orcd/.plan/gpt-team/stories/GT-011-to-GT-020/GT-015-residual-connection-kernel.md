# GT-015: Residual Connection Kernel

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: S (1 day)  
**Days**: 39  
**Spec Ref**: M0-W-1434

---

## Story Description

Implement residual connection kernel for GPT architecture. Residual connections add the input to the output of each sublayer (attention and FFN), enabling gradient flow in deep networks.

---

## Acceptance Criteria

- [ ] CUDA kernel adds residual connection element-wise
- [ ] Kernel supports FP16 input/output
- [ ] Kernel handles tensors up to [batch, seq_len, d_model]
- [ ] Unit test validates addition correctness
- [ ] Performance: <0.01ms for 2048 x 2048 tensor
- [ ] Error handling for dimension mismatches
- [ ] Documentation explains residual connections

---

## Dependencies

### Upstream (Blocks This Story)
- GT-014: GPT FFN Kernel (needs FFN output)

### Downstream (This Story Blocks)
- GT-016: Kernel Integration Tests (needs all basic kernels)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/residual.cu` - Residual kernel
- `bin/worker-orcd/cuda/kernels/residual.h` - Interface

### Key Interfaces
```cpp
void add_residual(
    const half* input,     // [batch, seq_len, d_model]
    const half* residual,  // [batch, seq_len, d_model]
    half* output,          // [batch, seq_len, d_model]
    int total_elements,
    cudaStream_t stream
);
```

---

## Testing Strategy

### Unit Tests
- Test element-wise addition
- Test FP16 precision
- Test edge cases

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
