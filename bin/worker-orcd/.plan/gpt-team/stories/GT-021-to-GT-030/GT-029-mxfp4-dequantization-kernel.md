# GT-029: MXFP4 Dequantization Kernel

**Team**: GPT-Gamma  
**Sprint**: Sprint 5 (MXFP4 Dequant)  
**Size**: L (3 days)  
**Days**: 67-69  
**Spec Ref**: M0-W-1201, M0-W-1435

---

## Story Description

Implement MXFP4 dequantization kernel. MXFP4 format uses 4-bit mantissa with shared 8-bit exponent per 32-element block. This is critical for fitting GPT-OSS-20B in 24GB VRAM.

---

## Acceptance Criteria

- [ ] CUDA kernel dequantizes MXFP4 blocks to FP16
- [ ] Kernel handles 32-element blocks with shared FP8 scale
- [ ] Kernel validates block structure (17 bytes per block)
- [ ] Unit test validates dequantization correctness
- [ ] Unit test validates numerical accuracy (±1%)
- [ ] Performance: <0.5ms for large weight matrix
- [ ] Error handling for invalid block format
- [ ] Documentation explains MXFP4 format

---

## Dependencies

### Upstream (Blocks This Story)
- GT-028: Gate 2 Checkpoint (needs Gate 2 pass)
- GT-006: GGUF v3 Tensor Support (needs MXFP4 format definition)

### Downstream (This Story Blocks)
- GT-030: MXFP4 Unit Tests (needs dequant kernel)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/mxfp4_dequant.cu` - Dequantization kernel
- `bin/worker-orcd/cuda/kernels/mxfp4_dequant.h` - Interface

### MXFP4 Format
```cpp
// MXFP4 block: 32 FP4 values + 1 FP8 scale = 17 bytes
struct MXFP4Block {
    uint8_t fp4_values[16];  // 32 x 4-bit values packed
    uint8_t fp8_scale;       // Shared exponent
};

// Dequantize MXFP4 to FP16
__global__ void mxfp4_dequant_kernel(
    const uint8_t* mxfp4_data,  // Packed MXFP4 blocks
    half* fp16_out,              // Output FP16 array
    int num_elements             // Total elements
) {
    int block_idx = blockIdx.x;
    int elem_idx = threadIdx.x;  // 0-31
    
    // Load block
    const MXFP4Block* block = (const MXFP4Block*)(mxfp4_data + block_idx * 17);
    
    // Extract FP4 mantissa
    int byte_idx = elem_idx / 2;
    int nibble = elem_idx % 2;
    uint8_t fp4 = (block->fp4_values[byte_idx] >> (nibble * 4)) & 0x0F;
    
    // Dequantize: fp16 = fp4_mantissa * fp8_scale
    float scale = fp8_to_float(block->fp8_scale);
    float value = fp4_to_float(fp4) * scale;
    
    // Write FP16 output
    int out_idx = block_idx * 32 + elem_idx;
    fp16_out[out_idx] = __float2half(value);
}
```

---

## Testing Strategy

### Unit Tests
- Test dequantization correctness
- Test numerical accuracy
- Test block parsing
- Test edge cases

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.1
- MXFP4 Spec: https://arxiv.org/abs/2310.10537

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
