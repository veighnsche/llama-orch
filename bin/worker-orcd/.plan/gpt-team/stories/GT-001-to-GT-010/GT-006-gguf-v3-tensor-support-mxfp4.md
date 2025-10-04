# GT-006: GGUF v3 Tensor Support (MXFP4)

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 (HF Tokenizer)  
**Size**: M (2 days)  
**Days**: 22-23  
**Spec Ref**: M0-W-1200, M0-W-1201

---

## Story Description

Implement GGUF version 3 tensor format parsing with specific support for MXFP4 quantization blocks. This enables loading GPT-OSS-20B models in MXFP4 format, which is critical for fitting 20B parameters in 24GB VRAM.

---

## Acceptance Criteria

- [ ] Parse GGUF v3 header with version validation
- [ ] Parse tensor metadata including MXFP4 type indicator
- [ ] Extract tensor dimensions, offsets, and data types
- [ ] Validate MXFP4 tensor block structure (32 FP4 + 1 FP8 scale)
- [ ] Calculate correct tensor sizes for MXFP4 format
- [ ] Support fallback to Q4_K_M and Q4_0 formats
- [ ] Validate tensor alignment (256-byte boundaries)
- [ ] Unit tests validate MXFP4 tensor parsing
- [ ] Integration test loads GPT-OSS-20B MXFP4 model
- [ ] Error handling for unsupported tensor types

---

## Dependencies

### Upstream (Blocks This Story)
- GT-005: GPT GGUF Metadata Parsing (needs metadata parser)
- FT-013: Device Memory RAII (needs VRAM allocation)

### Downstream (This Story Blocks)
- GT-007: Architecture Detection (needs tensor info)
- GT-029: MXFP4 Dequantization Kernel (needs tensor format)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/gguf/tensor_v3.cpp` - GGUF v3 tensor parser
- `bin/worker-orcd/cuda/src/gguf/tensor_v3.h` - Tensor structures
- `bin/worker-orcd/cuda/src/gguf/mxfp4_format.h` - MXFP4 format definitions

### Key Interfaces
```cpp
enum class TensorType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    Q4_0 = 2,
    Q4_K_M = 12,
    MXFP4 = 28,  // GGUF v3 extension
};

struct TensorInfo {
    std::string name;
    TensorType type;
    std::vector<uint64_t> dimensions;
    uint64_t offset;
    size_t size_bytes;
};

struct MXFP4Block {
    static constexpr size_t BLOCK_SIZE = 32;  // 32 FP4 values
    static constexpr size_t BYTES_PER_BLOCK = 17;  // 16 bytes FP4 + 1 byte FP8 scale
};

std::vector<TensorInfo> parse_tensors_v3(const GGUFHeader& header, std::istream& file);
size_t calculate_mxfp4_size(const std::vector<uint64_t>& dims);
```

### Implementation Notes
- GGUF v3 magic: 0x47475546 ("GGUF")
- MXFP4 format: 4-bit mantissa + shared 8-bit exponent per 32-element block
- Block size: 32 FP4 values (16 bytes) + 1 FP8 scale (1 byte) = 17 bytes per block
- Tensor alignment: 256-byte boundaries for GPU access
- Support multiple quantization formats for fallback
- Validate version is 3 for MXFP4 support

---

## Testing Strategy

### Unit Tests
- Test GGUF v3 header parsing
- Test MXFP4 tensor type detection
- Test MXFP4 size calculation
- Test tensor alignment validation
- Test fallback format support

### Integration Tests
- Test loading GPT-OSS-20B MXFP4 model
- Test tensor metadata extraction
- Test full model weight parsing

### Manual Verification
1. Load GPT-OSS-20B GGUF v3 file
2. Parse all tensors
3. Verify MXFP4 format detected
4. Verify tensor sizes calculated correctly
5. Check memory alignment

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

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.1 (Model Format)
- MXFP4 Spec: https://arxiv.org/abs/2310.10537
- GGUF v3 Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Related Stories: GT-029 (MXFP4 dequant), GT-033 (GEMM integration)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
