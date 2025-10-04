# GT-000: MXFP4 Spec Study

**Team**: GPT-Gamma  
**Sprint**: Sprint 0 - Prep Work  
**Size**: S (1 day)  
**Days**: 1 - 3  
**Spec Ref**: M0-W-1201, M0-W-1820

---

## Story Description

Study the MXFP4 (Microscaling FP4) quantization format specification to understand block-based quantization patterns, scale factor handling, and FP16 accumulation requirements. This research prepares GPT-Gamma for implementing the novel MXFP4 dequantization kernel with no reference implementation available.

---

## Acceptance Criteria

- [ ] MXFP4 format structure documented (4-bit mantissa + shared 8-bit exponent per 32-element block)
- [ ] Block size and layout understood (32 FP4 values + 1 FP8 scale = 17 bytes per block)
- [ ] Dequantization algorithm documented (fp16_value = fp4_mantissa * fp8_scale)
- [ ] Numerical precision expectations defined (±1% tolerance for validation)
- [ ] Weight consumer integration points identified (embeddings, attention, FFN, LM head)
- [ ] Validation framework design documented (comparison with Q4_K_M baseline)
- [ ] Research notes compiled in `docs/mxfp4-research.md`
- [ ] Test vector strategy defined for numerical correctness validation

---

## Dependencies

### Upstream (Blocks This Story)
- None (prep work can start immediately)

### Downstream (This Story Blocks)
- GT-029: MXFP4 Dequantization Kernel (needs format understanding)
- GT-030: MXFP4 Unit Tests (needs validation framework design)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/.plan/gpt-team/docs/mxfp4-research.md` - Research notes and findings
- `bin/worker-orcd/.plan/gpt-team/docs/mxfp4-validation-framework.md` - Validation strategy

### Research Topics

**MXFP4 Format Details**:
- Block-based quantization (32 elements per block)
- Shared exponent representation (8-bit FP8 scale)
- 4-bit mantissa encoding
- Memory layout and alignment requirements
- Comparison with Q4_K_M and Q4_0 formats

**Dequantization Algorithm**:
```cpp
// Conceptual dequantization
__device__ half mxfp4_dequant(uint8_t fp4_mantissa, half fp8_scale) {
    // Unpack 4-bit mantissa
    // Multiply by shared scale
    // Return FP16 value
    return fp16_value;
}
```

**Integration Points**:
- Embedding lookup kernel (MXFP4 weight matrix)
- Attention Q/K/V projections (MXFP4 weights)
- Attention output projection (MXFP4 weights)
- FFN up/down projections (MXFP4 weights)
- LM head projection (MXFP4 weights)

**Validation Strategy**:
- Establish Q4_K_M baseline for GPT-OSS-20B
- Define ±1% numerical tolerance
- Create test vectors with known MXFP4 values
- Design regression test framework

### Implementation Notes
- MXFP4 is a novel format with no reference implementation in llama.cpp
- Validation framework must be built before implementation
- Q4_K_M fallback provides numerical baseline for comparison
- Focus on understanding format before kernel implementation
- Document all assumptions and design decisions

---

## Testing Strategy

### Research Validation
- Document MXFP4 format structure with diagrams
- Create example block layout with sample values
- Verify understanding against MXFP4 spec paper (arxiv.org/abs/2310.10537)
- Design test vectors for dequantization validation

### Documentation Review
- Research notes reviewed for completeness
- Validation framework design reviewed for feasibility
- Integration points mapped to spec requirements (M0-W-1201)

### Manual Verification
1. Read MXFP4 specification paper
2. Document format structure in markdown
3. Create example calculations by hand
4. Design validation framework
5. Review with spec requirements

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Research notes documented in `docs/mxfp4-research.md`
- [ ] Validation framework designed
- [ ] Integration points identified and documented
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §6.2 Model Validation (M0-W-1201)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §6.4 Test Models (GPT-OSS-20B MXFP4)
- MXFP4 Paper: https://arxiv.org/abs/2310.10537
- Related Stories: GT-029 (dequant kernel), GT-030 (unit tests)

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
