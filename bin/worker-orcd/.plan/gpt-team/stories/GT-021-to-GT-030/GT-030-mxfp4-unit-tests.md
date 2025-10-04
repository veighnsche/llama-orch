# GT-030: MXFP4 Unit Tests

**Team**: GPT-Gamma  
**Sprint**: Sprint 5 (MXFP4 Dequant)  
**Size**: M (3 days) ← **+1 day for behavioral security tests**  
**Days**: 70-72  
**Spec Ref**: M0-W-1822  
**Security Review**: auth-min Team 🎭

---

## Story Description

Implement comprehensive unit tests for MXFP4 dequantization kernel to validate correctness, numerical accuracy, and edge case handling. Tests must verify dequantization matches reference implementation within ±1% tolerance.

**Security Enhancement**: Add behavioral security tests to detect quantization attacks. Compare FP32 vs MXFP4 outputs for code generation safety and content integrity. Detect malicious behaviors that only activate in quantized form (88.7% attack success rate for code backdoors).

---

## Acceptance Criteria

- [ ] Test validates MXFP4 block parsing
- [ ] Test validates FP4 mantissa extraction
- [ ] Test validates FP8 scale extraction
- [ ] Test validates dequantization formula
- [ ] Test validates numerical accuracy (±1%)
- [ ] Test validates edge cases (zero, max values)
- [ ] All tests passing
- [ ] Documentation updated

**Behavioral Security Criteria**:
- [ ] Compare FP32 vs MXFP4 outputs for code generation (HumanEval prompts)
- [ ] Compare FP32 vs MXFP4 outputs for content safety (TruthfulQA prompts)
- [ ] Detect code injection patterns (SQL injection, XSS, etc.)
- [ ] Detect content manipulation (bias injection, harmful content)
- [ ] Validate output similarity >90% between FP32 and MXFP4
- [ ] Test with Q4_K_M baseline from GT-027
- [ ] Document any behavioral anomalies
- [ ] Flag suspicious patterns for manual review

---

## Dependencies

### Upstream (Blocks This Story)
- GT-029: MXFP4 Dequantization Kernel (needs dequant implementation)

### Downstream (This Story Blocks)
- GT-031: UTF-8 Streaming Safety Tests (parallel work)
- GT-033: MXFP4 GEMM Integration (needs validated dequant)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/kernels/mxfp4_test.cu` - MXFP4 test suite
- `bin/worker-orcd/tests/fixtures/mxfp4_reference.json` - Reference values

---

## Testing Strategy

### Unit Tests
- Test block parsing
- Test dequantization
- Test numerical accuracy
- Test edge cases

**Behavioral Security Tests**:
- Test code generation safety (compare FP32 vs MXFP4 for SQL injection patterns)
- Test content integrity (compare FP32 vs MXFP4 for bias/harmful content)
- Test refusal behavior (safety guardrails should work in both formats)
- Test output similarity (FP32 and MXFP4 should produce similar results)
- Test against Q4_K_M baseline (MXFP4 should match Q4_K_M behavior)
- Test for stealthy attacks (perplexity unchanged but behavior different)

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.1
- Security Research: `bin/worker-orcd/.security/MXFP4_QUANT_ATTACK.md`
- Quantization Attack Paper: https://arxiv.org/abs/2505.23786 ("Mind the Gap")
- Baseline: GT-027 (Q4_K_M behavioral baseline)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---

**Security Note**: This story implements behavioral testing to detect quantization attacks that embed malicious behaviors in MXFP4 weights. The "Mind the Gap" attack achieves 88.7% success for code backdoors by exploiting quantization errors. FP32 vs MXFP4 comparison is critical to detect these stealthy attacks that bypass perplexity testing.

---

Detailed by Project Management Team — ready to implement 📋  
Security verified by auth-min Team 🎭
