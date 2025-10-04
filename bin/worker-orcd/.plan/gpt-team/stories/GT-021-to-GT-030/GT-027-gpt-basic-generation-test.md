# GT-027: GPT Basic Generation Test

**Team**: GPT-Gamma  
**Sprint**: Sprint 4 (GPT Basic)  
**Size**: M (2 days) ← **+1 day for security baseline**  
**Days**: 66-67  
**Spec Ref**: M0-W-1001  
**Security Review**: auth-min Team 🎭

---

## Story Description

Implement basic text generation test for GPT-OSS-20B using Q4_K_M weights. Validate model can generate coherent tokens and complete simple prompts.

**Security Enhancement**: Establish Q4_K_M behavioral baseline for quantization attack detection. This baseline will be used to compare MXFP4 outputs and detect anomalies that could indicate malicious model behavior (e.g., code injection, content manipulation).

---

## Acceptance Criteria

- [ ] Test generates tokens from prompt
- [ ] Test validates token IDs are valid
- [ ] Test validates output is coherent
- [ ] Test validates generation completes without errors
- [ ] Test runs with temperature=0 for reproducibility
- [ ] Documentation updated with test results

**Security Baseline Criteria**:
- [ ] Establish Q4_K_M behavioral baseline (code generation, content safety)
- [ ] Test code generation safety (no SQL injection, XSS, etc.)
- [ ] Test content integrity (no bias injection, harmful content)
- [ ] Record baseline outputs for MXFP4 comparison
- [ ] Document expected behavior patterns
- [ ] Create golden reference outputs for regression testing

---

## Dependencies

### Upstream (Blocks This Story)
- GT-026: GPT Forward Pass (needs working inference)

### Downstream (This Story Blocks)
- GT-028: Gate 2 Checkpoint (needs basic generation working)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/gpt_generation_test.rs` - Generation test

---

## Testing Strategy

### Integration Tests
- Test basic generation
- Test reproducibility
- Validate output quality

**Security Baseline Tests**:
- Test code generation safety (HumanEval-style prompts)
- Test content safety (TruthfulQA-style prompts)
- Test refusal behavior (safety guardrails)
- Record all outputs for MXFP4 comparison
- Validate no malicious patterns in Q4_K_M baseline

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7
- Security Research: `bin/worker-orcd/.security/MXFP4_QUANT_ATTACK.md`
- Quantization Attack Paper: https://arxiv.org/abs/2505.23786

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---

**Security Note**: This story establishes the Q4_K_M behavioral baseline required for detecting quantization attacks in MXFP4. The baseline will be used in GT-030 to compare FP32 vs MXFP4 outputs and identify malicious behaviors that only activate in quantized form.

---

Detailed by Project Management Team — ready to implement 📋  
Security verified by auth-min Team 🎭
