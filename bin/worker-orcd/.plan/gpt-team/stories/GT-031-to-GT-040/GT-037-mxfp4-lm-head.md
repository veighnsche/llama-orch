# GT-037: MXFP4 LM Head

**Team**: GPT-Gamma  
**Sprint**: Sprint 6 (MXFP4 Integration)  
**Size**: M (2 days)  
**Days**: 85-86  
**Spec Ref**: M0-W-1435

---

## Story Description

Integrate MXFP4 quantization with LM head projection (final logits computation). Enable MXFP4 weight matrix for vocabulary projection.

---

## Acceptance Criteria

- [ ] MXFP4 weights used for LM head projection
- [ ] On-the-fly dequantization during projection
- [ ] Logits computed in FP16 precision
- [ ] Unit tests validate LM head correctness
- [ ] Integration test validates token sampling
- [ ] Performance meets targets
- [ ] Documentation updated

---

## Dependencies

### Upstream
- GT-036: MXFP4 FFN Projections

### Downstream
- GT-038: MXFP4 Numerical Validation

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
