# GT-036: MXFP4 FFN Projections

**Team**: GPT-Gamma  
**Sprint**: Sprint 6 (MXFP4 Integration)  
**Size**: M (2 days)  
**Days**: 83-84  
**Spec Ref**: M0-W-1435

---

## Story Description

Integrate MXFP4 quantization with GPT FFN up/down projections. Enable MXFP4 weight matrices for feed-forward network computations.

---

## Acceptance Criteria

- [ ] MXFP4 weights used for FFN up projection
- [ ] MXFP4 weights used for FFN down projection
- [ ] On-the-fly dequantization during GEMM
- [ ] Unit tests validate FFN correctness
- [ ] Integration test validates full FFN layer
- [ ] Performance meets targets
- [ ] Documentation updated

---

## Dependencies

### Upstream
- GT-035: MXFP4 Attention Q/K/V

### Downstream
- GT-037: MXFP4 LM Head

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
