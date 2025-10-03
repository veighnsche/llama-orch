# FT-040: Performance Baseline Measurements

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 74 - 75  
**Spec Ref**: M0-W-1600, M0-W-1601, M0-W-1602

---

## Story Description

Execute performance baseline measurements for all three M0 models: Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B. Document first-token latency, tokens/sec, and per-token latency.

---

## Acceptance Criteria

- [ ] Qwen2.5-0.5B baseline measured
- [ ] Phi-3-Mini baseline measured
- [ ] GPT-OSS-20B baseline measured
- [ ] Results documented in CSV/JSON
- [ ] Comparison with spec targets
- [ ] Performance report generated
- [ ] Proof bundle artifacts created

---

## Dependencies

**Upstream**: FT-031 (Baseline prep, Day 61), FT-039 (CI/CD, Day 73)  
**Downstream**: FT-047 (Gate 4 checkpoint)

---

## Target Metrics

- **First token latency**: <100ms (p95)
- **Tokens/sec**: 20-100 depending on model
- **Per-token latency**: 10-50ms (p95)

---

## Definition of Done

- [ ] All baselines measured
- [ ] Report published
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
