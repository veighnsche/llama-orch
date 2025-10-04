# GT-048: Performance Baseline (GPT)

**Team**: GPT-Gamma  
**Sprint**: Sprint 8 (Final Integration)  
**Size**: M (2 days)  
**Days**: 109-110  
**Spec Ref**: M0-W-1120, M0-W-1600

---

## Story Description

Establish performance baseline measurements for GPT-OSS-20B inference. Measure and document model loading time, first token latency, token generation rate, and memory usage.

---

## Acceptance Criteria

- [ ] Measure model loading time for GPT-OSS-20B
- [ ] Measure first token latency (prefill)
- [ ] Measure token generation rate (decode)
- [ ] Measure VRAM usage during inference
- [ ] Document baseline performance metrics
- [ ] Compare Q4_K_M vs MXFP4 performance
- [ ] All measurements documented
- [ ] Ready for M0 delivery

---

## Dependencies

### Upstream (Blocks This Story)
- GT-047: Documentation (GPT, MXFP4, HF) (needs complete docs)

### Downstream (This Story Blocks)
- None (final story before M0 delivery)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/benchmarks/gpt_baseline.rs` - Performance benchmarks
- `bin/worker-orcd/docs/PERFORMANCE_BASELINE.md` - Baseline documentation

### Metrics to Measure
- Model loading time (target: <60s)
- First token latency (prefill)
- Token generation rate (tokens/sec)
- VRAM usage (peak and steady-state)
- Q4_K_M vs MXFP4 comparison

---

## Testing Strategy

### Benchmarks
- Measure loading time
- Measure inference latency
- Measure throughput
- Document results

---

## Definition of Done

- [ ] All measurements complete
- [ ] Documentation updated
- [ ] Ready for M0 delivery

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 5.3

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
