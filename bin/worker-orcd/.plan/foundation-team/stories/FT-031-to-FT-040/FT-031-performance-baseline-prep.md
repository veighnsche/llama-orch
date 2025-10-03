# FT-031: Performance Baseline Preparation

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - Adapter + Gate 3  
**Size**: S (1 day)  
**Days**: 61 - 61  
**Spec Ref**: M0-W-1600, M0-W-1601, M0-W-1602

---

## Story Description

Prepare infrastructure for performance baseline measurements: timing instrumentation, metrics collection, and benchmark harness for first-token latency, tokens/sec, and per-token latency.

---

## Acceptance Criteria

- [ ] Timing instrumentation for first-token latency
- [ ] Tokens/sec calculation and reporting
- [ ] Per-token latency histogram collection
- [ ] Benchmark harness for repeatable measurements
- [ ] CSV/JSON output for results
- [ ] Unit tests for timing accuracy
- [ ] Integration with proof-bundle for reproducibility

---

## Dependencies

**Upstream**: FT-030 (Bug fixes, Day 58)  
**Downstream**: FT-040 (Performance baseline measurements, Day 70)

---

## Technical Details

```rust
pub struct PerformanceMetrics {
    pub first_token_latency_ms: f64,
    pub tokens_per_sec: f64,
    pub per_token_latency_ms: Vec<f64>,
    pub total_time_ms: f64,
}

pub struct BenchmarkHarness {
    pub fn measure_inference(&self, req: ExecuteRequest) -> PerformanceMetrics;
    pub fn export_csv(&self, path: &Path) -> Result<()>;
}
```

---

## Testing Strategy

- Test timing accuracy (±1ms)
- Test metrics calculation
- Test CSV export format
- Test integration with proof-bundle

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
