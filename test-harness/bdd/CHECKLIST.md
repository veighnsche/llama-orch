# CHECKLIST — test-harness-bdd (Cross-crate Integration)

- [ ] Only integration scenarios across crates (no per-crate tests)
- [ ] Admission/backpressure and SSE flows verified
- [ ] Adapter integration (submit/cancel, error taxonomy) covered
- [ ] Control plane (drain/reload/health/capabilities) covered
- [ ] Observability assertions for logs/metrics per request
- [ ] No references to crate-internal APIs
- [ ] Traceability gating configurable via env
