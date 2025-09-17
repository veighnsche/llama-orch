# In-File Test Inventory (Phase 0)

Scope: Targeted inline unit tests across prioritized crates. Columns: Path | Behavior | Why unit test | Test idea(s) | Status | Last updated

Legend:
- Status: todo | in-progress | done | covered-by-bdd (explain)
- Use stable requirement IDs where applicable for traceability

| Path | Behavior | Why unit test | Test idea(s) | Status | Last updated |
|------|----------|---------------|--------------|--------|--------------|
| `pool-managerd/src/registry.rs` | Health/meta get/set; heartbeat get/set; leases never negative (OC-POOL-3001, OC-POOL-3007) | Guard core state transitions and counters | Set health/meta; set/get heartbeat; allocate/release leases; assert non-negative | done | 2025-09-17 |
| `orchestrator-core/src/queue.rs` | Bounded queue policies reject and drop-lru; FIFO within priority (OC-CORE-1001..1004) | Ensure queue semantics independent from HTTP | Enqueue to capacity; assert reject vs drop-lru; check FIFO order within class | done | 2025-09-17 |
| `orchestratord/src/http/control.rs` | Drain then reload toggles readiness; health includes last_error when set (ORCH-2006, ORCH-2003) | Critical control-plane lifecycle; cheap unit tests | Call `drain_pool` then `reload_pool` with API key; read `get_pool_health`; assert flags and headers | done | 2025-09-17 |
| `orchestratord/src/http/data.rs` | Guardrails and admission outcomes (ORCH-2001, ORCH-2007, OC-CORE-1001..1002) | Validate error mapping and 429 path deterministically | Send `TaskRequest` with sentinels: ctx<0 → 400 INVALID_PARAMS; retired/deprecated gating; huge expected_tokens → 429 with headers/body; enqueue reject path → 429 ADMISSION_REJECT | todo | 2025-09-17 |
| `orchestratord/src/metrics.rs` | Exporter names present; record_* helpers set metrics without panic (ORCH-METRICS-0001) | Observability contract stability | Assert `gather_metrics_text` contains names (done); call `record_stream_started/ended` with stub labels and ensure text contains updated counters/gauges | todo | 2025-09-17 |
| `orchestratord/src/backpressure.rs` | 429 headers and advisory body shape (ORCH-2007) | Stable backpressure representation | Assert required headers present; body has policy_label/retriable/retry_after_ms | done | 2025-09-17 |
| `orchestratord/src/http/auth.rs` | Require API key; accept only "valid" (planning stub) (ORCH-AUTH-0001) | Security guardrail | Missing/invalid header → 401; valid → Ok | done | 2025-09-17 |
| `worker-adapters/mock/src/lib.rs` | Submit yields started → token → end; props/health shape | Sanity check adapter API conformance | Collect stream to vector; assert order; props() has slots; health() ready | todo | 2025-09-17 |
| `orchestratord/src/admission.rs` | QueueWithMetrics emits metrics on enqueue/cancel | Inline smoke useful despite integration tests | Enqueue and cancel; scan metrics text for counters/gauges updates | covered-by-bdd (integration tests in `orchestratord/tests/admission_metrics.rs`); consider adding light inline smoke | 2025-09-17 |

Notes:
- Use `HeaderMap` with `X-API-Key: valid` for HTTP handler tests.
- Avoid non-determinism and external I/O; use defaults from `state::default_state()`.
- Keep test modules small and colocated under `#[cfg(test)]`.
