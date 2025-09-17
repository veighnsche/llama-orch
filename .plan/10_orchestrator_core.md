# Orchestrator-Core Plan (Queues & Placement)

## Goals
- Maintain FIFO queues with priorities (`interactive`, `batch`).
- Provide helpers for least-loaded GPU selection and queue estimates.
- Enforce guardrails (context length, token budgets) before enqueue.
- Supply determinism helpers for SSE pipeline.

## Tasks
1. Review `QueueWithMetrics` to ensure it surfaces queue length per priority and policy (`reject`/`drop-lru`).
2. Add simple throughput estimator used by `orchestratord` for `predicted_start_ms`.
3. Implement placement helper that consumes GPU telemetry (free VRAM) and returns target replica.
4. Extend property tests to cover policy behaviour, cancel races, and placement invariants.
5. Document public APIs in `README.md` or module docs for reuse by orchestrator and future services.

## Tests
- Property tests: `orchestrator-core/tests/props_queue.rs` (expand as tasks land).
- Unit tests for placement heuristics (simulate mixed VRAM devices).

## Integration Points
- `orchestratord/src/admission.rs` uses queue metrics + estimates.
- `pool-managerd` feeds GPU stats consumed by placement helper.
- `test-harness/determinism-suite/` relies on deterministic queue ordering.

## Risks
- Throughput estimation must remain simple; avoid heavy historical windows.
- Placement decisions need real-hardware validation (reference workstation).
