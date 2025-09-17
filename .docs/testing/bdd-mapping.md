# BDD → Handler Mapping (initial draft)

Scope: Map high-signal BDD features to primary handlers and modules. Use this to avoid duplicating coverage with in-file unit tests and to find gaps.

Conventions:
- Path format: `crate/path` for handlers, or `crate::module::func` for specific functions.
- Stable requirement IDs referenced inline where applicable.

---

## Control Plane

- test-harness/bdd/tests/features/control_plane/control_plane.feature
  - Handlers:
    - `orchestratord/src/http/control.rs:set_model_state` (ORCH-LC-1001)
    - `orchestratord/src/http/control.rs:drain_pool` (ORCH-2006)
    - `orchestratord/src/http/control.rs:reload_pool` (ORCH-2006)
    - `orchestratord/src/http/control.rs:get_pool_health` (ORCH-2003)
    - `orchestratord/src/http/control.rs:list_replicasets` (ORCH-2101)
    - `orchestratord/src/http/control.rs:get_capabilities` (ORCH-2102)
  - Cross-cutting:
    - `orchestratord/src/http/auth.rs:require_api_key` (ORCH-AUTH-0001)

## Data Plane — Admission and Streaming

- test-harness/bdd/tests/features/data_plane/enqueue_stream.feature
  - Handlers:
    - `orchestratord/src/http/data.rs:create_task` (ORCH-2001, OC-CORE-1001..1002)
    - `orchestratord/src/http/data.rs:stream_task` (SSE framing; ORCH-2010)
  - Cross-cutting:
    - `orchestratord/src/admission.rs:QueueWithMetrics`
    - `orchestrator-core/src/queue.rs:InMemoryQueue`

- test-harness/bdd/tests/features/data_plane/backpressure_429.feature
  - Handlers:
    - `orchestratord/src/http/data.rs:create_task` (429 sentinels)
    - `orchestratord/src/backpressure.rs:*` (ORCH-2007)

- test-harness/bdd/tests/features/data_plane/error_taxonomy.feature
  - Handlers:
    - `orchestratord/src/http/data.rs:create_task` (error envelopes)
    - `orchestratord/src/http/data.rs:stream_task` (SSE error mapping)
  - Types:
    - `orchestratord/src/errors.rs:ErrorEnvelope`

- test-harness/bdd/tests/features/sse/sse_details.feature
  - Handlers:
    - `orchestratord/src/http/data.rs:stream_task` (SSE order: started → token → metrics → end)

- test-harness/bdd/tests/features/sse/deadlines_sse_metrics.feature
  - Metrics helpers:
    - `orchestratord/src/metrics.rs:record_stream_started`
    - `orchestratord/src/metrics.rs:record_stream_ended`

## Observability

- test-harness/bdd/tests/features/observability/basic.feature
  - Handler:
    - `orchestratord/src/http/observability.rs:metrics_endpoint`
  - Helpers:
    - `orchestratord/src/metrics.rs:gather_metrics_text`

- test-harness/bdd/tests/features/observability/metrics_observability.feature
  - Helpers:
    - `orchestratord/src/metrics.rs:gather_metrics_text`

## Orchestrator Core

- test-harness/bdd/tests/features/orchestrator_core/admission_guards.feature
  - Components:
    - `orchestrator-core/src/queue.rs:InMemoryQueue`
    - `orchestratord/src/admission.rs:QueueWithMetrics`

- test-harness/bdd/tests/features/orchestrator_core/watchdog_timeouts.feature
  - Placeholder in current code; no direct handler; covered by future policy hooks.

## Placement & Queue Estimates

- test-harness/bdd/tests/features/scheduling/*.feature (to be rewritten for mixed-GPU scenarios)
  - Components:
    - `orchestratord/src/placement.rs` (least-loaded heuristic, queue estimates)
    - `orchestratord/src/http/data.rs:create_task` (queue position / predicted start)
    - `orchestratord/src/metrics.rs` (queue depth gauges)

## Security

- test-harness/bdd/tests/features/security/security.feature
  - Cross-cutting:
    - `orchestratord/src/http/auth.rs:require_api_key` used by all handlers.

## Pool Manager

- test-harness/bdd/tests/features/pool_manager/pool_manager_lifecycle.feature
  - Components:
    - `pool-managerd/src/registry.rs`
    - `orchestratord/src/http/control.rs:get_pool_health`, `drain_pool`, `reload_pool`

## Adapters

- test-harness/bdd/tests/features/adapters/*.feature
  - Components:
    - `worker-adapters/*/src/lib.rs` (adapter health/props; mock streams)
    - `orchestratord/src/http/control.rs:list_replicasets` (adapter-enriched payload)

---

Notes:
- This mapping is a starting point. Update it as handlers evolve and new features are introduced.
- Use this map to prioritize where in-file unit tests add value beyond BDD coverage.
