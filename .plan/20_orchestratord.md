# Orchestratord Implementation Plan (Home Profile)

Spec references: `.specs/00_llama-orch.md`, `.specs/20-orchestratord.md`, `.specs/00_home_profile.md`

## Stage Alignment

| Stage | Focus | Deliverables |
|-------|-------|--------------|
| 6 | Admission → SSE | POST enqueue w/ metadata, SSE stream, cancel |
| 7 | Catalog & Reloads | Catalog CRUD, drain/reload, lifecycle Active/Retired |
| 8 | Capability Discovery | `/v1/replicasets` or `/v1/capabilities` with limits |
| 9 | Placement Heuristics | Mixed-GPU scheduling, queue estimates |
| 10 | Budgets & Sessions | Budget enforcement, session introspection |
| 11 | Tooling Policy | API token auth + outbound HTTP policy hook |
| 12 | BDD Coverage | Journeys for admission, stream, catalog, artifacts, budgets |
| 13 | Dashboards | Prometheus metrics, Grafana sample dashboards |
| 14 | Startup Self-tests | Preload, minimal decode, cancel, telemetry checks |
| 15 | Haiku | Haiku E2E gate on real hardware |

## Backlog (Current Focus: Stage 6–8)

1. **Admission Metadata**
   - Compute `queue_position`, `predicted_start_ms`, `backoff_ms` using `QueueWithMetrics` + simple throughput heuristic.
   - Echo correlation IDs, optional budget headers.
   - Tests: provider verify, BDD `admission_queue.feature`.

2. **SSE Streaming**
   - Stream adapters to SSE pipeline, inject `metrics` frames (queue depth, on_time_probability, budgets).
   - Determinism checks on same replica with seeds.

3. **Cancel Semantics**
   - Support cancel for queued + active jobs; update metrics and logs.

4. **Catalog & Artifact Storage**
   - Persist catalog entries (filesystem/sqlite) with verification warnings.
   - Local artifact registry storing plan snapshots & diffs via `/v1/artifacts`.

5. **Drain/Reload Lifecycle**
   - Drain flips readiness, reload swaps models atomically, rollback on failure.
   - Update `pool_managerd` readiness + metrics.

6. **Capability Endpoint**
   - Return engine versions, max context, concurrency hints; align with CLI expectations.

7. **Placement Heuristic**
   - Implement least-loaded GPU selection with VRAM awareness, update queue estimates.

8. **Session Registry & Budgets**
   - TTL, turn limits, KV usage, optional budgets exposed via GET/DELETE endpoints.

9. **Policy Hook**
   - Middleware exposing allow/deny decisions for outbound HTTP tools, logging decisions with IDs.

## Testing Checklist

- Provider verification (`orchestratord/tests/provider_verify.rs`).
- SSE integration tests (`orchestratord/tests/sse_tests.rs`, to be expanded).
- BDD features in `test-harness/bdd/tests/features/{data_plane,control_plane}`.
- Determinism suite using real adapters when available.
- Metrics lint (`test-harness/metrics-contract`).

## Observability & Docs

- Update `.docs/HOME_PROFILE.md` and `.docs/HOME_PROFILE_TARGET.md` when behaviour changes.
- Keep `ci/metrics.lint.json` synced with spec (`.specs/metrics/otel-prom.md`).
- Record progress in `TODO.md` and archive when complete.

## Risks / Open Items

- Mixed-GPU scheduling heuristics need validation on reference hardware.
- Artifact storage path and retention policy to be finalised.
- Outbound tool policy needs CLI integration tests.

Keep this plan in sync with the root TODO and README stage tracker.
