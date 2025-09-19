# pool-managerd — Component Specification (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## 0) Purpose & Scope

`pool-managerd` manages engine pools on a host: device discovery, preload and readiness gating, supervised engine lifecycle, draining/reload flows, and a registry consumed by placement. It does not implement placement itself.

In scope:
- Device discovery and representation (device masks; optional MIG awareness).
- Preload path (model ensure-present + engine ensure) and readiness transitions.
- Health checks, backoff, restart supervision.
- Registry of pools/replicas and capacity/health fields used by consumers.

Out of scope:
- HTTP API surface (exposed by `orchestratord`).
- Placement logic (owned by `orchestrator-core`).

## 1) Normative Requirements (RFC-2119)

- [ORCH-3500] The crate MUST define a `HealthStatus { live: bool, ready: bool }` used to report liveness and readiness.
- [ORCH-3501] The crate MUST provide a `Registry` with `PoolEntry` entries tracking at minimum:
  - `health: HealthStatus`, `last_heartbeat_ms: Option<i64>`, `version: Option<String>`, `last_error: Option<String>`, `active_leases: i32`.
- [ORCH-3502] Lease accounting MUST be saturating non-negative. `release_lease` MUST NOT underflow; `get_active_leases` MUST return `0` for unknown pools.
- [ORCH-3503] The registry MUST expose setters/getters: `set_health`, `get_health`, `set_last_error`, `get_last_error`, `set_heartbeat`, `get_heartbeat`, `set_version`, `get_version`, `allocate_lease`, `release_lease`, `get_active_leases`.
- [ORCH-3504] Readiness Gating: a pool MUST only flip `ready=true` after (a) model artifacts are present locally via `model-provisioner::ModelProvisioner::ensure_present*` and (b) engine staged via `engine-provisioner::EngineProvisioner::ensure` and (c) health check passes.
- [ORCH-3505] Draining: when draining is initiated, the pool MUST refuse new leases and SHOULD wait for in-flight leases to complete until deadline before forceful stop.
- [ORCH-3506] Device masks: pools SHOULD publish a stable `device_mask`; compute capability and VRAM totals/free SHOULD be captured for capacity reporting.
- [ORCH-3507] Observability: the crate SHOULD expose metrics/log fields for readiness transitions, restarts, backoff counts, `active_leases`, and version changes (emitted by the owning daemon).
- [ORCH-3508] Security: supervised engines SHOULD run as non-root where possible; container modes SHOULD default to rootless (`podman`).

## 2) Data Types & Semantics (current code)

- `HealthStatus { live, ready }` — see `src/health.rs`.
- `Registry`/`PoolEntry` — see `src/registry.rs` (fields listed in [ORCH-3501]).

## 3) Interfaces & Contracts

- Registry API as per [ORCH-3503]. Unknown pools are created on first setter call with defaults.
- Preload path interacts with `model-provisioner` and `engine-provisioner`; errors MUST be recorded via `set_last_error`.

## 4) Observability

- Suggested fields in logs: `pool_id`, `engine`, `engine_version`, `device_mask`, `model_id`, `restart_count`, `backoff_ms`, `last_error`.
- Suggested metrics (emitted by daemon): readiness transitions, restarts/backoff, capacity gauges `slots_*`, optional VRAM gauges.

## 5) Security

- No secrets stored; minimize privileges for supervised processes; isolate workdir.

## 6) Testing & Proof Bundle

- Unit tests MUST cover: registry getters/setters round-trip; non-negative leases.
- Integration tests SHOULD cover: preload gates readiness; drain/reload; supervision/backoff (with mocks/stubs).
- Include logs/snapshots for readiness transitions in proof bundles.

## 7) Open Questions

- How to represent mixed-GPU (tensor split) constraints in masks consistently across adapters?
- Which minimal health checks are required per engine (HTTP, gRPC, TCP)?

## 8) Refinement Opportunities

- Extend `PoolEntry` with capacity and VRAM/compute capability for placement.
- Publish performance hints (`perf_tokens_per_s`, `first_token_ms`) with bounded cardinality.
- Backoff tuning knobs and adaptive restart suppression.
