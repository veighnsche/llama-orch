# pool-managerd — Contracts

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Provided Contracts

- Registry API (in-crate)
  - `set_health/get_health`, `set_last_error/get_last_error`, `set_heartbeat/get_heartbeat`, `set_version/get_version`, `allocate_lease/release_lease/get_active_leases`.
  - Semantics: unknown pools created on first setter with defaults; leases never negative.
- Health/readiness model
  - `HealthStatus { live, ready }` for liveness and readiness reporting.
- Preload lifecycle (behavioral)
  - Readiness flips to true only after model staged and engine ensured and health check passes.

## Consumed Contracts (Expectations on Others)

- Model staging
  - `model-provisioner::ModelProvisioner::ensure_present*` — returns `ResolvedModel { id, local_path }`.
- Engine preparation
  - `engine-provisioner::EngineProvisioner::ensure` — builds/spawns engine, returns `Result<()>`.
- Orchestrator control plane
  - `orchestratord` will query health and reflect registry state via `GET /v1/pools/{id}/health` and control drain/reload.

## Data Exchange

- From model-provisioner: `ResolvedModel` id/path; optional digest verification status.
- To engine-provisioner: pool config and device mask for ensure path; ports and flags.
- To orchestrator: health summary, last_error, version, heartbeat.

## Error Semantics

- Preload errors recorded via `set_last_error`; readiness remains false.
- Draining respects deadlines; force-stop escalates error in registry.

## Versioning & Compatibility

- Registry fields may evolve; `orchestratord` should tolerate additional fields and missing optional ones.

## Observability

- Emit logs with fields: `pool_id`, `engine`, `engine_version`, `device_mask`, `model_id`, `restart_count`, `backoff_ms`, `last_error`.

## Security & Policy

- Engines should run as non-root; prefer rootless containers; isolate workdir.

## Testing Expectations

- Unit: registry round-trip and lease accounting.
- Integration: preload gates readiness; drain/reload; backoff supervision with stubs.

## Test Ownership

- Crate-local tests OWN registry behavior, readiness gating, and supervision/backoff logic. Cross-crate flows (admission→stream/cancel over HTTP) are validated by the root BDD harness; see `/.specs/72-bdd-harness.md`.

## Refinement Opportunities

- Publish capacity/VRAM and perf hints for placement; bounded cardinality.
- Standardize health probe contracts per engine.
