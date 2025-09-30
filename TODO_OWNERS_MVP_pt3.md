# Owners MVP pt3 — Provisioners Connection Plan (Owner E & Owner F)

This plan focuses on the two crates that directly connect to the provisioners per `CHECKLIST_HAIKU.md`: `pool-managerd/` (readiness + lifecycle registry) and `orchestratord/` (handoff auto-bind + dispatch/stream gating). It lists what is done, what remains, and the testing plan in the Spec → Contract → Tests → Code cadence.

## Scope and References

- Provisioners (done):
  - `libs/provisioners/engine-provisioner/` → `providers/llamacpp/mod.rs` writes handoff JSON and starts `llama-server`. TODOs: `ENGINE-PROV-POOL-NOTIFY-0003`, `ENGINE-PROV-CLEANUP-0004`, `ENGINE-PROV-GPU-ENFORCE-0007`.
  - `libs/provisioners/model-provisioner/` → present; engine-provisioner calls it internally to ensure GGUF.
- Direct connections to wire now:
  - `libs/pool-managerd/` → `src/registry.rs` (health/slots/heartbeat/version) to be notified by provisioners.
  - `bin/orchestratord/` → `app/bootstrap.rs` (auto-bind from handoff), `services/streaming.rs` (health-gated dispatch), `api/data.rs` (admission streams/preparation), `api/control.rs` (optional worker register).
- Checklist: `CHECKLIST_HAIKU.md` steps 6, 7, and related TODO markers.

---

## Owner E — pool-managerd (registry readiness + lifecycle)

### What is done

- `libs/pool-managerd/src/registry.rs` provides:
  - `set_health()`, `get_health()`, `set_last_error()`, `get_last_error()`
  - `set_version()`, `get_version()`, `set_heartbeat()`, `get_heartbeat()`
  - Lease counters (`allocate_lease()`, `release_lease()`, `get_active_leases()`)
- Unit tests exist validating health/meta and lease invariants.

### What remains (mid-term target to support provisioners)

- [x] Add a minimal readiness API for provisioners to call:
  - [x] `register_ready_from_handoff(pool_id, handoff: serde_json::Value)` that sets:
    - [x] `health = { live: true, ready: true }`
    - [x] `engine_version`, `device_mask` (from handoff or config), `slots_total/free`
    - [x] `last_heartbeat_ms = now()`; clear `last_error`
  - [x] Optional: `set_engine_meta(pool_id, version, digest, catalog_id)` if we want typed fields.
- Expose this via one of:
  - [x] a small in-crate helper callable by orchestrator on ingest, or
  - an HTTP control endpoint in `orchestratord` that forwards to the registry (Bearer auth).
- Backoff and drains integration stubs are present in crate; expose them as needed for placement gates.

### Testing plan

- Unit tests (add):
  - [x] `OC-POOL-3101`: registering from a valid handoff sets `ready=true`, fills version/slots, clears `last_error`.
  - [x] `OC-POOL-3102`: repeated registrations update heartbeat/version; never panic on partial handoff fields.
- Integration test (once orchestrator autobind exists):
  - Start orchestrator with file watcher, write a valid handoff JSON → assert `GET /control/pools/{id}/health` reflects Live+Ready and engine_version.

### File touch points

- [x] `libs/pool-managerd/src/registry.rs`
- Optional orchestrator glue: `bin/orchestratord/src/services/control.rs` or a new services module

---

## Owner F — orchestrator autobind + health-gated dispatch

### What is done

- [x] `bin/orchestratord/src/app/bootstrap.rs` binds a single llama.cpp adapter only when the feature `llamacpp-adapter` is enabled and envs are set (`ORCHD_LLAMACPP_URL`, pool/replica ids). This is an MVP shim.
- [x] `bin/orchestratord/src/services/streaming.rs` can stream via a bound adapter; otherwise falls back to deterministic SSE (MVP). Health gating implemented via `should_dispatch()`.
- [x] `bin/orchestratord/src/api/data.rs` (enqueue) returns `202 Accepted` and records admissions; `streams` and `preparation` fields populated.
- [x] `bin/orchestratord/src/api/control.rs::register_worker()` exists (Bearer-protected) and can bind a mock adapter in `mock-adapters` feature.

### What remains (near-term target path to happy flow)

- [x] Implement `ORCHD-HANDOFF-AUTOBIND-0002`:
  - [x] File watcher for `.runtime/engines/*.json` (handoffs written by engine-provisioner).
  - [x] For each handoff: bind `llamacpp-http` adapter using `url`, associate `pool_id`/`replica_id`.
  - [x] Update `state.pool_manager` via Owner E API to mark Ready and set engine meta.
  - **Implemented in**: `bin/orchestratord/src/services/handoff.rs`
- [x] Health/placement gate in streaming:
  - [x] Before dispatch: consult `state.pool_manager` for Live+Ready with `slots_free > 0`; else backoff/emit retry hints.
  - [x] Replace request stub: build adapter request from original admission (`TaskRequest`).
  - **Implemented in**: `bin/orchestratord/src/services/streaming.rs` (lines 36-50, 62-71, 243-289)
- [x] Admission response:
  - [x] Populate `streams.sse` and `streams.sse_verbose` (OpenAPI v2).
  - [x] Populate `preparation.steps` during provisioning when autobind triggers.
  - **Implemented in**: `bin/orchestratord/src/api/data.rs` (lines 109-136)

### Testing plan

- Unit tests (add):
  - [x] `ORCHD-AUTOBIND-UT-1001`: given a handoff JSON on disk, the watcher binds an adapter and updates registry.
    - **Implemented in**: `bin/orchestratord/src/services/handoff.rs` (test module, lines 162-195)
  - [x] `ORCHD-STREAM-UT-1101`: streaming consults pool health and refuses dispatch if not Ready.
    - **Implemented in**: `bin/orchestratord/src/services/streaming.rs` (test module, lines 334-394)
- Integration tests (stub engine):
  - [x] Start a stub Axum SSE server (like `llamacpp-http` tests) → write a handoff with its URL → POST `/v2/tasks` then `GET /v2/tasks/{id}/events` → assert event order and presence of `streams` in admission.
    - **Implemented in**: `bin/orchestratord/tests/handoff_autobind_integration.rs`
- Determinism tests (later, once real engine path is stable):
  - With `REQUIRE_REAL_LLAMA=1`, verify byte-exact token sequences for identical seeds and sampler profile flags (`--parallel 1`, `--no-cont-batching`).

### File touch points

- [x] `bin/orchestratord/src/app/bootstrap.rs` (updated to call handoff watcher)
- [x] `bin/orchestratord/src/services/streaming.rs` (health gate implemented)
- [x] `bin/orchestratord/src/api/data.rs` (streams/preparation populated)
- [x] `bin/orchestratord/src/services/handoff.rs` (NEW: handoff autobind watcher)
- [x] `bin/orchestratord/tests/handoff_autobind_integration.rs` (NEW: integration tests)

---

## Cross-crate: Provisioner follow-ups (E+F dependencies)

- Engine-provisioner (`ENGINE-PROV-POOL-NOTIFY-0003`): after health OK, call orchestrator control (Bearer) or write a sidecar `ready.json` that the watcher ingests to update `pool-managerd`.
- Engine-provisioner (`ENGINE-PROV-CLEANUP-0004`): ensure child kill + pid/handoff cleanup on failure.
- GPU enforcement (`ENGINE-PROV-GPU-ENFORCE-0007`): current configure path enforces GPU when flags request it; add explicit workspace policy gate to forbid CPU builds where required.

---

## Verification commands

- Unit:
  - `cargo test -p pool-managerd -- --nocapture`
  - `cargo test -p orchestratord -- --nocapture`
- Integration (stub engine):
  - `cargo test -p worker-adapters-llamacpp-http --tests -- --nocapture`
  - Orchestrator autobind IT: start orchestrator (feature `llamacpp-adapter`), start stub SSE server, write handoff, curl POST/GET.
- Real-run gate (optional):
  - `cargo run -p provisioners-engine-provisioner -- --config requirements/50-engine-provisioner.yaml`
  - `ORCHD_LLAMACPP_URL=http://127.0.0.1:PORT cargo run -p orchestratord --features llamacpp-adapter`

---

## Owner assignment summary

- Owner E: pool-managerd registry wiring (readiness API, unit tests, orchestrator glue)
- Owner F: orchestrator handoff autobind + health-gated streaming, admission streams/preparation, tests

Notes:

- Follow Spec → Contract → Tests → Code. Update `.specs/` and `contracts/` when adding new fields/paths; then regenerate via `cargo xtask regen-openapi`/`regen-schema`.
- Emit proof bundles in tests per `libs/proof-bundle` standard when feasible (see `libs/worker-adapters/http-util/tests/unit.rs`).
