# E2E haiku Pipeline Checklist (llama.cpp, uninstantiated engine)
This document audits the happy-path pipeline and lists concrete tasks to pass the e2e-haiku test. Each item references code paths and includes TODO IDs added in code for traceability.
**Updated for Cloud Profile**: This checklist now reflects both HOME_PROFILE (single-machine) and CLOUD_PROFILE (distributed) architectures.
## Happy Path Timeline (Current)
### CLOUD_PROFILE (Current - Distributed)
0. Control plane (orchestratord) starts, validates config
1. GPU worker (pool-managerd) registers via `POST /v2/nodes/register` with Bearer token
2. pool-managerd sends periodic heartbeats (`POST /v2/nodes/{id}/heartbeat`) with pool status
3. Client sends `POST /v2/tasks` to orchestratord
4. orchestratord checks service registry for healthy nodes with model availability
5. orchestratord performs model-aware placement (least-loaded, model filter)
6. orchestratord dispatches to pool via HTTP (adapter binds to remote pool URL)
7. On GPU worker: pool-managerd watches local handoff files, manages engine lifecycle
8. Adapter (on orchestratord or worker) streams tokens via SSE back through orchestratord
9. orchestratord relays SSE stream to client
### Legacy HOME_PROFILE (Deprecated - Removed)
**Note**: HOME_PROFILE handoff watcher was removed as part of cloud profile migration.
The project is now cloud-first. For single-machine deployments, run orchestratord and pool-managerd
on the same node with CLOUD_PROFILE mode.
## Client Call Sequence (spec-confirmed, v2)
- **[enqueue]** `POST /v2/tasks` → returns `202 Accepted` with `AdmissionResponseV2 { task_id, queue_position, predicted_start_ms, backoff_ms, streams?, preparation? }`.
  - `streams`: `{ sse, sse_verbose }` direct URLs for streaming.
  - `preparation`: optional steps `{ steps: [{ kind: engine_provision|model_fetch|pool_warmup, description?, estimated_ms? }] }`.
- **[stream]** Use `streams.sse` (base) or `streams.sse_verbose` (equivalent to `?verbose=true`) → `GET /v2/tasks/{id}/events` returns `text/event-stream` with events: `started` → repeated `token` → optional repeated `metrics` → `end` (or `error`).
- **[narration logs]** are emitted via JSON logs (with a `human` string), correlated by `X-Correlation-Id`. SSE is not a raw log stream.
- References: `contracts/openapi/data.yaml`, `CONSUMER_CAPABILITIES.md`, and `consumers/llama-orch-sdk/CLIENT_HANDBOOK.md` (all v2).
## Findings and Gaps (by step)
- 0 — Config validation
  - Current: Config module loads and validates orchestrator configuration at startup with fail-fast semantics.
  - Implementation: `bin/orchestratord/src/config.rs` provides `Config::load()` with validation for admission capacity, queue policy, placement strategy, and cloud profile settings. Bootstrap calls `Config::load()` and panics on validation failure per OC-CONFIG-6001.
  - Code: `bin/orchestratord/src/app/bootstrap.rs`, `bin/orchestratord/src/config.rs`.
  - TODO: ORCHD-CONFIG-VALIDATE-0001.
  - Status: ✅ Implemented
- 1 — Request ingress
  - Current: `POST /v2/tasks` returns `202 Accepted` with `AdmissionResponseV2` (includes the same `task_id`). Enqueue admits into the queue and seeds data for `started` SSE.
  - Gap: Sentinel validations used; need policy-backed checks and budgets.
  - Code: `bin/orchestratord/src/api/data.rs` (`create_task()`). Spec: `contracts/openapi/data.yaml`.
  - Status: ✅ Basic admission done, ⚠️ needs sentinel removal & budget enforcement
- 2 — Pool-manager check for running model/engine
  - Current: `AppState.pool_manager` holds a `pool_managerd::registry::Registry`, streaming path checks health via `should_dispatch()`.
  - Gap: Admission path doesn't check pool health before enqueue.
  - Code: `bin/orchestratord/src/api/data.rs`, `bin/orchestratord/src/services/streaming.rs`.
  - TODOs: to be added when wiring real placement/health gating.
  - Status: ✅ Streaming has health gate (ORCHD-POOL-HEALTH-GATE-0010 done), ⚠️ admission needs it
- 2.5 — Catalog check
  - Current: `catalog-core` fully implemented with FsCatalog, ModelRef parsing, verify_digest. HTTP endpoints exist.
  - Gap: Admission doesn't call catalog to verify model presence.
  - Code: `bin/orchestratord/src/api/data.rs`, `libs/catalog-core/src/lib.rs`.
  - TODO: ORCHD-CATALOG-CHECK-0006.
  - Status: ✅ Infrastructure complete, ❌ integration in admission missing
- 3 — Provisioning policy
  - Current: No policy layer to auto-provision engine/model; admission ignores model presence.
  - Gap: Policy to authorize provisioning and choose pool.
  - Code: `bin/orchestratord/src/api/data.rs`.
  - TODO: ORCHD-PROVISION-POLICY-0005.
  - Status: ❌ Not implemented
- 4 — Engine provision
  - Current: `engine-provisioner` supports llama.cpp from source with Arch-friendly preflight (pacman). Starts `llama-server`, writes handoff JSON.
  - Gap A: GPU-only enforcement isn’t strict; build can proceed without CUDA flags.
  - Gap B: No notification to orchestrator/pool-manager on Ready.
  - Code: `libs/provisioners/engine-provisioner/src/providers/llamacpp/mod.rs`.
  - TODOs: ENGINE-PROV-POOL-NOTIFY-0003, ENGINE-PROV-CLEANUP-0004, ENGINE-PROV-GPU-ENFORCE-0007 (see below under TODO markers).
- 5 — Model provision
  - Current: `model-provisioner` ensures presence; `hf:` uses `huggingface-cli` fallback.
  - Gap: Orchestrator doesn’t call it; only engine-provisioner does internally.
  - Code: `libs/provisioners/model-provisioner/src/lib.rs`.
- 6 — Provisioners notify pool-manager
  - **CLOUD_PROFILE**: Handoff watcher in pool-managerd (local filesystem watch)
  - Current: pool-managerd watches handoffs locally, reports readiness via heartbeats
  - Code: `libs/gpu-node/handoff-watcher/` (CLOUD_PROFILE), `bin/pool-managerd/` (registry + heartbeat)
  - Status: ✅ CLOUD_PROFILE heartbeat reporting complete, ✅ Handoff watcher moved to pool-managerd
  - **Removed**: `bin/orchestratord/src/services/handoff.rs` (HOME_PROFILE handoff watcher deleted)
- 7 — Orchestrator binds adapter and dispatches
  - **CLOUD_PROFILE**: Adapter binds to remote pool URL from service registry
  - Current: Model-aware placement in `bin/orchestratord/src/services/placement_v2.rs`
  - Current: Placement strategies: round-robin, least-loaded (filters by model availability)
  - Code: `bin/orchestratord/src/app/bootstrap.rs`, `bin/orchestratord/src/services/placement_v2.rs`
  - Status: ✅ Model-aware placement done (CLOUD_PROFILE), ⚠️ pin override enforcement missing
- 8 — Streaming tokens SSE
  - Current: `services/streaming.rs` tries adapter with health gate, falls back to deterministic SSE. Request built from admission snapshot.
  - Gap A: Cancellation uses polling, not structured token.
  - Gap B: Error events not emitted as SSE frames.
  - TODOs: ORCHD-STREAM-1101/1102/1103, OwnerB-LLAMACPP-STREAM, -HEALTH, -PROPS, -CANCEL, -VERSION.
  - Status: ✅ Health gate done, ✅ request building from admission done, ⚠️ cancellation & error events need work
## Additional Gaps
- Placement overrides (pin model/engine to GPU/pool) not enforced.
  - Code: `services/placement.rs` uses default route.
  - Status: ⚠️ Contract exists (`TaskRequest.placement.pin_pool_id`), enforcement missing
- Catalog infrastructure complete, integration in admission missing.
  - Code: `libs/catalog-core/src/lib.rs` ✅, `bin/orchestratord/src/api/data.rs` ❌
- Worker registration flow with AUTH token exists, but engine-provisioner doesn't call it.
  - Code: `bin/orchestratord/src/api/control.rs::register_worker()` ✅
## TODO Markers Added in Code
- `bin/orchestratord/src/app/bootstrap.rs`
  - ORCHD-CONFIG-VALIDATE-0001: validate config at startup ✅ DONE
- `bin/orchestratord/src/api/data.rs`
  - ORCHD-CATALOG-CHECK-0006: catalog existence/state check ❌
  - ORCHD-PROVISION-POLICY-0005: invoke provisioning per policy ❌
  - ORCHD-ADMISSION-STREAMS-0008: populate `AdmissionResponse.streams` ✅ DONE
  - ORCHD-ADMISSION-PREPARATION-0009: populate `AdmissionResponse.preparation` ✅ DONE
- `libs/provisioners/engine-provisioner/src/providers/llamacpp/mod.rs`
  - ENGINE-PROV-POOL-NOTIFY-0003: notify pool-manager/orchestrator on readiness
  - ENGINE-PROV-CLEANUP-0004: cleanup on failures to avoid stale state
  - ENGINE-PROV-GPU-ENFORCE-0007: enforce GPU-only per workspace policy (to add)
 - `bin/orchestratord/src/api/data.rs::stream_task` and `services/streaming.rs`
  - ORCHD-STREAM-VERBOSE-0011: accept `?verbose=true` and emit narration in some `metrics` frames
These complement existing TODOs:
- `bin/orchestratord/src/services/streaming.rs`: ORCHD-STREAM-1101..1103 and ORCHD-REQUEST-STUB
- `libs/worker-adapters/llamacpp-http/src/lib.rs`: health/props/stream/cancel/version TODOs
## Minimal Work to Pass e2e-haiku (near-term)
- [x] 1. Orchestrator returns 202 for enqueue and 200 for stream (already true).
  - Set `X-Correlation-Id` on POST and GET; collect JSON logs filtered by that ID (narration `human` lines included).
  - Parse SSE incrementally and persist a transcript (ordered frames with timestamps).
  - Build a : `request.json`, `admission.json`, `sse_transcript.jsonl`, `logs.jsonl`, `verification.json`.
  - haiku anti-stub rule: include the current minute spelled out (client derives the minute at stream start and verifies it appears exactly once in the generated text).
- 3. For real-run gate (REQUIRE_REAL_LLAMA=1):
  - Option A (fast path): start llama-server manually and export `ORCHD_LLAMACPP_URL`, build with feature `llamacpp-adapter`. This satisfies streaming via adapter without provisioning.
  - Option B (target path): ORCHD-HANDOFF-AUTOBIND-0002 watcher implemented in `bin/orchestratord/src/services/handoff.rs`
- See `consumers/llama-orch-sdk/CLIENT_HANDBOOK.md` for SDK instructions and example sequences.
Verification commands:
- `cargo test -p test-harness-e2e-haiku -- --ignored --nocapture` (with REQUIRE_REAL_LLAMA=1)
- `cargo run -p engine-provisioner -- --config requirements/50-engine-provisioner.yaml` (as applicable)
{{ ... }}
## Target Path to Full Happy Flow (mid-term)
- [x] Implement config load/validation (ORCHD-CONFIG-VALIDATE-0001) ✅ DONE.
- Implement catalog check + provisioning policy (ORCHD-CATALOG-CHECK-0006, ORCHD-PROVISION-POLICY-0005).
- Engine-provisioner: notify pool-manager and orchestrator (ENGINE-PROV-POOL-NOTIFY-0003), enforce GPU-only (ENGINE-PROV-GPU-ENFORCE-0007).
- [x] Auto-bind adapters from handoff files (ORCHD-HANDOFF-AUTOBIND-0002) ✅ DONE.
- Placement service using pool health, slots, and overrides; support explicit pinning.
- [x] Streaming: build request from admission ✅ DONE, incremental SSE ✅ DONE, cancel-on-disconnect ⚠️ needs work.
## Refinement Opportunities
- Catalog fetchers: native `hf:` fetcher instead of shelling to `huggingface-cli`.
- Observability: align metric names/labels with `.specs/metrics/otel-prom.md` and `README_LLM`.
- Security: worker registration with scoped tokens, rotate tokens, avoid env var leaks in logs.
- Test harness: add assertion on tokens_out delta and started/token/end order for real engine runs.
## Cloud Profile Architecture References
**CLOUD_PROFILE (Current Architecture)**:
- Orchestrator router: `bin/orchestratord/src/app/router.rs`
- Streaming: `bin/orchestratord/src/services/streaming.rs`
- Pool registry: `bin/pool-managerd/src/core/registry.rs`
- Node endpoints: `bin/orchestratord/src/api/nodes.rs` (register, heartbeat, deregister)
- Service registry: `libs/control-plane/service-registry/` (node tracking, health)
- Node registration: `libs/gpu-node/node-registration/` (GPU worker registration)
- Handoff watcher: `libs/gpu-node/handoff-watcher/` (local filesystem watch on GPU node)
- Placement v2: `bin/orchestratord/src/services/placement_v2.rs` (model-aware, least-loaded)
- Catalog availability: `bin/orchestratord/src/api/catalog_availability.rs`
**Common**:
- Engine provisioning: `libs/provisioners/engine-provisioner/src/providers/llamacpp/mod.rs`
- Model provisioning: `libs/provisioners/model-provisioner/src/lib.rs`
- Worker adapter: `libs/worker-adapters/llamacpp-http/src/lib.rs`
- haiku harness: `test-harness/e2e-haiku/tests/e2e_client.rs`
- Client flow note: `FINDINGS_ENQUEUE_STREAM_FLOW.md`
- SDK guide: `consumers/llama-orch-sdk/CLIENT_HANDBOOK.md`
## Spec Change Recap (v2)
- Data-plane OpenAPI (`contracts/openapi/data.yaml`) updated to v2 paths and now includes:
  - `GET /v2/tasks/{id}/events?verbose=true` (optional).
  - `AdmissionResponseV2.streams` (links to `sse`, `sse_verbose`).
  - `AdmissionResponseV2.preparation` (list of steps before decode).
- Docs updated: `CONSUMER_CAPABILITIES.md`, `CLIENT_HANDBOOK.md`.
- TODOs added: ORCHD-STREAM-VERBOSE-0011, ORCHD-ADMISSION-PREPARATION-0009.
- TODO: refresh diagrams/examples referencing `/v1` in `README.md` and BDD tests.
