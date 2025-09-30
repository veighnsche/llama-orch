# E2E haiku Pipeline Checklist (llama.cpp, uninstantiated engine)

This document audits the happy-path pipeline and lists concrete tasks to pass the e2e-haiku test. Each item references code paths and includes TODO IDs added in code for traceability.

## Happy Path Timeline (expected)

0. Config validated at `orchestratord` startup
1. Client requests `llama-3` on `llama.cpp`
2. Orchestrator checks pool-manager for an existing ready replica of that model/engine
2.5. Orchestrator checks catalog for model and engine artifacts readiness
3. If not present, consult policy to provision
4. Engine provisioner builds/starts `llama.cpp` (llama-server)
5. Model provisioner downloads/prepares `llama-3` GGUF
6. Provisioners notify pool-manager that pool is ready & healthy
7. Orchestrator binds worker adapter (llamacpp-http) to that pool/replica and dispatches the task
8. llama.cpp streams tokens via SSE back to user

## Client Call Sequence (spec-confirmed, v2)

- **[enqueue]** `POST /v2/tasks` → returns `202 Accepted` with `AdmissionResponseV2 { task_id, queue_position, predicted_start_ms, backoff_ms, streams?, preparation? }`.
  - `streams`: `{ sse, sse_verbose }` direct URLs for streaming.
  - `preparation`: optional steps `{ steps: [{ kind: engine_provision|model_fetch|pool_warmup, description?, estimated_ms? }] }`.
- **[stream]** Use `streams.sse` (base) or `streams.sse_verbose` (equivalent to `?verbose=true`) → `GET /v2/tasks/{id}/events` returns `text/event-stream` with events: `started` → repeated `token` → optional repeated `metrics` → `end` (or `error`).
- **[narration logs]** are emitted via JSON logs (with a `human` string), correlated by `X-Correlation-Id`. SSE is not a raw log stream.
- References: `contracts/openapi/data.yaml`, `CONSUMER_CAPABILITIES.md`, and `consumers/llama-orch-sdk/CLIENT_HANDBOOK.md` (all v2).

## Findings and Gaps (by step)

- 0 — Config validation
  - Current: No config-schema backed load/validation at startup.
  - Gap: Validate and fail-fast on missing/invalid pools/placement.
  - Code: `bin/orchestratord/src/app/bootstrap.rs`.
  - TODO: ORCHD-CONFIG-VALIDATE-0001.

- 1 — Request ingress
  - Current: `POST /v2/tasks` returns `202 Accepted` with `AdmissionResponseV2` (includes the same `task_id`). Enqueue admits into the queue and seeds data for `started` SSE.
  - Gap: Sentinel validations used; need policy-backed checks and budgets.
  - Code: `bin/orchestratord/src/api/data.rs` (`create_task()`). Spec: `contracts/openapi/data.yaml`.

- 2 — Pool-manager check for running model/engine
  - Current: `AppState.pool_manager` holds a `pool_managerd::registry::Registry`, but create/stream paths don't consult it.
  - Gap: Before enqueue/dispatch, check target pool health/readiness.
  - Code: `bin/orchestratord/src/api/data.rs`, `bin/orchestratord/src/services/streaming.rs`.
  - TODOs: to be added when wiring real placement/health gating.

- 2.5 — Catalog check
  - Current: `catalog-core` exists; `services/catalog.rs` is stub.
  - Gap: Verify model presence (Active) or trigger provisioning per policy.
  - Code: `bin/orchestratord/src/api/data.rs`.
  - TODO: ORCHD-CATALOG-CHECK-0006.

- 3 — Provisioning policy
  - Current: No policy layer to auto-provision engine/model; admission ignores model presence.
  - Gap: Policy to authorize provisioning and choose pool.
  - Code: `bin/orchestratord/src/api/data.rs`.
  - TODO: ORCHD-PROVISION-POLICY-0005.

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
  - Current: Missing. `pool-managerd::registry` has methods, but no calls from provisioners.
  - Gap: After health OK, set health Ready, version, slots, heartbeat.
  - Code: `libs/provisioners/engine-provisioner/.../llamacpp/mod.rs`, `libs/pool-managerd/src/registry.rs`.
  - TODO: ENGINE-PROV-POOL-NOTIFY-0003.

- 7 — Orchestrator binds adapter and dispatches
  - Current: Adapter binding requires feature `llamacpp-adapter` and env `ORCHD_LLAMACPP_URL`.
  - Gap A: No auto-bind from handoff files.
  - Gap B: No placement policy; always default pool.
  - Code: `bin/orchestratord/src/app/bootstrap.rs`, `bin/orchestratord/src/services/placement.rs`.
  - TODOs: ORCHD-HANDOFF-AUTOBIND-0002, placement implementation.

- 8 — Streaming tokens SSE
  - Current: `services/streaming.rs` tries adapter then falls back to deterministic SSE; adapter exists but health/props/cancel/streaming are MVP.
  - Gap A: Request building is stubbed (doesn’t use user’s original request). 
  - Gap B: Incremental decode/backpressure/cancel-on-disconnect not wired (proposal exists).
  - Code: `bin/orchestratord/src/services/streaming.rs`, `libs/worker-adapters/llamacpp-http/src/lib.rs`.
  - TODOs: ORCHD-STREAM-1101/1102/1103, OwnerB-LLAMACPP-STREAM, -HEALTH, -PROPS, -CANCEL, -VERSION.

## Additional Gaps

- Placement overrides (pin model/engine to GPU/pool) not implemented.
  - Code: `services/placement.rs` uses default route.
- Catalog service is a stub.
  - Code: `services/catalog.rs`.
- Worker registration flow with AUTH token exists, but engine-provisioner doesn’t call it.
  - Code: `bin/orchestratord/src/api/control.rs::register_worker()`.

## TODO Markers Added in Code

- `bin/orchestratord/src/app/bootstrap.rs`
  - ORCHD-CONFIG-VALIDATE-0001: validate config at startup
  - ORCHD-HANDOFF-AUTOBIND-0002: auto-bind adapters from engine handoff files
- `bin/orchestratord/src/api/data.rs`
  - ORCHD-CATALOG-CHECK-0006: catalog existence/state check
  - ORCHD-PROVISION-POLICY-0005: invoke provisioning per policy
  - ORCHD-ADMISSION-STREAMS-0008: populate `AdmissionResponse.streams`
  - ORCHD-ADMISSION-PREPARATION-0009: populate `AdmissionResponse.preparation`
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

- 1. Orchestrator returns 202 for enqueue and 200 for stream (already true).
- 2. SDK responsibilities for the test:
  - Set `X-Correlation-Id` on POST and GET; collect JSON logs filtered by that ID (narration `human` lines included).
  - Parse SSE incrementally and persist a transcript (ordered frames with timestamps).
  - Build a proof bundle: `request.json`, `admission.json`, `sse_transcript.jsonl`, `logs.jsonl`, `verification.json`.
  - haiku anti-stub rule: include the current minute spelled out (client derives the minute at stream start and verifies it appears exactly once in the generated text).
- 3. For real-run gate (REQUIRE_REAL_LLAMA=1):
  - Option A (fast path): start llama-server manually and export `ORCHD_LLAMACPP_URL`, build with feature `llamacpp-adapter`. This satisfies streaming via adapter without provisioning.
  - Option B (target path): implement ORCHD-HANDOFF-AUTOBIND-0002 watcher so engine-provisioner’s handoff auto-registers the adapter.
- See `consumers/llama-orch-sdk/CLIENT_HANDBOOK.md` for SDK instructions and example sequences.

Verification commands:
- `cargo test -p test-harness-e2e-haiku -- --ignored --nocapture` (with REQUIRE_REAL_LLAMA=1)
- `cargo run -p engine-provisioner -- --config requirements/50-engine-provisioner.yaml` (as applicable)
- `cargo run -p orchestratord` (with `--features llamacpp-adapter` and `ORCHD_LLAMACPP_URL=http://127.0.0.1:PORT` for Option A)

## Target Path to Full Happy Flow (mid-term)

- Implement config load/validation (ORCHD-CONFIG-VALIDATE-0001).
- Implement catalog check + provisioning policy (ORCHD-CATALOG-CHECK-0006, ORCHD-PROVISION-POLICY-0005).
- Engine-provisioner: notify pool-manager and orchestrator (ENGINE-PROV-POOL-NOTIFY-0003), enforce GPU-only (ENGINE-PROV-GPU-ENFORCE-0007).
- Auto-bind adapters from handoff files (ORCHD-HANDOFF-AUTOBIND-0002).
- Placement service using pool health, slots, and overrides; support explicit pinning.
- Streaming: build request from admission, incremental SSE, cancel-on-disconnect.

## Refinement Opportunities

- Catalog fetchers: native `hf:` fetcher instead of shelling to `huggingface-cli`.
- Observability: align metric names/labels with `.specs/metrics/otel-prom.md` and `README_LLM`.
- Security: worker registration with scoped tokens, rotate tokens, avoid env var leaks in logs.
- Test harness: add assertion on tokens_out delta and started/token/end order for real engine runs.

## Cross-References

- Orchestrator router: `bin/orchestratord/src/app/router.rs`
- Streaming: `bin/orchestratord/src/services/streaming.rs`
- Pool registry: `libs/pool-managerd/src/registry.rs`
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
