- 2025-09-30: Owner D — Model provisioner MVP features implemented
  - Code: `libs/provisioners/model-provisioner/src/lib.rs` adds config parsing (YAML/JSON), strict sha256 verification, metadata emission, handoff JSON writer, provenance JSONL appender, and unit tests.
  - README: `libs/provisioners/model-provisioner/README.md` documents deterministic Haiku model profile, handoff format, usage, and Refinement Opportunities.
  - Requirements: `requirements/55-model-provisioner.yaml` now includes ORCH-MODP-55xx IDs for MVP behaviors.
  - Specs: Verified `.specs/55-model-provisioner.md` exists; further schema details to be expanded during Owner C wiring.

# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Home Profile v2.1 Compliance (Remaining)

### Contracts — OpenAPI (control)

- [ ] Discovery MUST use `GET /v1/capabilities`; remove `/v1/replicasets` from OpenAPI, code, tests, and docs (no back-compat pre‑1.0).

### Implementation — Data Plane & Sessions

- [ ] Admission estimates: incorporate active leases/concurrency (and later GPU throughput) for `queue_position` and `predicted_start_ms`.
- [ ] Determinism: honor `seed`/`determinism` end-to-end through adapters; expand determinism tests.
- [ ] Session store: eviction/expiry policy and precise cost budget integration; persist budgets across restarts.
- [ ] Cancel semantics: propagate cancel to running streams/adapters; assert no tokens after cancel (race-free) and add tests.
- [ ] Enforce session max turns (default 8) at admission; expose in session info responses.

### Implementation — Control Plane & Catalog

- [ ] Catalog persistence: local storage (sqlite/JSON) for manifests/signatures/SBOM; permissive trust warnings.
- [ ] Pool drain/reload/health: connect to real `pool-managerd` and adapters; readiness/leases reflect real state.

### Implementation — Artifact Registry

- [ ] Persistent storage backend (filesystem content-addressed); unit tests and examples.
- [ ] Access control and retention/GC policies with tests.

### Capability Discovery & Placement

- [ ] Capability snapshot: derive from active pools/adapters with ctx limits/concurrency/rate limits; include API version.
- [ ] Least-loaded GPU scheduling (3090/3060) with NVML telemetry.
- [ ] Concurrency + lease accounting, exposed in metrics and capabilities.
- [ ] Remove `/v1/replicasets` route and handler; update router, tests, requirements.

### Worker Runtime & Adapters

- [ ] Replace mock adapter with real engine clients for llamacpp, vLLM, TGI, Triton; wire configs.
- [ ] Flesh out `pool-managerd` (preload, drain, health, hetero split, GPU pinning) and tests.

### Config Schema & Auth

- [ ] Home profile schema: add budgets/determinism/API key/bind/artifact storage/capability overrides; regenerate schema.
- [ ] Auth token configuration: real API key (and optional mTLS/OIDC) loading and enforcement.

### Observability & Metrics

- [ ] GPU + NVML integration for metrics and SSE `metrics` frames.
- [ ] Logging/tracing: structured logs for placement decisions; optional OpenTelemetry exporters.
- [ ] Add `engine_version` and `sampler_profile_version` to admission/stream logs per spec.

### Security & Tool Policy Hooks

- [ ] HTTP tooling guardrails (allow-list, secret redaction) with config and tests.
- [ ] Secrets handling & audit trail; redaction tests.

### CLI Consumer & Tooling

- [ ] Implement `llama-orch-cli` covering admission/streaming/sessions/artifacts/capabilities/policy.
- [ ] Developer ergonomics: queue status, metrics tail, artifact diff helpers.

### Removal & Simplification

- [ ] Remove legacy reduction scaffolding (e.g., `orchestratord/src/http/handlers.rs` re-export) and unused modules.
- [ ] Clean up placeholder logic (dummy IDs, static transcripts, stub catalog responses) per new implementations.

### Testing & Tooling

- [ ] Update provider/pact/BDD suites for budgets, determinism, artifacts, policy, mixed GPU scheduling; keep CI green.
- [ ] Reference environment integration test (feature-gated) against workstation; collect latency/budget metrics.

### Documentation & Examples

- [ ] Refresh HOME/spec docs after features land.
- [ ] Update `COMPLIANCE.md` mapping requirement IDs to proofs.
- [ ] Provide sample configs/runbooks for RTX 3090/3060 reference environment.

## P1 — Reference Environment Automation & Delivery

- [ ] Workstation bootstrap scripts for orchestrator/adapters; verify GPU detection.
- [ ] Dev box helpers (SSH tunnel, CLI env, artifact sync).
- [ ] Nightly validation job against reference hardware with archived artifacts.
- [ ] Packaging: release binaries + checksums; offline install instructions.

## P2 — Quality, Stretch, and Sustainability

- [ ] Optional mTLS/OIDC for remote teams.
- [ ] Local artifact browser UI (static site).
- [ ] Adaptive concurrency auto-tuning via telemetry.
- [ ] Incremental model download/cache management.
- [ ] Pluggable policy engine (Rego/WASM).

### Model Provisioner follow-ups (Owner D stubs)

- [ ] Cache eviction policy skeleton → implement LRU accounting and GC for model cache (`ModelProvisioner`) per `TODO(OwnerD-CACHE-EVICT-SKELETON)`.
- [ ] GGUF header parsing to populate `ctx_max` and tokenizer info in `ModelMetadata` per `TODO(OwnerD-CTX-PROBE)`.
- [ ] Provenance extension to include verification outcome and link to catalog entry per `TODO(OwnerD-PROVENANCE-EXTEND)`.

## Progress Log (append entries here)

- 2025-09-30: Model provisioner test suite parity with engine provisioner
  - Unit tests added under `libs/provisioners/model-provisioner/src/lib.rs` covering: relative/absolute path resolution, idempotent ensure, strict sha256 verification, HF fallback behavior with PATH-faked `huggingface-cli`, handoff + provenance writes. All tests run offline and are deterministic.
  - E2E smoke added at `libs/provisioners/model-provisioner/tests/hf_smoke.rs` (ignored by default). Run with `MODEL_ORCH_SMOKE=1 MODEL_ORCH_SMOKE_REF="hf:org/repo/path.gguf" cargo test -p model-provisioner -- --ignored`.
  - Dev-deps ensured: `tempfile = "3"`. No host mutations; tests isolate `PATH`/`HOME` via locks and temp dirs.
  - Repro: `cargo test -p model-provisioner --all-features -- --nocapture`; `cargo clippy -p model-provisioner --all-targets --all-features -- -D warnings`.

- 2025-09-17: Removed back-compat endpoints and shims per README_LLM golden rule
  - Specs: `.specs/20-orchestratord.md` now mandates `GET /v1/capabilities`; marks `/v1/replicasets` REMOVED pre‑1.0.
  - Crate Spec: `orchestratord/.specs/00_orchestratord.md` updated to state replicasets is removed; capability payload guidance extended.
  - OpenAPI: `contracts/openapi/control.yaml` path `/v1/replicasets` removed; description updated.
  - Router/Handlers: removed `/v1/replicasets` route from `orchestratord/src/lib.rs`; deleted `list_replicasets` handler from `orchestratord/src/http/control.rs`.
  - Tests/BDD: updated provider verify, BDD steps and feature to use `/v1/capabilities`; security step now targets capabilities for auth checks.
  - Harness: removed use of `orchestratord::http::handlers` shim; dispatches to concrete modules.
  - Requirements: updated `requirements/orchestratord.yaml` to reflect capabilities-only discovery.
