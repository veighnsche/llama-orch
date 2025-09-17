# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Home Profile v2.1 Compliance (Remaining)

### Contracts — OpenAPI (control)
- [ ] Unify capability discovery on `GET /v1/capabilities`; fully deprecate `/v1/replicasets` in code/tests/docs.

### Implementation — Data Plane & Sessions
- [ ] Admission estimates: incorporate active leases/concurrency (and later GPU throughput) for `queue_position` and `predicted_start_ms`.
- [ ] Determinism: honor `seed`/`determinism` end-to-end through adapters; expand determinism tests.
- [ ] Session store: eviction/expiry policy and precise cost budget integration; persist budgets across restarts.

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

### Worker Runtime & Adapters
- [ ] Replace mock adapter with real engine clients for llamacpp, vLLM, TGI, Triton; wire configs.
- [ ] Flesh out `pool-managerd` (preload, drain, health, hetero split, GPU pinning) and tests.

### Config Schema & Auth
- [ ] Home profile schema: add budgets/determinism/API key/bind/artifact storage/capability overrides; regenerate schema.
- [ ] Auth token configuration: real API key (and optional mTLS/OIDC) loading and enforcement.

### Observability & Metrics
- [ ] GPU + NVML integration for metrics and SSE `metrics` frames.
- [ ] Logging/tracing: structured logs for placement decisions; optional OpenTelemetry exporters.

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

## Progress Log (append entries here)

