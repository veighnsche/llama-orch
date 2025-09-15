# llama-orch-cli — Dev Tool CLI (Auto Programmer)

This document defines what the Dev Tool CLI expects from the llama-orch backend to enable a high‑performance, multi‑agent, multi‑GPU autonomous development loop (Spec → Contract → Tests → Code, continuously).

The CLI is a frontend. It orchestrates agents and consumes backend capabilities; it does not run model inference itself.

## High-level Responsibilities (CLI)

- Drive continuous loops that plan, write, and refine:
  - Specs (RFC‑2119 with stable IDs)
  - Contracts (OpenAPI/JSON Schema/Pact)
  - Tests (BDD/property/consumer/provider)
  - Code (proposed diffs) and verification runs
- Run many agents concurrently, saturating backend GPU capacity.
- Persist and index artifacts (plans, summaries, diffs, test results) for traceability.

## Expectations From llama-orch (Backend)

Each expectation below is RFC‑2119 and references existing or proposed APIs. Where possible, these map directly to `contracts/openapi/*.yaml` today.

### 1) Task Admission & Streaming (Data Plane)

- OC-CLI-2001 (MUST): Queue API to submit tasks (fields in `contracts/openapi/data.yaml`):
  - `task_id`, `session_id`, `workload`, `model_ref`, `engine`, `ctx`, `priority`, `seed`, `determinism`, `sampler_profile_version`, `prompt`/`inputs`, `max_tokens`, `deadline_ms`, optional `kv_hint`.
- OC-CLI-2002 (MUST): SSE streaming events: `started`, `token`, `metrics`, `end`, `error` (`GET /v1/tasks/{id}/stream`).
- OC-CLI-2003 (MUST): Backpressure semantics on `POST /v1/tasks`: HTTP 429 with `Retry-After` and `X-Backoff-Ms`, plus `policy_label` in body.
- OC-CLI-2004 (SHOULD): Session management (`GET/DELETE /v1/sessions/{id}`) for KV reuse and context lifetimes.
- OC-CLI-2005 (SHOULD): Admission response with `predicted_start_ms` and `queue_position` for scheduling planners.
- OC-CLI-2006 (MUST): Determinism knobs honored (`seed`, `determinism`), enabling reproducibility for gated merges.

Mapping: Satisfied by `contracts/openapi/data.yaml`.

### 2) Control Plane for Fleet/Model Management

- OC-CLI-2101 (MUST): Catalog/trust policy (`/v1/catalog/models`, `verify`, `state`).
- OC-CLI-2102 (MUST): Pool drain/reload with deadlines; health probes (`/v1/pools/*`).
- OC-CLI-2103 (SHOULD): `GET /v1/replicasets` with load/SLO snapshots to inform agent scheduling.

Mapping: Satisfied by `contracts/openapi/control.yaml`.

### 3) Capability Discovery & Versioning

- OC-CLI-2201 (MUST): Discover engines, max context, rate limits, feature flags.
  - Option A: enrich `/v1/replicasets` with per‑engine limits.
  - Option B (new): `GET /v1/capabilities` returning limits and feature flags.
- OC-CLI-2202 (MUST): Stable API versions via OpenAPI `info.version`. CLI pins a compatible range.

### 4) Artifact Storage & Indexing

- OC-CLI-2301 (SHOULD): Artifact store API for agent outputs (plans/summaries/diffs/traces) with content‑addressed IDs, tags, lineage.
  - Proposed: `POST /v1/artifacts`, `GET /v1/artifacts/{id}`.
- OC-CLI-2302 (MAY): Git integration hooks for diffs and CI status.

Interim: CLI can write to local repo and use VCS/PR APIs directly.

### 5) Multi‑Agent Scheduling & Budgets

- OC-CLI-2401 (MUST): Scale to N concurrent tasks matched to GPU capacity; derive from `/v1/replicasets` + admission 429 backpressure.
- OC-CLI-2402 (SHOULD): Budget controls per session (token/time/cost) to enforce guardrails.
- OC-CLI-2403 (MAY): Priority classes (`interactive|batch`) mapped to queue priorities for agent roles.

### 6) Tool Access & Internet I/O

- OC-CLI-2501 (SHOULD): Safe HTTP fetch for external docs (proxied or policy‑guarded for auditability).
- OC-CLI-2502 (MAY): Web search and headless browse tools as pluggable adapters with caching.
- OC-CLI-2503 (MUST): Policy hooks to restrict outbound domains and redact sensitive data (policy host/sdk).

### 7) Contracts & Schemas

- OC-CLI-2601 (MUST): OpenAPI specs for data/control planes present (`contracts/openapi/*.yaml`).
- OC-CLI-2602 (MUST): JSON‑schema backed API types as a crate (`contracts/api-types`).
- OC-CLI-2603 (SHOULD): Pact files (`contracts/pacts/*`) for consumer expectations; CLI can generate/verify.

### 8) Observability, Metrics, Tracing

- OC-CLI-2701 (MUST): `X-Correlation-Id` echo for request/response tracing.
- OC-CLI-2702 (SHOULD): OpenTelemetry traces and Prometheus metrics (see `.specs/metrics/otel-prom.md`).
- OC-CLI-2703 (SHOULD): SSE `metrics` frames (`on_time_probability`, `queue_depth`, `kv_warmth`, …) for adaptive planning.

### 9) Error Taxonomy & Recovery

- OC-CLI-2801 (MUST): Standard error envelopes per data plane (`code`, `message`, `engine`, etc.) for 400/429/500/503.
- OC-CLI-2802 (SHOULD): Retries/backoff respecting 429 headers.
- OC-CLI-2803 (MAY): Advisory `policy_label` to explain throttling/rejection.

### 10) Determinism & Reproducibility

- OC-CLI-2901 (MUST): Support `seed` and deterministic decoding; CLI records seeds and sampler profiles.
- OC-CLI-2902 (SHOULD): Stable `sampler_profile_version` naming and compatibility notes.

### 11) Security, Identity, Policy

- OC-CLI-3001 (MUST): Practical AuthN/Z (tokens/mTLS/OIDC) with local dev mode.
- OC-CLI-3002 (MUST): Policy integration via policy host/sdk for tools and data flows.
- OC-CLI-3003 (SHOULD): Redaction/secret-handling guidance for logs and artifacts.

### 12) Test Harness Integration

- OC-CLI-3101 (SHOULD): Stable BDD runner interface (binary `bdd-runner`, env `LLORCH_BDD_FEATURE_PATH`) for targeted gates.
- OC-CLI-3102 (SHOULD): Provider verification tests aligned with OpenAPI/Pact to prevent drift.

### 13) Performance & Scale Expectations

- OC-CLI-3201 (MUST): Predictable queueing and horizontal scaling across GPUs/replicas.
- OC-CLI-3202 (SHOULD): Admission prediction (`predicted_start_ms`) accuracy that remains useful under load.
- OC-CLI-3203 (MAY): Pool selection by `engine`/`model_ref` labels for specialized agents (summarization vs coding).

## CLI Command Surface (Conceptual)

Examples (non‑binding):

- `llama-orch plan` — Generate/update plan artifacts from specs.
- `llama-orch contract sync` — Update Pact(s) from OpenAPI and CLI needs.
- `llama-orch tests plan` — Propose BDD/property tests from specs and gaps.
- `llama-orch impl apply` — Propose diffs and run checks; open a PR.
- `llama-orch loop run --agents auto` — Run multi‑agent loop with budget/gates, saturating GPUs via backend.
- `llama-orch status` — Show pool capacity, queue predictions, agent progress.

## References

- `contracts/openapi/data.yaml`
- `contracts/openapi/control.yaml`
- `contracts/api-types/`
- `orchestratord/tests/provider_verify.rs`
- `.specs/metrics/otel-prom.md`
- `test-harness/bdd/`
