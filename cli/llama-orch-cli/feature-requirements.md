# llama-orch-cli → llama-orch Feature Requirements

Purpose: Define what the Dev Tool CLI (auto programmer) needs from the llama‑orch backend to deliver a fast, multi‑agent, multi‑GPU Spec → Contract → Tests → Code loop. These are implementation‑agnostic, expressed with RFC‑2119 language.

## Summary of Capabilities

- Task admission and streaming with backpressure (data plane)
- Fleet and model lifecycle control (control plane)
- Capability discovery and versioning
- Artifact storage/indexing for agent outputs
- Multi‑agent scheduling, budgets, and priorities
- Internet tool access behind policy
- Stable contracts and schemas (OpenAPI/JSON‑Schema/Pact)
- Observability: correlation IDs, metrics, tracing
- Error taxonomy, retries, and recovery
- Determinism and reproducibility
- Security, identity, and policy enforcement
- Test harness integration
- Performance and scale SLOs

## Requirements (RFC‑2119)

1) Data Plane — Tasking

- FR‑DP‑001 (MUST): `POST /v1/tasks` accepts: `task_id`, `session_id`, `workload`, `model_ref`, `engine`, `ctx`, `priority`, `seed`, `determinism`, `sampler_profile_version`, `prompt|inputs`, `max_tokens`, `deadline_ms`, optional `kv_hint`.
- FR‑DP‑002 (MUST): `GET /v1/tasks/{id}/stream` emits SSE events: `started`, `token`, `metrics`, `end`, `error`.
- FR‑DP‑003 (MUST): Backpressure: `429` with headers `Retry-After` (seconds) and `X-Backoff-Ms` (ms) and advisory `policy_label` in body.
- FR‑DP‑004 (SHOULD): Sessions: `GET/DELETE /v1/sessions/{id}` for KV reuse and eviction.
- FR‑DP‑005 (SHOULD): Admission response includes `queue_position` and `predicted_start_ms`.

2) Control Plane — Fleet & Models

- FR‑CP‑001 (MUST): Model catalog with trust policy: `POST /v1/catalog/models`, `POST /v1/catalog/models/{id}/verify`, `POST /v1/catalog/models/{id}/state`.
- FR‑CP‑002 (MUST): Pools: `POST /v1/pools/{id}/drain|reload`, `GET /v1/pools/{id}/health`.
- FR‑CP‑003 (SHOULD): `GET /v1/replicasets` with engine, capacity, load, SLO snapshots.

3) Discovery & Versioning

- FR‑DV‑001 (MUST): Capability discovery (engines, max context, rate limits, feature flags).
  - Option A: enrich `GET /v1/replicasets` with limits
  - Option B (new): `GET /v1/capabilities`
- FR‑DV‑002 (MUST): Stable API versions via OpenAPI `info.version`.

4) Artifacts

- FR‑AR‑001 (SHOULD): Artifact registry API for plans/summaries/diffs/traces with content‑addressed IDs, tags, lineage.
  - Proposed: `POST /v1/artifacts`, `GET /v1/artifacts/{id}`

5) Multi‑Agent & Budgets

- FR‑MA‑001 (MUST): Horizontal concurrency matched to GPU capacity derived from `replicasets` + 429 backpressure.
- FR‑MA‑002 (SHOULD): Per‑session budgets (token/time/cost) with enforcement.
- FR‑MA‑003 (MAY): Priorities mapped to queue (`interactive|batch`).

6) Tools & Policy

- FR‑TL‑001 (SHOULD): Safe HTTP fetch/search tools with policy guardrails and audit.
- FR‑TL‑002 (MUST): Policy host/sdk hooks to constrain outbound domains and redact secrets.

7) Contracts & Schemas

- FR‑CT‑001 (MUST): OpenAPI maintained for data/control planes in `contracts/openapi/*.yaml`.
- FR‑CT‑002 (MUST): API types in `contracts/api-types/` kept in sync.
- FR‑CT‑003 (SHOULD): Pact files under `contracts/pacts/` for consumer expectations.

8) Observability

- FR‑OB‑001 (MUST): `X-Correlation-Id` accepted and echoed in responses.
- FR‑OB‑002 (SHOULD): OpenTelemetry tracing and Prometheus metrics per `.specs/metrics/otel-prom.md`.
- FR‑OB‑003 (SHOULD): Rich `metrics` SSE frames (e.g., `on_time_probability`, `queue_depth`, `kv_warmth`).

9) Errors & Recovery

- FR‑ER‑001 (MUST): Standard error envelope (`code`, `message`, `engine`, etc.) across 400/429/500/503.
- FR‑ER‑002 (SHOULD): Retries/backoff aligned to headers.

10) Determinism

- FR‑DE‑001 (MUST): Honor `seed` and `determinism` for reproducibility; document guarantees.

11) Security & Identity

- FR‑SI‑001 (MUST): Practical AuthN/Z (tokens/mTLS/OIDC) with local dev mode.
- FR‑SI‑002 (MUST): Policy enforcement pathway for tool use and data flows.

12) Test Harness

- FR‑TH‑001 (SHOULD): Stable `bdd-runner` interface with `LLORCH_BDD_FEATURE_PATH`.
- FR‑TH‑002 (SHOULD): Provider verification tests remain aligned to OpenAPI/Pact.

13) Performance & Scale

- FR‑PF‑001 (MUST): Predictable queueing and scaling across GPUs/replicas.
- FR‑PF‑002 (SHOULD): Useful `predicted_start_ms` under load.

## Mapping to This Repository

- Data plane contract: `contracts/openapi/data.yaml`
- Control plane contract: `contracts/openapi/control.yaml`
- API types: `contracts/api-types/`
- Consumer Pacts: `contracts/pacts/`
- Provider verification: `orchestratord/tests/provider_verify.rs`
- BDD runner: `test-harness/bdd/` (binary `bdd-runner`)

## Gaps / Proposed Additions

- Capability discovery endpoint (`GET /v1/capabilities`) or enrich `GET /v1/replicasets`.
- Artifact registry endpoints (`/v1/artifacts`).
- Budgets per session and reporting.
- Tooling proxy (HTTP fetch/search) guarded by policy.

## Acceptance Criteria (MVP)

- A CLI can:
  - Submit tasks and stream results reliably with backpressure respected.
  - Derive safe concurrency from capacity/backpressure.
  - Reuse sessions and cancel tasks.
  - Identify capabilities and versions.
  - Trace requests via correlation IDs and basic metrics.

This document is the source of truth for the CLI→backend contract while the actual CLI remains a stub.
