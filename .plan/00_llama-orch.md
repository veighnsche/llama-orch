# Umbrella Orchestrator — Implementation Plan (ORCH-*)

Spec: `.specs/00_llama-orch.md`
Scope: NVIDIA-only inference, queues/admission, replicas/placement, sessions/KV, cancellations/idempotency, observability, config/lifecycle, rollouts, security/tenancy, resilience, SLOs, storage, API contracts & determinism, policy plugins, testing & validation.

## Stages and Deliverables (mapped to `.docs/workflow.md`)

- Stage 0 — Contract Freeze
  - OpenAPI `{control.yaml,data.yaml}` with `x-req-id`, typed errors, SSE schemas, correlation ID; examples per UX/DX proposal.
  - Config schema types for pools, engines, devices, quotas, deadlines, preemption, tenants.
  - Metrics contract aligned with `.specs/metrics/otel-prom.md`.

- Stage 1 — CDC Consumer + Snapshots
  - Pact interactions for data-plane and sessions; CLI snapshots via `insta`.

- Stage 2 — Provider Verification
  - Verify typed errors, backpressure headers + bodies, SSE framing, admission checks.

- Stage 3 — Properties & Invariants
  - Queue invariants, least-loaded placement, device masks, session affinity, guardrails.

- Stage 4 — Determinism
  - Enforce replica-set pinning (`engine_version`, `sampler_profile_version`); byte-exact suite per engine; adapters normalize templates and sampler profiles.

- Stage 5 — Observability & SLOs
  - Logs fields and metrics presence; Grafana dashboards and alerts.

- Stage 6 — Real-Model E2E (Haiku)
  - End-to-end via OrchQueue v1 against real worker; anti-cheat and metrics delta.

- Stage 7 — Chaos & Load (nightly)
  - Driver resets, OOM handling, circuit breakers; short load SLO checks.

- Stage 8 — Compliance & Release
  - Regenerate `requirements/*.yaml`; `COMPLIANCE.md`; `CHANGELOG_SPEC.md`.

## Tests (selection)

- CDC: `cli/consumer-tests/` → `contracts/pacts/*.json`.
- Provider: `orchestratord/tests/provider_verify.rs`.
- Properties: `orchestrator-core/tests/props_queue.rs`.
- BDD: `test-harness/bdd/tests/features/{data_plane,scheduling,observability,security,policy,orchestrator_core,pool_manager,sse}/`.
- Determinism: `test-harness/determinism-suite/`.
- Metrics contract: `test-harness/metrics-contract/` + `ci/metrics.lint.json`.

## Acceptance Criteria (per ORCH requirement)

- Contract coverage; consumer pact exists; provider verify green.
- Properties pass; determinism suite pass per engine; logs+metrics conform; `requirements/00_llama-orch.yaml` links IDs → tests → code.

## Backlog (initial)

- OpenAPI authoring with examples and typed errors/correlation id.
- SSE event framing and payload structs.
- Admission: ctx/token budget validation; backpressure headers + policy label body; quotas and rate limits.
- Placement: Ready gating, device masks, heterogeneous split ratios; session affinity.
- Security: API key middleware; secret redaction in logs.
- Resilience: watchdog timeouts; CUDA/driver error transitions with backoff; circuit breakers.
- Observability: metrics/log fields and dashboards.

## Notes

- BDD runner: `test-harness/bdd` binary `bdd-runner`; set `LLORCH_BDD_FEATURE_PATH` to target `tests/features` subtrees.
