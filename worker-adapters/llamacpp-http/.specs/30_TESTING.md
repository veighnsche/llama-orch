# llamacpp-http — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Adapter behavior against llama.cpp HTTP server: request shaping, streaming decode, retries, error mapping.

## Test Catalog

- Unit
  - Request serialization: headers, query/body, sampler/seed fields
  - Error mapping: HTTP status/body → `WorkerError`; redaction of secrets

- Integration
  - End-to-end streaming against a local stub or real llama.cpp (feature-gated): happy path, mid-stream error, timeouts
  - Cancel propagation and disconnect
  - Retries with capped+jittered backoff via `http-util`

- Contract
  - Validate llama.cpp response shapes (delta/token events) the adapter relies on; pin API version

## Execution & Tooling

- Run: `cargo test -p worker-adapters-llamacpp-http -- --nocapture` (package name may differ)
- For real-engine tests, gate with a feature and document setup; default CI uses stubs only

## Traceability

- Error taxonomy: ORCH‑3330/3331
- SSE error frame (when surfaced via orchestrator): ORCH‑3406..3409
- Metrics: `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Golden SSE transcript fixtures for typical prompts.
