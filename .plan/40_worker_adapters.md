# Worker Adapters — Implementation Plan (OC-ADAPT-5xxx)

Specs: `.specs/40-worker-adapters-llamacpp-http.md`, `41-*.md`, `42-*.md`, `43-*.md`
Scope: engine-specific adapter mapping, health/properties, completion SSE, cancel, metrics, determinism normalization, version capture.

## Stages and Deliverables

- Stage 2 — Adapter Integration
  - Implement adapters for llama.cpp, vLLM, TGI, Triton/TRT-LLM.
  - Map engine-native endpoints to internal worker contract; do not expose OpenAI-compat publicly.
  - Capture `engine_version`, `model_digest` (and `trtllm_version` where applicable).
  - Normalize detokenization templates and sampler profiles to keep per-replica determinism.

- Stage 5 — Observability
  - Expose per-adapter metrics and health.

## Tests

- Unit/integration tests under each adapter crate `worker-adapters/*/tests/`.
- BDD adapter features: `test-harness/bdd/tests/features/adapters/`.
- Determinism suite per engine (settings, single-slot modes).

## Acceptance Criteria

- OC-ADAPT IDs mapped to tests; adapters pass integration and determinism requirements; version labels present in metrics/logging.

## Backlog (initial)

- HTTP clients and endpoint mappers per engine.
- SSE token stream bridge, cancel mapping.
- Health/properties and slot accounting for continuous batching engines.
- Detokenization/template normalization modules.
