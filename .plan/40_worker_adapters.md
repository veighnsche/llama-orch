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

- Stage 8 — Product: Adapters conformance (aligns with README_LLM Stage 8)
  - SSE framing contract end-to-end (`started|token|metrics|end|error`) with byte boundaries preserved; backpressure signaling propagated.
  - Timeouts, retries, and typed error envelopes with engine context; map engine-specific errors to contract taxonomy.
  - Determinism normalization per engine (templates, seed handling, single-slot flags); verify within-replica-set byte-exact behavior.
  - Metrics: increment tokens_in/out, observe latency histograms; labels include `{engine, engine_version, pool_id, replica_id, priority}`.
  - Engines delivered in order: llama.cpp → vLLM → TGI → Triton/TRT-LLM.

## Tests

- Unit/integration tests under each adapter crate `worker-adapters/*/tests/`.
- BDD adapter features: `test-harness/bdd/tests/features/adapters/`.
- Determinism suite per engine (settings, single-slot modes).
- Orchestrator integration tests exercising adapters via `/v1/tasks` stream; assert SSE framing, backpressure, and metrics side-effects.

## Acceptance Criteria

- OC-ADAPT IDs mapped to tests; adapters pass integration and determinism requirements; version labels present in metrics/logging.
- SSE contract observed exactly; typed errors surfaced to orchestrator; timeouts/retries validated under fault injection.
- tokens_in/out counters and latency histograms incremented with correct label sets; scrape validates names/labels.

## Backlog (initial)

- HTTP clients and endpoint mappers per engine.
- SSE token stream bridge, cancel mapping.
- Health/properties and slot accounting for continuous batching engines.
- Detokenization/template normalization modules.

---

## DX Modularization Proposal (Adapters)

Goal: unify adapter contract and shared utilities to reduce duplication and speed development across engines.

Introduce a new shared crate:

- `worker-adapters/adapter-api/` (lib)
  - Defines `WorkerAdapter` trait: `submit()`, `cancel()`, `health()`, `properties()`; streaming via a typed `SseEvent` enum (`Started|Token|Metrics|End|Error`).
  - Shared helpers: SSE framing builder/parser, metrics label helpers, error taxonomy mapping to orchestrator envelopes, determinism normalization utilities (templates/seed handling).
  - No engine-specific deps; minimal `serde`, `thiserror`, `futures`.

Layering and dependency rules:

- Engine crates (`llamacpp-http`, `vllm-http`, `tgi-http`, `triton`, `mock`) implement `adapter-api`.
- Orchestrator consumes only `adapter-api` (trait) — never import engine crates directly.
- Pool-managerd depends on engine crates only for lifecycle, or calls via `adapter-api` if we centralize process control.

Rollout plan:

1) Stage 8.0: Draft `adapter-api` trait and helpers; migrate `mock` adapter first.
2) Stage 8.1: Port `llamacpp-http`; validate SSE, backpressure, errors, metrics.
3) Stage 8.2: Port remaining engines; factor any new shared logic back into `adapter-api`.
4) Guardrails: deny `pub` by default (`pub(crate)`), expose a minimal surface; unit tests live in `adapter-api` for helpers.

## Proposal (Accepted)

- Adopt shared `worker-adapters/adapter-api` crate providing the trait + shared helpers. Orchestrator depends only on this trait.
- Migrate adapters in order: mock → llama.cpp → vLLM → TGI → Triton, pushing common logic into `adapter-api`.
- Keep engines isolated; enforce typed errors and SSE contract; ensure determinism normalization.
