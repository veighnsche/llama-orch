# TODO — Pre‑Contract‑Freeze Analysis Tracker

This is the single active TODO tracker. Perform these analyses in order before declaring a Contract Freeze (Stage 0 in `.docs/workflow.md`). Keep entries factual with proofs and links to SPEC IDs.

## P0 — MUST: Pre‑Freeze Analyses (in order)

- [ ] UX/DX gaps likely needed pre‑1.0 (analyze and propose before freeze)
  - API ergonomics
    - Ensure consistent field names/types across endpoints; add missing examples (`x-examples`) in OpenAPI for common requests and error bodies.
    - Error envelopes: consider adding `retriable` (bool) and `retry_after_ms` (int) hints; ensure `policy_label` appears where relevant; include a `correlation_id` in responses/logs.
    - Confirm headers vs body duplication is intentional (e.g., `Retry-After` header vs `backoff_ms` in body) and documented.
    - Proof: propose SPEC/OpenAPI deltas (PR), update CDC tests/examples, regenerate clients.
    - Preliminary findings (from current repo):
      - OpenAPI lacks request/response `x-examples` for key flows (enqueue, 429, cancel, sessions).
      - `429` uses `ErrorEnvelope` without a `policy_label` field, but `.specs/00_llama-orch.md §6.2` requires JSON to include the full policy label. Propose adding `policy_label` to `ErrorEnvelope` or a dedicated `BackpressureError`.
      - `ErrorEnvelope` has no `retriable`/`retry_after_ms`; consider adding to improve client UX.
      - No explicit `correlation_id` surfaced in responses; consider standard `X-Correlation-Id` header and echoing in error bodies/logs.
  - Artifacts
    - Investigation write‑up: `.docs/investigations/ux-dx-pre-freeze.md`
    - Proposal: `.specs/proposals/2025-09-15-ux-dx-improvements.md`
  - Contract changes to apply (pre‑freeze)
    - [ ] OpenAPI `x-examples` for enqueue/stream/cancel/sessions
    - [ ] 429 JSON body includes `policy_label` (in addition to headers)
    - [ ] `ErrorEnvelope` gains optional `retriable` and `retry_after_ms`
    - [ ] Transport: standardize `X-Correlation-Id` request/response behavior, document logging alignment
    - [ ] CDC: update consumer examples and provider verification for 429/correlation id
    - Proof:
      - `cargo xtask regen-openapi && git diff --exit-code`
      - `cargo test -p cli-consumer-tests -- --nocapture` and `cargo test -p orchestratord --test provider_verify -- --nocapture`
  - Developer tooling
    - `xtask` conveniences: `ci:haiku:gpu`, `dev:spin-up:<engine>` (CPU fallback), `regen:all` meta-task; ensure deterministic runners exist.
    - Local bootstrap: model cache script, sample `.env`, sample config per engine, one‑shot scripts to start a CPU worker behind the orchestrator.
    - Proof: add/verify `xtask` commands and docs; quickstart runs end‑to‑end locally.
  - Observability & debugging
    - Standardize log fields and tracing spans (request/stream IDs, engine/version, sampler profile); add log examples in docs.
    - Provide Grafana dashboards with sample data; ensure metrics label budgets enforced by linter.
    - Proof: dashboards render in CI; metrics-contract tests green.
  - Config UX
    - Ensure schema descriptions cover defaults and constraints; provide per‑engine config templates; add `cargo xtask validate-config <file>`.
    - Validation errors include JSON Pointer path and guidance.
    - Proof: try invalid samples; confirm helpful errors.
  - SDK/Client DX
    - Verify generated client ergonomics (types, streaming helpers, retries); add simple examples for enqueue/stream/cancel in Rust and curl.
    - Proof: `tools/openapi-client` trybuild/UI tests cover typical flows.
  - Documentation entry points
    - Link `README.md` → `README_LLM.md` and `.docs/workflow.md`; add quickstart for local CPU flow with constraints and caveats.
    - Proof: link checker green; quickstart steps succeed.

- [ ] SPEC IDs and traceability are complete and stable
  - Verify `ORCH-*` IDs in `.specs/00_llama-orch.md` and `OC-*` IDs in component specs (`.specs/10-*,20-*,30-*,40-*,50-*,60-*,70-*`).
  - Check for uniqueness, no reuse, and consistent RFC‑2119 language.
  - Ensure each `ORCH-*` is referenced by OpenAPI (via `x-req-id`) or an explicit rationale exists; ensure `requirements/*.yaml` contains all IDs.
  - Proof:
    - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
    - `rg -n "\b(ORCH|OC)-[A-Z0-9-]+\b" -- **/*.{md,rs}`

- [ ] OpenAPI coverage and SSE contract are authoritative
  - `contracts/openapi/data.yaml` and `control.yaml` annotated with `x-req-id: ORCH-*` on all public operations and relevant components.
  - SSE events defined as schemas (`SSEStarted`, `SSEToken`, `SSEMetrics`, `SSEEnd`, `SSEError`) and referenced under `x-sse-events`.
  - Error envelope (`ErrorEnvelope`, `ErrorKind`) typed and linked to `ORCH-2006`; 429 backpressure response linked to `ORCH-2007` with headers.
  - Proof:
    - `cargo xtask regen-openapi && git diff --exit-code`
    - `rg -n "x-req-id: ORCH-" contracts/openapi`

- [ ] SPEC v3.2 — Catalog/Lifecycle & Advanced Scheduling (contracts‑first)
  - Catalog & Trust
    - [x] Control plane catalog endpoints added: create/get/verify/state (`/v1/catalog/models*`) in `contracts/openapi/control.yaml`
    - [x] Metrics spec includes `catalog_verifications_total`; linter updated
    - [ ] Example catalog payloads and strict policy doc (refs, mirrors, CA roots)
  - Lifecycle
    - [x] Lifecycle states defined in SPEC addendum; control op to set state with optional drain deadline
    - [x] Typed error `MODEL_DEPRECATED` added
    - [ ] Example state transition flow and deprecation behavior documented
  - Advanced Scheduling
    - [x] Config schema extended: WFQ with weights, tenants/quotas, preemption block
    - [x] Data plane SSE metrics include `on_time_probability`
    - [x] Typed errors include `DEADLINE_UNMET`, `UNTRUSTED_ARTIFACT`
    - [x] Metrics spec adds `admission_share`, `deadlines_met_ratio`, `preemptions_total`, `resumptions_total`; linter updated
    - [ ] Add example configs showing default weights and tenant quotas; document EDF/feasibility policy
  - SPEC integration
    - [x] v3.2 Addendum added to `.specs/00_llama-orch.md` with ORCH IDs, traceability, checklist
  - Tests (to add)
    - [ ] Config schema example validates new fields (WFQ/tenants/preemption) — extend docs with examples
    - [ ] Metrics contract: sample emission for new metrics (placeholder or harness validation)
    - [ ] CDC: add DEADLINE_UNMET path and MODEL_DEPRECATED rejection example
  - Proof:
    - `cargo xtask regen-schema && git diff --exit-code`
    - `cargo test --workspace --all-features -- --nocapture`
    - `bash ci/scripts/spec_lint.sh && bash ci/scripts/check_links.sh`

- [ ] Config Schema is complete and generated deterministically
  - All config types defined in `contracts/config-schema/` with `schemars` derivations; unknown field policy defined; required vs optional documented.
  - JSON Schema regenerated deterministically; examples compile in docs if present.
  - Proof:
    - `cargo xtask regen-schema && git diff --exit-code`

- [ ] Metrics contract matches `.specs/metrics/otel-prom.md`
  - Metric names, units, and labels align; labels MUST include `engine`; engine‑specific version labels (e.g., `engine_version`, `trtllm_version`) present where applicable; exceptions documented (e.g., admission‑level counters may omit `engine_version`).
  - `ci/metrics.lint.json` agrees with the SPEC.
  - Proof:
    - `cargo test -p test-harness-metrics-contract -- --nocapture`

- [ ] CDC (Pact) coverage exists for OrchQueue v1
  - Consumer pacts cover: `POST /v1/tasks`, `GET /v1/tasks/{id}/stream`, `POST /v1/tasks/{id}/cancel`, session `GET`/`DELETE`.
  - Pact JSON files committed under `contracts/pacts/`.
  - Proof:
    - `cargo test -p cli-consumer-tests -- --nocapture`

- [ ] Provider verification passes on real handlers
  - Orchestratord provider verification loads pact files and validates handlers against OpenAPI.
  - Proof:
    - `cargo test -p orchestratord --test provider_verify -- --nocapture`

- [ ] Determinism primitives and pinning are explicit
  - Defaults: `seed = hash(job_id)` when omitted; `sampler_profile_version = "v1"` pinned; `engine_version` pinned per replica set.
  - Adapter guidance for single‑slot/single‑request mode documented (e.g., llama.cpp `--parallel 1 --no-cont-batching`).
  - Proof:
    - `rg -n "sampler_profile_version|engine_version|--no-cont-batching|parallel 1" -- **/*`

- [ ] Engine adapter contract mapping is complete per engine
  - For each of: llama.cpp, vLLM, TGI, Triton: health/props/completion/cancel/metrics mapping documented and tested; any OpenAI‑compat surface is internal only.
  - Proof:
    - `rg -n "ORCH-305[4-8]" -- .specs/4*-worker-adapters-*.md`

- [ ] Security & tenancy are captured
  - Auth: API key day‑1; quotas for concurrent jobs, tokens/min, KV‑MB enforced pre‑admission (where applicable).
  - Proof: SPEC sections align; OpenAPI errors/types cover auth and quota failures.

- [ ] Observability: logs and metrics fields are sufficient
  - Logs include the required fields (see `ORCH-3027` list); metrics exported and align with SPEC; observability examples/dashboards under `ci/dashboards/` exist or have placeholders.
  - Proof:
    - `rg -n "ORCH-3027|ORCH-3028" -- **/*.md`

- [ ] Idempotent regenerators and docs lint
  - Regenerators are diff‑clean on second run; link checker passes; SPEC linter passes.
  - Proof:
    - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
    - `bash ci/scripts/check_links.sh`
    - `bash ci/scripts/spec_lint.sh`

- [ ] Backwards‑compat remnants removed (pre‑v1 no‑BC policy)
  - No legacy references like `orchestrator-spec` → `index.yaml` or `requirements/index.yaml` remain; shims/adapters for old names removed.
  - Proof:
    - `rg -n "orchestrator-spec|requirements/index.yaml" -- **/*`

- [ ] Documentation entry points
  - `README.md` links to `README_LLM.md` and `.docs/workflow.md`; quickstart points to contracts/specs.
  - Proof: manual review + link checker.

## P1 — SHOULD: Nice‑to‑haves before freeze

- [ ] CI job gating for TODO discipline (optional)
  - Fail PR if root `TODO.md` is missing or stale; allow release branches to archive via `ci/scripts/archive_todo.sh`.

- [ ] Dashboard sample data/render step (optional)
  - Provide sample metrics payload for CI to render Grafana dashboards.

## Proof Commands (quick)

```bash
cargo fmt --all -- --check && \
  cargo clippy --all-targets --all-features -- -D warnings && \
  cargo xtask regen-openapi && cargo xtask regen-schema && \
  cargo run -p tools-spec-extract --quiet && git diff --exit-code && \
  cargo test --workspace --all-features -- --nocapture && \
  bash ci/scripts/check_links.sh && \
  bash ci/scripts/spec_lint.sh
```

## Notes / Blockers

- [ ] …

## Progress Log (what changed)

- 2025-09-15: Created pre‑freeze analysis TODO.md aligned with `README_LLM.md` and `.docs/workflow.md`.
- 2025-09-15: Wrote investigation: `.docs/investigations/ux-dx-pre-freeze.md` (UX/DX gaps, proposed fields/headers, examples plan).
- 2025-09-15: Drafted spec proposal: `.specs/proposals/2025-09-15-ux-dx-improvements.md` (OpenAPI examples, 429 policy_label, advisory retry fields, correlation ID).
- 2025-09-15: Updated worker adapter implementation hints to document capabilities (multi‑GPU, SSE, cancel, metrics, pinning) for llama.cpp, vLLM, TGI, Triton; fixed fenced code block languages.
- 2025-09-15: Fixed broken link in `.docs/testing/VIBE_CHECK.md` (LICENSE) and re‑ran `check_links.sh` + `spec_lint.sh` (green).
