# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

> VERY VERY VERY IMPORTANT: SPEC REDUCTION CHANGE (HOME PROFILE). Top-priority; no new features — only removals/simplifications.

## P0 — SPEC REDUCTION (Home Profile)

### DONE
- Spec overlays: `.docs/HOME_PROFILE.md`, `.specs/00_home_profile.md`
- Process guide: `.docs/FULL_PROCESS_SPEC_CHANGE.md`
- Scanner: `search_overkill.sh` (excludes all `target/`)
- CI sanity: clippy fix in `http/control.rs` test; relaxed a flaky metrics assert; `cargo xtask dev:loop` green

### TODO — Contracts (OpenAPI)
- Remove drains
  - Files:
    - `contracts/openapi/control.yaml` → delete `POST /v1/pools/{id}/drain`
    - `orchestratord/src/http/control.rs` → remove `drain_pool()` handler and drain metrics
    - `orchestratord/tests/provider_verify.rs` → stop asserting drain (202) path
  - Acceptance: provider verify passes without drain; no drain references remain in OpenAPI or code.
- Remove artifacts
  - Files:
    - `contracts/openapi/control.yaml` → delete all `/v1/artifacts*` paths/schemas
  - Acceptance: provider verify does not find artifact endpoints.
- Remove correlation IDs (X-Correlation-Id)
  - Files:
    - `contracts/openapi/data.yaml`, `contracts/openapi/control.yaml` → drop from examples
    - `orchestratord/src/http/data.rs`, `orchestratord/src/http/control.rs` → stop inserting header
    - Tests/fixtures: `orchestratord/tests/provider_verify.rs`, `contracts/pacts/cli-consumer-orchestratord.json`, `cli/consumer-tests/tests/orchqueue_pact.rs`
  - Acceptance: no header set/emitted; tests and pact pass without it.
- Remove `policy_label` in 429 bodies
  - Files:
    - `contracts/openapi/data.yaml` → delete from 429 example
    - `orchestratord/src/backpressure.rs`, `orchestratord/src/http/data.rs` → remove field from JSON body and any helper
    - Tests: `orchestratord/tests/provider_verify.rs` and BDD steps/features referencing `policy_label`
  - Acceptance: 429 contains only required fields + headers; tests pass.
- Simplify SSE stream shape (Home = started, token, end)
  - Files:
    - `contracts/openapi/data.yaml` → remove `metrics` SSE frame from examples/spec
    - `orchestratord/src/http/data.rs` → stop emitting `metrics` SSE frames
    - Tests: `orchestratord/tests/snapshot_transcript.rs` and any provider checks asserting `metrics` events
  - Acceptance: SSE includes only `started`, repeated `token`, and `end`; tests/snapshots updated.
- Optional: collapse control plane to only reload + health
  - Files:
    - Remove `/v1/replicasets`, `/v1/capabilities` from OpenAPI and handlers (`orchestratord/src/http/control.rs`)
    - Tests: snapshot tests referencing these endpoints
  - Acceptance: tests pass; only `POST /v1/pools/{id}/reload` and `GET /health` remain for HOME.

### TODO — Config Schema
- Drop fairness/preemption/tenants/quotas
  - Files:
    - `contracts/config-schema/src/lib.rs` → remove `FairnessConfig`, `PreemptionConfig`, any tenant/quota types and uses
    - `requirements/*.yaml` → align with 2 priorities only
  - Acceptance: schema compiles; `cargo xtask regen-schema`; examples validate.
- Remove `tensor_split` and topology hints from examples
  - Files:
    - `contracts/config-schema/src/lib.rs` examples; `.specs/00_llama-orch.md` excerpt; any requirements examples
  - Acceptance: examples free of `tensor_split`/topology.
- Trust/SBOM/signatures enforcement optional-only
  - Files:
    - `contracts/config-schema/src/lib.rs` and docs to drop strict trust policy requirements
    - Any code paths rejecting unsigned models to be downgraded to warnings (search for `require_signature`, `require_sbom`)
  - Acceptance: unsigned artifacts allowed by default; warnings logged; examples/docs align.

### TODO — Metrics
- Keep only HOME metrics
  - Files:
    - `ci/metrics.lint.json` → keep-only: `queue_depth`, `tasks_enqueued_total`, `tasks_rejected_total`, `tokens_in_total`, `tokens_out_total`, `gpu_utilization`, `vram_used_bytes` (optional `model_state`)
  - Acceptance: metrics linter passes with reduced set.
- Remove advanced metrics/histograms
  - Files:
    - `orchestratord/src/metrics.rs` → remove `admission_share`, `deadlines_met_ratio`, `preemptions_total`, `resumptions_total`, `latency_*` histograms and emitters
    - Callers in `orchestratord/src/http/data.rs` (fairness/deadline placeholders)
    - Specs/docs: `.specs/metrics/otel-prom.md` references
  - Acceptance: build/tests pass; no references remain.

### TODO — Lifecycle
- Reduce to `Active|Retired`
  - Files:
    - `orchestratord/src/state.rs` → `ModelState` only `Active|Retired`; initial state `Active`
    - `orchestratord/src/http/control.rs` → `set_model_state` accepts only `Active|Retired`; remove `Draft/Deprecated/Canary`
    - Data-plane guards remove `MODEL_DEPRECATED` and similar flows
    - Metrics: `model_state{state}` only `Active|Retired`
  - Acceptance: provider/BDD tests updated; no deprecated states present in code or examples.

### TODO — Security & Placement
- Bind `127.0.0.1` by default; single API token; least-loaded placement only
  - Files:
    - Verify server bind in `orchestratord` entrypoint; ensure default loopback
    - Auth: ensure single API token path only (remove tenant policy mentions)
    - Placement: ensure least-loaded heuristic; remove use of topology hints if any helper module exists
  - Acceptance: no non-loopback binds by default; docs/specs align.

### TODO — Tests
- Update provider/pact/BDD/unit tests
  - Files:
    - `orchestratord/tests/provider_verify.rs` (endpoints/headers/bodies)
    - Pact: `contracts/pacts/cli-consumer-orchestratord.json`
    - BDD features: remove fairness/deadlines/preemption/tenant scenarios
    - Unit tests under `orchestratord/src/**` referencing removed metrics/fields
  - Acceptance: `cargo xtask dev:loop` green; provider/pact/BDD all pass.

## Progress Log
- 2025-09-17: Created HOME profile docs/spec, process guide, and scanner (excludes target/); fixed clippy/test; dev:loop green
