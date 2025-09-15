# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [x] Docs vs requirements outputs — adopt per-spec YAML files under `requirements/*.yaml` (e.g., `00_llama-orch.yaml`) and update references in `.docs/workflow.md` accordingly. Rationale: simpler, deterministic artifacts per spec; aggregated `index.yaml` is unnecessary at this time.
- [x] Add minimal Grafana dashboards placeholders under `ci/dashboards/` to satisfy Stage 5 gate in `.docs/workflow.md`.
- [ ] Confirm pacts cover all endpoints and status codes for OrchQueue v1 and sessions. Review `cli/consumer-tests/tests/orchqueue_pact.rs` and ensure `contracts/pacts/*.json` exercise 202/400/429/503/500 where applicable; SSE transcript shape for `GET /v1/tasks/{id}/stream`; 204 for cancel and delete.
- [ ] Optional: implement `xtask` helpers `pact:verify` and `pact:publish` to streamline dev loop (non-blocking).

## Progress Log (what changed)

2025-09-15 23:38 — Updated workflow doc to reference per-spec requirement files. Edited `.docs/workflow.md` tool section to: `spec-extract/ → requirements/*.yaml` and Stage 8 to say `requirements/*.yaml` instead of `index.yaml`.

2025-09-15 23:38 — Added Grafana dashboard placeholders to `ci/dashboards/`:

- `orchqueue_admission.json`
- `replica_load_slo.json`

2025-09-16 01:19 — Added Dev Tool CLI study stub and docs

- Created `cli/llama-orch-cli` as a study stub: `src/main.rs` prints a message and points to design docs (no functional CLI yet).
- Authored `cli/llama-orch-cli/DEV_TOOL_SPEC.md` to capture the Dev Tool CLI expectations from backend (capabilities, artifacts, budgets, SSE signals, etc.).
- Authored `cli/llama-orch-cli/feature-requirements.md` as a focused CLI→backend checklist (RFC-2119).
- Updated `cli/llama-orch-cli/README.md` to clearly state stub status and link to the above documents.
- Rationale: capture and align on the backend interfaces the CLI needs before implementing any real commands.

2025-09-16 01:22 — Extended OpenAPI contracts for CLI needs

- `contracts/openapi/control.yaml`:
  - Added `GET /v1/capabilities` with `Capabilities` schema (engines, ctx_max, supported_workloads, rate_limits, features, api_version).
  - Added Artifact Registry endpoints: `POST /v1/artifacts` (→201 with `ArtifactRef`) and `GET /v1/artifacts/{id}` (→200 with `Artifact`).
  - Added component schemas: `Workload`, `Capabilities`, `ArtifactKind`, `ArtifactRef`, `Artifact`.
- `contracts/openapi/data.yaml`:
  - Extended `SSEMetrics` with `queue_depth`, `kv_warmth`, `tokens_budget_remaining`, `time_budget_remaining_ms`, `cost_budget_remaining` (additive, backward-compatible).
  - Wired budgets via optional response headers on 202 (POST /v1/tasks) and 200 (GET /v1/tasks/{id}/stream):
    - `X-Budget-Tokens-Remaining`, `X-Budget-Time-Remaining-Ms`, `X-Budget-Cost-Remaining`.
  - Added budget fields to `SessionInfo`.

2025-09-16 01:24 — SPEC updates to encode CLI-dependent capabilities

- `.specs/00_llama-orch.md`: added §§ 3.19–3.22 for Capabilities/Artifacts/Budgets/SSE metrics signals.
- `.specs/20-orchestratord.md`: added Capabilities & Discovery (OC-CTRL-2060/2061), Artifact Registry (OC-CTRL-2065/2066), Budgets & Guardrails (OC-CTRL-2068), SSE Metrics signals (OC-CTRL-2023), and response `X-Correlation-Id` guidance (OC-CTRL-2052).
- `.specs/10-orchestrator-core.md`: added budget enforcement/visibility (OC-CORE-1023/1024).
- `.specs/71-metrics-contract.md`: added SSE metrics signal fields (OC-METRICS-7110/7111).
- `.specs/50-plugins-policy-host.md`: added HTTP fetch/search proxy requirements (OC-POLICY-4030..4033) for safe internet tool access.
- `.specs/51-plugins-policy-sdk.md`: added SDK helpers/redaction for tool invocation (OC-POLICY-SDK-4111/4112).

2025-09-16 01:27 — Provider-side tests to keep contracts honest

- Updated `orchestratord/tests/provider_verify.rs`:
  - Assert `GET /v1/capabilities` (200), `POST /v1/artifacts` (201), and `GET /v1/artifacts/{id}` (200) are present in control-plane OpenAPI.
  - Assert data-plane `SSEMetrics` includes extended fields and budget headers are declared for `POST /v1/tasks` 202 and `GET /v1/tasks/{id}/stream` 200.

2025-09-16 01:29 — Validated OpenAPI and kept generated artifacts consistent

- Ran `cargo run -p xtask -- regen-openapi` to validate `contracts/openapi/data.yaml` and `control.yaml` and refresh deterministic generated artifacts.
- Note: Control-plane Rust bindings are template-driven; to expose new endpoints in generated code, extend `xtask` templates.
