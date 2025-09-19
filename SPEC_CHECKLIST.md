# SPEC_CHECKLIST.md — Comprehensive Spec Migration Checklist (2025-09-19)

Use this checklist to apply and verify all spec changes stemming from the approved proposals and policy decisions. It complements `SPEC_CHANGES_NEEDED.md` by providing actionable, box-by-box items to drive PRs to completion.

Legend:
- [ ] Pending
- [x] Done

## 0) Approvals (statuses to flip)

- [X] Set proposals to Accepted
  - [X] `/.specs/proposals/2025-09-19-human-narration-logging.md`
  - [X] `/.specs/proposals/2025-09-19-performance-streamlining.md`
  - [X] `/.specs/proposals/2025-09-19-adapter-host-and-http-util.md`
  - [X] `/.specs/proposals/2025-09-19-testing-ownership-and-scope.md`
- [X] Set `/.specs/11_min_auth_hooks.md` to “Status: Accepted” (spec seam; no runtime change yet)

## 1) Root `.specs/` edits

- [x] `/.specs/00_llama-orch.md`
  - [x] Add Observability/Narration (ORCH‑33xx): emission points (admission, placement, stream, cancel), redaction, proof bundle expectations
  - [x] Security & Policy: reference `/.specs/11_min_auth_hooks.md` (AUTH‑1001..1008) and loopback posture
  - [x] Affirm GPU-only policy (no CPU inference paths) and engine auto‑provisioning alignment in §2.12
- [x] `/.specs/10-orchestrator-core.md`
  - [x] Introduce canonical `ModelRequirements` in Data Types
  - [x] Define derivation rules from catalog + adapter metadata
  - [x] Clarify placement feasibility inputs and tie‑break determinism mapping into `PlacementInput`
- [x] `/.specs/20-orchestratord.md`
  - [x] Make HTTP/2 for SSE (fallback to HTTP/1.1) normative
  - [x] Buffered SSE emitter and optional micro‑batch flag (default off; bounded) normative
  - [x] Keep `started → token* → end` ordering; heartbeat optional and compatible
  - [x] Add narration hooks and correlation ID propagation references
  - [x] Remove overlapping items from Refinements after promotion
- [x] `/.specs/25-catalog-core.md`
  - [x] Promote `exists(id|ref)` and `locate(ModelRef)` to Requirements and cross‑link crate contracts
- [x] `/.specs/30-pool-managerd.md`
  - [x] Verify “CPU inference spillover is disallowed” is present and consistent with GPU‑only policy
- [x] `/.specs/35-worker-adapters.md`
  - [x] Promote shared HTTP util usage across HTTP adapters (timeouts, retries w/ jitter, error taxonomy mapping, redaction)
  - [x] Require deterministic, low‑alloc streaming decode path
  - [x] Reference `adapter-host` and `worker-adapters/http-util`
- [x] `/.specs/40-worker-adapters-llamacpp-http.md`
  - [x] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [x] `/.specs/41-worker-adapters-vllm-http.md`
  - [x] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [x] `/.specs/42-worker-adapters-tgi-http.md`
  - [x] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [x] `/.specs/43-worker-adapters-triton.md`
  - [x] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [x] `/.specs/44-worker-adapters-openai-http.md`
  - [x] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [x] `/.specs/50-engine-provisioner.md`
  - [x] Keep GPU‑only language explicit (fail fast if CUDA/GPU unavailable; GPU is required)
  - [x] Ensure PreparedEngine summary requirement reflected (engine name/version, build ref, digest, flags, mode, binary path)
- [x] `/.specs/55-model-provisioner.md`
  - [x] Reference catalog helpers `exists(id|ref)` and `locate(ModelRef)` as normative fast‑paths
- [x] `/.specs/56-engine-catalog.md`
  - [x] Confirm no conflicts; reference EngineEntry usage if needed (informative)
- [x] `/.specs/60-config-schema.md`
  - [x] Add AUTH keys: `BIND_ADDR`/`ORCHD_ADDR`, `AUTH_TOKEN`, `AUTH_OPTIONAL`, `TRUST_PROXY_AUTH`
  - [x] Add `x-examples` for typical auth configurations (loopback optional, non‑loopback required)
- [x] `/.specs/70-determinism-suite.md`
  - [x] Clarify boundaries and keep determinism focus unchanged
- [x] `/.specs/71-metrics-contract.md` & `/.specs/metrics/otel-prom.md`
  - [x] If latency bucket guidance for SSE is needed, propose updates here first (optional)
- [x] `/.specs/72-bdd-harness.md`
  - [x] Clarify test ownership & scope (crate‑local vs cross‑crate)
  - [x] Include positive/negative Minimal Auth scenarios in BDD outline
- [x] `/.specs/00_home_profile.md`
  - [x] Cross‑link `/.specs/11_min_auth_hooks.md`; recommend loopback posture & optional token for LAN
  - [x] Confirm GPU‑only stance (no CPU inference) remains consistent

## 2) Crate `.specs/` edits

### Per-crate documentation deliverables (runtime crates only)
- [x] For each runtime crate, ensure a local `.specs/` exists (create if missing)
- [x] Add `30_TESTING.md` (umbrella) that links to per-layer test docs below and summarizes scope per ORCH‑3200..3208
- [x] Add per-layer testing docs (create only those that apply; others may be stubs referencing root harness):
  - [x] `31_UNIT.md` — unit tests owned by the crate
  - [x] `32_PROPERTY.md` — property tests (if applicable)
  - [x] `33_INTEGRATION.md` — narrow integration tests within crate boundary
  - [x] `34_CONTRACT.md` — contract/provider tests (reference OpenAPI/config schema if applicable)
  - [x] `35_BDD.md` — cross‑crate flows delegated to root harness; document boundaries and references
  - [x] `36_DETERMINISM.md` — determinism expectations/applicability for this crate
  - [x] `37_METRICS.md` — metric names/labels emitted by the crate; reference `.specs/metrics/otel-prom.md`
- [x] Add `40_ERROR_MESSAGING.md` — enumerate error types/envelopes this crate emits, mapping to codes/statuses, with examples and remediation hints

#### Documentation structure & QA
- [x] Each new doc MUST include front‑matter: Title (H1), Status, Owner, Date
- [x] Each new doc MUST include a `## Refinement Opportunities` section
- [x] Do NOT mint new ORCH‑IDs in crate docs; reference root `/.specs/00_llama-orch.md` IDs
- [x] Metrics naming in `37_METRICS.md` MUST align with `/.specs/metrics/otel-prom.md`; do not invent names/labels
- [x] `40_ERROR_MESSAGING.md` MUST map to canonical error envelopes/codes used by orchestrator; include example payloads and redaction notes

### orchestrator-core/.specs/
 - [x] `00_orchestrator_core.md`
  - [x] Reference canonical `ModelRequirements` (root) and how core consumes it
 - [x] `10_contracts.md`
  - [x] Add “Test Ownership” note (crate‑local vs BDD cross‑crate)
 - [x] `11_pool_managerd.md`
  - [x] Reference `PoolSnapshot` shape; remove duplication once canonicalized
 - [x] `12_catalog_core.md`
  - [x] Reference `ModelRequirements` derivation (no local duplication)
 - [x] `13_engine_provisioner.md`
  - [x] Remove any references to “CPU‑only fallback”
 - [x] `14_model_provisioner.md`
  - [x] Reference catalog helpers and `ModelRequirements`
 - [x] `15_worker_adapters.md`
  - [x] Keep minimal; ensure references to shared definitions
 - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `32_PROPERTY.md`, `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### orchestratord/.specs/
 - [x] `00_orchestratord.md`
  - [x] Document Bearer seam & identity breadcrumbs; reference Minimal Auth Hooks
 - [x] `10_orchestratord_v2_architecture.md`
  - [x] Document HTTP/2 preference, buffered SSE emitter, micro‑batch flag, CPU budget expectations, narration hooks
 - [x] `20_contracts.md`
  - [x] Map error taxonomy & headers; add correlation ID, auth error envelopes; add Test Ownership note
 - [x] `21_orchestrator_core.md`
  - [x] Reference canonical `ModelRequirements` & snapshot mapping
 - [x] `22_worker_adapters.md`
  - [x] Reference Adapter Host facade adoption; shared HTTP util norms
 - [x] Add docs: `40_TESTING.md`, `41_UNIT.md`, `43_INTEGRATION.md`, `44_CONTRACT.md`, `45_BDD.md` (delegation), `47_METRICS.md`, `50_ERROR_MESSAGING.md`

### catalog-core/.specs/
- [x] `00_catalog_core.md`
  - [x] Align with ORCH‑ID policy (root‑normative). Remove local ORCH‑IDs or remap to root entries
- [x] `10_contracts.md`
  - [x] Promote `exists(id|ref)` & `locate(ModelRef)` to Requirements
- [x] `11_model_provisioner.md`
  - [x] Reference helpers and error propagation
- [x] `12_engine_provisioner.md`
  - [x] Reference helpers & error propagation
- [x] `13_orchestrator_core.md`
  - [x] Reference canonical `ModelRequirements`
- [x] `14_pool_managerd.md`
  - [x] Confirm readiness & error logging alignment
 - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### pool-managerd/.specs/
 - [x] `00_pool_managerd.md`
  - [x] Attach `Authorization: Bearer` on registration/health calls to orchestrator when configured
  - [x] Keep GPU‑only readiness expectations
 - [x] `10_contracts.md`
  - [x] Add Test Ownership note
 - [x] `11_model_provisioner.md`, `12_engine_provisioner.md`, `13_orchestrator_core.md`, `14_catalog_core.md`
  - [x] Reference canonical types; avoid duplication
 - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `35_BDD.md` (delegation), `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### provisioners/engine-provisioner/.specs/
 - [x] `00_engine_provisioner.md`
  - [x] Remove CPU fallback statement (no forcing `-DGGML_CUDA=OFF`); fail fast with actionable diagnostics; GPU‑only
  - [x] Ensure PreparedEngine summary requirement is explicit
 - [x] `10_contracts.md`
  - [x] Remove “CUDA→CPU fallback” from Testing Expectations
 - [x] `11_model_provisioner.md`
  - [x] Reference catalog helpers & error propagation
 - [x] `12_pool_managerd.md`
  - [x] Remove CPU‑only normalization examples (e.g., `--n-gpu-layers 0`)
 - [x] `13_catalog_core.md`
  - [x] Reference helpers & error propagation
 - [x] `14_orchestrator_core.md`
  - [x] Remove “fallback signals (e.g., CPU‑only)”
 - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `32_PROPERTY.md` (flags normalization), `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### provisioners/model-provisioner/.specs/
 - [x] `00_model_provisioner.md`
  - [x] Reference catalog helpers; fast‑path exists/locate behavior
 - [x] `10_contracts.md`
  - [x] Add Test Ownership note
 - [x] `11_catalog_core.md`, `12_engine_provisioner.md`, `13_pool_managerd.md`, `14_orchestrator_core.md`
  - [x] Reference helpers & canonical types
 - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### worker-adapters/.specs/
 - [x] `00_worker_adapters.md`
  - [x] Promote shared HTTP util norms
 - [x] `10_contracts.md`
  - [x] Add Test Ownership note
 - [x] `11_orchestratord.md`
  - [x] When configured, attach `Authorization: Bearer` on calls to orchestrator
 - [x] `12_pool_managerd.md`
  - [x] Keep references minimal; ensure no contradictions
 - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `34_CONTRACT.md` (if applicable), `35_BDD.md` (delegation), `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### Adapter crates (per engine) — if/when present locally
- [x] For each adapter crate (e.g., `worker-adapters/llamacpp-http`, `worker-adapters/mock`, future `vllm-http`, `tgi-http`, `triton`):
  - [x] Ensure local `.specs/` exists (create if missing)
  - [x] Add `30_TESTING.md` (umbrella) and per-layer docs as applicable (`31_UNIT.md`, `33_INTEGRATION.md`, `34_CONTRACT.md` if exposing external APIs, `37_METRICS.md`)
  - [x] Add `40_ERROR_MESSAGING.md` with HTTP mapping and redaction rules; align error taxonomy with orchestrator envelopes

### Observability and shared runtime libs
- [x] `observability/narration-core/.specs/`
  - [x] Create `.specs/` if missing
  - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (if any crate-boundary tests), `37_METRICS.md` (fields surfaced), `40_ERROR_MESSAGING.md`
- [x] `adapter-host/.specs/`
  - [x] Create `.specs/` if missing
  - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (facade/registry), `37_METRICS.md` (narration/metrics wrappers), `40_ERROR_MESSAGING.md`
- [x] `worker-adapters/http-util/.specs/`
  - [x] Create `.specs/` if missing
  - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (client init/retry), `34_CONTRACT.md` (if any provider verify applies), `37_METRICS.md` (if emitting), `40_ERROR_MESSAGING.md`

### CLI crates
- [x] `cli/llama-orch-cli/.specs/`
  - [x] Create `.specs/` if missing
  - [x] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (against mock endpoints), `34_CONTRACT.md` (if verifying OpenAPI client usage), `40_ERROR_MESSAGING.md` (CLI error taxonomy, exit codes)
- [x] `cli/consumer-tests/.specs/` (if considered part of runtime deliverables)
  - [x] Create `.specs/` if missing
  - [x] Add docs: `30_TESTING.md`, per-layer stubs as applicable; `40_ERROR_MESSAGING.md` optional (document tooling errors)

### test-harness/.specs/
- [ ] `test-harness/bdd/.specs/00_test_harness.md`
  - [ ] Include Minimal Auth positive/negative scenarios
- [ ] `test-harness/bdd/.specs/11_orchestratord.md`, `12_worker_adapters.md`, `13_pool_managerd.md`
  - [ ] Align steps with new seams; avoid crate‑local behavior leakage
- [ ] `test-harness/e2e-haiku/.specs/*`
  - [ ] No changes expected; sanity‑check references
- [ ] `test-harness/metrics-contract/.specs/00_test_harness.md`
  - [ ] No changes expected; sanity‑check lints

## 3) New spec files to create

- [x] `observability/narration-core/.specs/00_narration_core.md` (facade API, fields, redaction, test capture)
- [x] `adapter-host/.specs/00_adapter_host.md` (registry/facade, narration/metrics wrappers, correlation)
- [x] `worker-adapters/http-util/.specs/00_http_util.md` (shared HTTP client, retries/backoff/HTTP2, streaming decode helpers, redaction)

## 4) Clean-up: GPU-only (remove CPU fallback mentions)

- [x] Search & scrub CPU fallback language across specs
  - [x] `provisioners/engine-provisioner/.specs/00_engine_provisioner.md`
  - [x] `provisioners/engine-provisioner/.specs/10_contracts.md`
  - [x] `provisioners/engine-provisioner/.specs/12_pool_managerd.md`
  - [x] `orchestrator-core/.specs/13_engine_provisioner.md`
  - [x] `provisioners/engine-provisioner/.specs/14_orchestrator_core.md`

## 5) ORCH‑ID policy normalization

- [x] Keep ORCH‑IDs normative in `/.specs/00_llama-orch.md`
- [x] In crate specs, reference root ORCH‑IDs (do not mint new ones)
- [x] Migrate or delete local ORCH‑IDs in `catalog-core/.specs/00_catalog_core.md`

## 6) Verification & proof bundles

- [ ] Update `.docs/testing/` to reference narration coverage and SSE micro‑batch proof artifacts
- [ ] Ensure proof bundles include SSE transcripts, narration coverage excerpts, determinism outputs, metrics lints, EngineEntry snapshots
- [ ] Run verification locally:
  - [ ] `cargo fmt --all -- --check`
  - [ ] `cargo clippy --all-targets --all-features -- -D warnings`
  - [ ] `cargo test --workspace --all-features -- --nocapture`
  - [ ] `cargo xtask regen-openapi`
  - [ ] `cargo xtask regen-schema`
  - [ ] `cargo run -p tools-spec-extract --quiet`
  - [ ] `cargo xtask dev:loop`

## 7) Post‑spec follow‑ups (docs/READMEs)

- [ ] Propagate High/Mid/Low behavior sections across crate READMEs (code‑grounded) after specs settle
- [ ] SECURITY.md: update text to reflect Minimal Auth Hooks acceptance (still documentation‑only seam)

---

Owner: @llama-orch-maintainers
Date: 2025-09-19
