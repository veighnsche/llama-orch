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

- [ ] `/.specs/00_llama-orch.md`
  - [ ] Add Observability/Narration (ORCH‑33xx): emission points (admission, placement, stream, cancel), redaction, proof bundle expectations
  - [ ] Security & Policy: reference `/.specs/11_min_auth_hooks.md` (AUTH‑1001..1008) and loopback posture
  - [ ] Affirm GPU-only policy (no CPU inference paths) and engine auto‑provisioning alignment in §2.12
- [ ] `/.specs/10-orchestrator-core.md`
  - [ ] Introduce canonical `ModelRequirements` in Data Types
  - [ ] Define derivation rules from catalog + adapter metadata
  - [ ] Clarify placement feasibility inputs and tie‑break determinism mapping into `PlacementInput`
- [ ] `/.specs/20-orchestratord.md`
  - [ ] Make HTTP/2 for SSE (fallback to HTTP/1.1) normative
  - [ ] Buffered SSE emitter and optional micro‑batch flag (default off; bounded) normative
  - [ ] Keep `started → token* → end` ordering; heartbeat optional and compatible
  - [ ] Add narration hooks and correlation ID propagation references
  - [ ] Remove overlapping items from Refinements after promotion
- [ ] `/.specs/25-catalog-core.md`
  - [ ] Promote `exists(id|ref)` and `locate(ModelRef)` to Requirements and cross‑link crate contracts
- [ ] `/.specs/30-pool-managerd.md`
  - [ ] Verify “CPU inference spillover is disallowed” is present and consistent with GPU‑only policy
- [ ] `/.specs/35-worker-adapters.md`
  - [ ] Promote shared HTTP util usage across HTTP adapters (timeouts, retries w/ jitter, error taxonomy mapping, redaction)
  - [ ] Require deterministic, low‑alloc streaming decode path
  - [ ] Reference `adapter-host` and `worker-adapters/http-util`
- [ ] `/.specs/40-worker-adapters-llamacpp-http.md`
  - [ ] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [ ] `/.specs/41-worker-adapters-vllm-http.md`
  - [ ] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [ ] `/.specs/42-worker-adapters-tgi-http.md`
  - [ ] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [ ] `/.specs/43-worker-adapters-triton.md`
  - [ ] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [ ] `/.specs/44-worker-adapters-openai-http.md`
  - [ ] Reference `worker-adapters/http-util` + Adapter Host; confirm streaming order & redaction expectations
- [ ] `/.specs/50-engine-provisioner.md`
  - [ ] Keep GPU‑only language explicit (“fail fast if CUDA/GPU unavailable; no CPU fallback”)
  - [ ] Ensure PreparedEngine summary requirement reflected (engine name/version, build ref, digest, flags, mode, binary path)
- [ ] `/.specs/55-model-provisioner.md`
  - [ ] Reference catalog helpers `exists(id|ref)` and `locate(ModelRef)` as normative fast‑paths
- [ ] `/.specs/56-engine-catalog.md`
  - [ ] Confirm no conflicts; reference EngineEntry usage if needed (informative)
- [ ] `/.specs/60-config-schema.md`
  - [ ] Add AUTH keys: `BIND_ADDR`/`ORCHD_ADDR`, `AUTH_TOKEN`, `AUTH_OPTIONAL`, `TRUST_PROXY_AUTH`
  - [ ] Add `x-examples` for typical auth configurations (loopback optional, non‑loopback required)
- [ ] `/.specs/70-determinism-suite.md`
  - [ ] Clarify boundaries and keep determinism focus unchanged
- [ ] `/.specs/71-metrics-contract.md` & `/.specs/metrics/otel-prom.md`
  - [ ] If latency bucket guidance for SSE is needed, propose updates here first (optional)
- [ ] `/.specs/72-bdd-harness.md`
  - [ ] Clarify test ownership & scope (crate‑local vs cross‑crate)
  - [ ] Include positive/negative Minimal Auth scenarios in BDD outline
- [ ] `/.specs/00_home_profile.md`
  - [ ] Cross‑link `/.specs/11_min_auth_hooks.md`; recommend loopback posture & optional token for LAN
  - [ ] Confirm GPU‑only stance (no CPU inference) remains consistent

## 2) Crate `.specs/` edits

### Per-crate documentation deliverables (runtime crates only)
- [ ] For each runtime crate, ensure a local `.specs/` exists (create if missing)
- [ ] Add `30_TESTING.md` (umbrella) that links to per-layer test docs below and summarizes scope per ORCH‑3200..3208
- [ ] Add per-layer testing docs (create only those that apply; others may be stubs referencing root harness):
  - [ ] `31_UNIT.md` — unit tests owned by the crate
  - [ ] `32_PROPERTY.md` — property tests (if applicable)
  - [ ] `33_INTEGRATION.md` — narrow integration tests within crate boundary
  - [ ] `34_CONTRACT.md` — contract/provider tests (reference OpenAPI/config schema if applicable)
  - [ ] `35_BDD.md` — cross‑crate flows delegated to root harness; document boundaries and references
  - [ ] `36_DETERMINISM.md` — determinism expectations/applicability for this crate
  - [ ] `37_METRICS.md` — metric names/labels emitted by the crate; reference `.specs/metrics/otel-prom.md`
- [ ] Add `40_ERROR_MESSAGING.md` — enumerate error types/envelopes this crate emits, mapping to codes/statuses, with examples and remediation hints

#### Documentation structure & QA
- [ ] Each new doc MUST include front‑matter: Title (H1), Status, Owner, Date
- [ ] Each new doc MUST include a `## Refinement Opportunities` section
- [ ] Do NOT mint new ORCH‑IDs in crate docs; reference root `/.specs/00_llama-orch.md` IDs
- [ ] Metrics naming in `37_METRICS.md` MUST align with `/.specs/metrics/otel-prom.md`; do not invent names/labels
- [ ] `40_ERROR_MESSAGING.md` MUST map to canonical error envelopes/codes used by orchestrator; include example payloads and redaction notes

### orchestrator-core/.specs/
- [ ] `00_orchestrator_core.md`
  - [ ] Reference canonical `ModelRequirements` (root) and how core consumes it
- [ ] `10_contracts.md`
  - [ ] Add “Test Ownership” note (crate‑local vs BDD cross‑crate)
- [ ] `11_pool_managerd.md`
  - [ ] Reference `PoolSnapshot` shape; remove duplication once canonicalized
- [ ] `12_catalog_core.md`
  - [ ] Reference `ModelRequirements` derivation (no local duplication)
- [ ] `13_engine_provisioner.md`
  - [ ] Remove any references to “CPU‑only fallback”
- [ ] `14_model_provisioner.md`
  - [ ] Reference catalog helpers and `ModelRequirements`
- [ ] `15_worker_adapters.md`
  - [ ] Keep minimal; ensure references to shared definitions
 - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `32_PROPERTY.md`, `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### orchestratord/.specs/
- [ ] `00_orchestratord.md`
  - [ ] Document Bearer seam & identity breadcrumbs; reference Minimal Auth Hooks
- [ ] `10_orchestratord_v2_architecture.md`
  - [ ] Document HTTP/2 preference, buffered SSE emitter, micro‑batch flag, CPU budget expectations, narration hooks
- [ ] `20_contracts.md`
  - [ ] Map error taxonomy & headers; add correlation ID, auth error envelopes; add Test Ownership note
- [ ] `21_orchestrator_core.md`
  - [ ] Reference canonical `ModelRequirements` & snapshot mapping
- [ ] `22_worker_adapters.md`
  - [ ] Reference Adapter Host facade adoption; shared HTTP util norms
 - [ ] Add docs: `40_TESTING.md`, `41_UNIT.md`, `43_INTEGRATION.md`, `44_CONTRACT.md`, `45_BDD.md` (delegation), `47_METRICS.md`, `50_ERROR_MESSAGING.md`

### catalog-core/.specs/
- [ ] `00_catalog_core.md`
  - [ ] Align with ORCH‑ID policy (root‑normative). Remove local ORCH‑IDs or remap to root entries
- [ ] `10_contracts.md`
  - [ ] Promote `exists(id|ref)` & `locate(ModelRef)` to Requirements
- [ ] `11_model_provisioner.md`
  - [ ] Reference helpers and error propagation
- [ ] `12_engine_provisioner.md`
  - [ ] Reference helpers & error propagation
- [ ] `13_orchestrator_core.md`
  - [ ] Reference canonical `ModelRequirements`
- [ ] `14_pool_managerd.md`
  - [ ] Confirm readiness & error logging alignment
 - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### pool-managerd/.specs/
- [ ] `00_pool_managerd.md`
  - [ ] Attach `Authorization: Bearer` on registration/health calls to orchestrator when configured
  - [ ] Keep GPU‑only readiness expectations
- [ ] `10_contracts.md`
  - [ ] Add Test Ownership note
- [ ] `11_model_provisioner.md`, `12_engine_provisioner.md`, `13_orchestrator_core.md`, `14_catalog_core.md`
  - [ ] Reference canonical types; avoid duplication
 - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `35_BDD.md` (delegation), `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### provisioners/engine-provisioner/.specs/
- [ ] `00_engine_provisioner.md`
  - [ ] Remove CPU fallback statement (no forcing `-DGGML_CUDA=OFF`); fail fast with actionable diagnostics; GPU‑only
  - [ ] Ensure PreparedEngine summary requirement is explicit
- [ ] `10_contracts.md`
  - [ ] Remove “CUDA→CPU fallback” from Testing Expectations
- [ ] `11_model_provisioner.md`
  - [ ] Reference catalog helpers & error propagation
- [ ] `12_pool_managerd.md`
  - [ ] Remove CPU‑only normalization examples (e.g., `--n-gpu-layers 0`)
- [ ] `13_catalog_core.md`
  - [ ] Reference helpers & error propagation
- [ ] `14_orchestrator_core.md`
  - [ ] Remove “fallback signals (e.g., CPU‑only)”
 - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `32_PROPERTY.md` (flags normalization), `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### provisioners/model-provisioner/.specs/
- [ ] `00_model_provisioner.md`
  - [ ] Reference catalog helpers; fast‑path exists/locate behavior
- [ ] `10_contracts.md`
  - [ ] Add Test Ownership note
- [ ] `11_catalog_core.md`, `12_engine_provisioner.md`, `13_pool_managerd.md`, `14_orchestrator_core.md`
  - [ ] Reference helpers & canonical types
 - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### worker-adapters/.specs/
- [ ] `00_worker_adapters.md`
  - [ ] Promote shared HTTP util norms
- [ ] `10_contracts.md`
  - [ ] Add Test Ownership note
- [ ] `11_orchestratord.md`
  - [ ] When configured, attach `Authorization: Bearer` on calls to orchestrator
- [ ] `12_pool_managerd.md`
  - [ ] Keep references minimal; ensure no contradictions
 - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`, `34_CONTRACT.md` (if applicable), `35_BDD.md` (delegation), `37_METRICS.md`, `40_ERROR_MESSAGING.md`

### Adapter crates (per engine) — if/when present locally
- [ ] For each adapter crate (e.g., `worker-adapters/llamacpp-http`, `worker-adapters/mock`, future `vllm-http`, `tgi-http`, `triton`):
  - [ ] Ensure local `.specs/` exists (create if missing)
  - [ ] Add `30_TESTING.md` (umbrella) and per-layer docs as applicable (`31_UNIT.md`, `33_INTEGRATION.md`, `34_CONTRACT.md` if exposing external APIs, `37_METRICS.md`)
  - [ ] Add `40_ERROR_MESSAGING.md` with HTTP mapping and redaction rules; align error taxonomy with orchestrator envelopes

### Observability and shared runtime libs
- [ ] `observability/narration-core/.specs/`
  - [ ] Create `.specs/` if missing
  - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (if any crate-boundary tests), `37_METRICS.md` (fields surfaced), `40_ERROR_MESSAGING.md`
- [ ] `adapter-host/.specs/`
  - [ ] Create `.specs/` if missing
  - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (facade/registry), `37_METRICS.md` (narration/metrics wrappers), `40_ERROR_MESSAGING.md`
- [ ] `worker-adapters/http-util/.specs/`
  - [ ] Create `.specs/` if missing
  - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (client init/retry), `34_CONTRACT.md` (if any provider verify applies), `37_METRICS.md` (if emitting), `40_ERROR_MESSAGING.md`

### CLI crates
- [ ] `cli/llama-orch-cli/.specs/`
  - [ ] Create `.specs/` if missing
  - [ ] Add docs: `30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md` (against mock endpoints), `34_CONTRACT.md` (if verifying OpenAPI client usage), `40_ERROR_MESSAGING.md` (CLI error taxonomy, exit codes)
- [ ] `cli/consumer-tests/.specs/` (if considered part of runtime deliverables)
  - [ ] Create `.specs/` if missing
  - [ ] Add docs: `30_TESTING.md`, per-layer stubs as applicable; `40_ERROR_MESSAGING.md` optional (document tooling errors)

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

- [ ] `observability/narration-core/.specs/00_narration_core.md` (facade API, fields, redaction, test capture)
- [ ] `adapter-host/.specs/00_adapter_host.md` (registry/facade, narration/metrics wrappers, correlation)
- [ ] `worker-adapters/http-util/.specs/00_http_util.md` (shared HTTP client, retries/backoff/HTTP2, streaming decode helpers, redaction)

## 4) Clean-up: GPU-only (remove CPU fallback mentions)

- [ ] Search & scrub CPU fallback language across specs
  - [ ] `provisioners/engine-provisioner/.specs/00_engine_provisioner.md`
  - [ ] `provisioners/engine-provisioner/.specs/10_contracts.md`
  - [ ] `provisioners/engine-provisioner/.specs/12_pool_managerd.md`
  - [ ] `orchestrator-core/.specs/13_engine_provisioner.md`
  - [ ] `provisioners/engine-provisioner/.specs/14_orchestrator_core.md`

## 5) ORCH‑ID policy normalization

- [ ] Keep ORCH‑IDs normative in `/.specs/00_llama-orch.md`
- [ ] In crate specs, reference root ORCH‑IDs (do not mint new ones)
- [ ] Migrate or delete local ORCH‑IDs in `catalog-core/.specs/00_catalog_core.md`

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
