# SPEC changes needed for proposals migration and alignment (2025-09-19)

This document captures concrete edits needed to align all specs under `.specs/` across the workspace and to migrate all approved proposals under `/.specs/proposals/` into the normative specs. It follows the repo’s Spec-First workflow and testing guidelines.

## Scope of audit

- Root specs inventoried under `/.specs/` including metrics and proposals.
- Crate-level specs under `orchestrator-core/.specs/`, `orchestratord/.specs/`, `catalog-core/.specs/`, `pool-managerd/.specs/`, `provisioners/*/.specs/`, `worker-adapters/.specs/`.
- Test harness specs in `test-harness/*/.specs/`.

No numbering collisions were found within directories. Cross-crate wiring uses a consistent 11–1x scheme. All checked specs contain a “Refinement Opportunities” section as preferred.

## Decisions to set/affirm repo-wide

- ORCH-ID policy: Root is normative
  - Keep all ORCH-#### requirement IDs in `/.specs/00_llama-orch.md` and reference them from crate specs. Remove or remap crate-local ORCH-IDs to root entries.
  - Action: Review `catalog-core/.specs/00_catalog_core.md` which currently embeds ORCH-33xx locally; migrate those to root or explicitly reference root equivalents.

- Refinement Opportunities sections
  - Keep required across all specs. Items promoted to normative must be removed from Refinements to avoid duplication.

- Metrics contract centralization
  - Names/labels remain authoritative in `/.specs/metrics/otel-prom.md` and `ci/metrics.lint.json`. Crate specs should reference, not redefine, metric names.

- Minimal Auth Hooks adoption (spec-only, no runtime change yet)
  - Treat `/.specs/11_min_auth_hooks.md` as Accepted for planning purposes and integrate its requirements into the relevant specs.
  - Root spec `/.specs/00_llama-orch.md` Security & Policy should reference AUTH-IDs (AUTH-1001..AUTH-1008) and the loopback posture.
  - Orchestratord specs should document Bearer token seam, startup refusal on non-loopback without token, and log identity breadcrumbs.
  - Config schema specs should add keys: `BIND_ADDR`/`ORCHD_ADDR`, `AUTH_TOKEN`, `AUTH_OPTIONAL`, `TRUST_PROXY_AUTH`.

- GPU-only policy (no CPU fallback anywhere)
  - Affirm the GPU-only stance: provisioning and runtime MUST NOT fallback to CPU in error paths.
  - Remove remaining mentions of CPU-only fallback from engine-provisioner and wiring specs; keep `/.specs/50-engine-provisioner.md` GPU-only language as normative.

## Proposals → migration plan (mark as Accepted and integrate)

1) Proposal: Human‑Readable Narration in Logs (repo‑wide)
- Status: set to Accepted in `/.specs/proposals/2025-09-19-human-narration-logging.md`.
- Root spec update: Add an Observability/Narration subsection in `/.specs/00_llama-orch.md` with ORCH‑33xx series (emission points for admission, placement, stream, cancel; redaction; coverage in proof bundles).
- New crate spec: Create `observability/narration-core/.specs/00_narration_core.md` describing minimal API, log event schema/fields, redaction gates, and tests.
- Orchestratord spec updates: In `orchestratord/.specs/10_orchestratord_v2_architecture.md` describe narration hooks; in `orchestratord/.specs/20_contracts.md` reference correlation ID propagation and narration mapping.
- Pool-managerd/provisioners: Add narration integration requirements into `pool-managerd/.specs/00_pool_managerd.md` and `provisioners/*/.specs/00_*` (preflight/build/fallback/spawn/readiness narration).
- Proof bundles: Update `.docs/testing/` guidance to include narration coverage and log excerpts.

2) Proposal: Performance Streamlining Across Crate Boundaries (repo‑wide)
- Status: set to Accepted in `/.specs/proposals/2025-09-19-performance-streamlining.md`.
- Orchestratord root spec: Promote to normative in `/.specs/20-orchestratord.md`:
  - Prefer HTTP/2 for SSE with fallback to HTTP/1.1.
  - Use a buffered writer for SSE emission; optional micro-batch flag (default off) with bounds.
  - Preserve `started → token* → end` order; keep heartbeat optional and compatible.
- Orchestratord architecture: In `orchestratord/.specs/10_orchestratord_v2_architecture.md`, document emitter internals, buffering, backpressure, and CPU budget expectations. Remove any overlapping Refinements that become normative.
- Metrics: Ensure no changes violate `/.specs/metrics/otel-prom.md`; if additional histograms/buckets are needed, propose updates there.

3) Proposal: Adapter Host (in‑process) + Shared HTTP Util for Adapters
- Status: set to Accepted in `/.specs/proposals/2025-09-19-adapter-host-and-http-util.md`.
- New crate specs:
  - `adapter-host/.specs/00_adapter_host.md`: Registry, facade (submit/cancel/health/props), narration & metrics wrappers, correlation ID propagation.
  - `worker-adapters/http-util/.specs/00_http_util.md`: Shared reqwest client, retry/backoff, streaming decode helpers, redaction, HTTP/2 keep-alive.
- Promote to normative in `/.specs/35-worker-adapters.md`:
  - Require shared HTTP util usage across HTTP-based adapters; consistent timeouts, retries with jitter, and error taxonomy mapping.
  - Require deterministic streaming decode path and redaction helpers.
- Orchestratord integration: Update `orchestratord/.specs/22_worker_adapters.md` to reference Adapter Host facade usage.

4) Proposal: Testing Ownership & Scope — Per‑Crate Behavior; BDD for Cross‑Crate
- Status: set to Accepted in `/.specs/proposals/2025-09-19-testing-ownership-and-scope.md`.
- Root harness spec: Update `/.specs/72-bdd-harness.md` to make ownership boundaries normative:
  - Crate-local tests cover crate behavior; BDD focuses on cross-crate flows only.
  - Step registry rejects unknown/ambiguous steps; clear scoping per crate boundary.
- Crate specs: For each crate’s `10_contracts.md`, add a short “Test Ownership” note referencing the root harness spec and what is expected locally vs. in BDD.

## Cross-cutting consolidations to avoid duplication

- Canonical ModelRequirements
  - Define once under `/.specs/10-orchestrator-core.md` (Data Types) and reference from wiring specs:
    - `orchestrator-core/.specs/12_catalog_core.md`, `14_model_provisioner.md`
    - `catalog-core/.specs/13_orchestrator_core.md`
    - `orchestratord/.specs/21_orchestrator_core.md`
  - Provide derivation rules from catalog + adapter metadata; prefer a shared helper crate eventually.

- Catalog read-only helpers as normative
  - Promote `exists(id|ref)` and `locate(ModelRef)` from Refinements to Requirements in `catalog-core/.specs/10_contracts.md` and reflect in `/.specs/25-catalog-core.md`.
  - Update dependent specs to reference these helpers:
    - `provisioners/model-provisioner/.specs/00_model_provisioner.md` and `10_contracts.md`
    - `provisioners/engine-provisioner/.specs/00_engine_provisioner.md` and `10_contracts.md`
    - `pool-managerd/.specs/00_pool_managerd.md` (readiness logic may leverage helpers).

## Repo-wide Integration Matrix (Root + All Crates)

- Human-Readable Narration in Logs
  - Root specs: `/.specs/00_llama-orch.md` (Observability/Narration subsection), `/.specs/20-orchestratord.md` (SSE/narration hooks).
  - Crates: `orchestratord/.specs/10_orchestratord_v2_architecture.md`, `orchestratord/.specs/20_contracts.md`, `pool-managerd/.specs/00_pool_managerd.md`, `provisioners/*/.specs/00_*`, plus new `observability/narration-core/.specs/00_narration_core.md`.

- Performance Streamlining
  - Root: `/.specs/20-orchestratord.md` (HTTP/2 for SSE, buffered emitter, micro-batch flag).
  - Crates: `orchestratord/.specs/10_orchestratord_v2_architecture.md` (internals).

- Adapter Host + HTTP Util
  - Root: `/.specs/35-worker-adapters.md` (normative shared HTTP util).
  - Crates: `orchestratord/.specs/22_worker_adapters.md` (Adapter Host integration), new `adapter-host/.specs/00_adapter_host.md`, new `worker-adapters/http-util/.specs/00_http_util.md`, and adapter specs under `/.specs/40..44` to reference the util.

- Testing Ownership & Scope
  - Root: `/.specs/72-bdd-harness.md` (ownership boundaries), ensure cross-crate only.
  - Crates: each `*/.specs/10_contracts.md` adds “Test Ownership” note.

- Minimal Auth Hooks (AUTH-1001..AUTH-1008)
  - Root: `/.specs/00_llama-orch.md` Security & Policy subsection references `/.specs/11_min_auth_hooks.md`.
  - Crates: `orchestratord/.specs/00_orchestratord.md`, `10_orchestratord_v2_architecture.md`, `20_contracts.md` (Bearer seam, startup refusal, identity logs); `pool-managerd/.specs/00_pool_managerd.md` (attach `Authorization: Bearer` on registration/health calls to orchestrator when configured); `worker-adapters/.specs/11_orchestratord.md` (clients add `Authorization: Bearer` when configured).
  - Config: `contracts/config-schema/.specs` via `/.specs/60-config-schema.md` to add AUTH keys.
  - Test harness: `test-harness/bdd/.specs/00_test_harness.md` to include positive/negative auth scenarios.

- GPU-only (No CPU fallback)
  - Root: keep `/.specs/50-engine-provisioner.md` GPU-only normative text.
  - Crates: remove CPU fallback mentions in `provisioners/engine-provisioner/.specs/*` and wiring specs; ensure `/.specs/30-pool-managerd.md` “CPU inference spillover is disallowed” remains.

## Root `.specs/` changes due to proposal approvals

- Proposals — change status to Accepted
  - `/.specs/proposals/2025-09-19-human-narration-logging.md`
  - `/.specs/proposals/2025-09-19-performance-streamlining.md`
  - `/.specs/proposals/2025-09-19-adapter-host-and-http-util.md`
  - `/.specs/proposals/2025-09-19-testing-ownership-and-scope.md`
  - Also set `/.specs/11_min_auth_hooks.md` to “Status: Accepted” (spec seam, no runtime change).

- `/.specs/00_llama-orch.md`
  - Add an Observability/Narration subsection (ORCH‑33xx) describing narration emission points (admission, placement, stream, cancel), redaction, and proof bundle expectations.
  - In Security & Policy, reference `/.specs/11_min_auth_hooks.md` (AUTH‑1001..1008) and the loopback posture. Keep home-profile “open locally” stance; clarify this is a spec seam for future adoption.
  - Affirm GPU-only policy (no CPU inference paths) and engine auto‑provisioning alignment in §2.12.

- `/.specs/20-orchestratord.md`
  - Promote: HTTP/2 preferred for SSE (fallback to HTTP/1.1), buffered SSE emitter, optional micro-batch flag (default off, bounded), preserved event order (`started → token* → end`), optional heartbeat compatibility.
  - Add correlation ID propagation and narration hook references.
  - Move overlapping items out of Refinements into the normative sections.

- `/.specs/10-orchestrator-core.md`
  - Introduce canonical `ModelRequirements` in Data Types; define derivation rules from catalog + adapter metadata.
  - Clarify tie-break determinism and placement feasibility inputs (VRAM, ctx, quantization, extensions) and how they map into `PlacementInput`.
  - Ensure references from wiring specs point to this canonical definition.

- `/.specs/35-worker-adapters.md`
  - Promote to normative: shared HTTP util usage across HTTP adapters (timeouts, retries with jitter, error taxonomy mapping, redaction helpers), deterministic low‑alloc streaming decode path.
  - Reference `adapter-host` and `worker-adapters/http-util` specs.

- `/.specs/40-worker-adapters-llamacpp-http.md`, `/.specs/41-worker-adapters-vllm-http.md`, `/.specs/42-worker-adapters-tgi-http.md`, `/.specs/43-worker-adapters-triton.md`, `/.specs/44-worker-adapters-openai-http.md`
  - Add references to `worker-adapters/http-util` and Adapter Host integration; confirm streaming order and redaction expectations.

- `/.specs/50-engine-provisioner.md`
  - Keep GPU‑only language explicit (“fail fast if CUDA/GPU unavailable; no CPU fallback”).
  - Add PreparedEngine summary requirement alignment if not already present (engine name/version, build ref, digest, flags, mode, binary path) as an observable output.

- `/.specs/55-model-provisioner.md`
  - Reference catalog helpers `exists(id|ref)` and `locate(ModelRef)` as normative fast‑paths to avoid redundant staging.

- `/.specs/25-catalog-core.md`
  - Promote read‑only helpers `exists(id|ref)` and `locate(ModelRef)` from Refinements to Requirements and cross‑link crate contracts.

- `/.specs/60-config-schema.md`
  - Add AUTH config keys aligned with minimal auth hooks: `BIND_ADDR`/`ORCHD_ADDR`, `AUTH_TOKEN`, `AUTH_OPTIONAL`, `TRUST_PROXY_AUTH`.
  - Keep home‑profile defaults and add `x-examples` for auth configurations.

- `/.specs/00_home_profile.md`
  - Cross-link to `/.specs/11_min_auth_hooks.md`; reiterate recommended loopback posture and optional token usage for LAN exposure.
  - Confirm GPU-only stance and remove any lingering CPU inference wording (none found today; keep vigilance).

- `/.specs/70-determinism-suite.md` and `/.specs/72-bdd-harness.md`
  - Clarify test ownership & scope (crate‑local vs cross‑crate) and add auth scenarios to the BDD outline.

- `/.specs/71-metrics-contract.md` and `/.specs/metrics/otel-prom.md`
  - No required changes from the proposals, unless we decide to add bucket guidance for SSE latency metrics. If so, propose in metrics spec first.

## File-specific edits (concrete)

- `/.specs/00_llama-orch.md`
  - Add Observability/Narration subsection (ORCH‑33xx) with emission points and redaction rules.
  - Ensure Arch/CachyOS package policy for provisioning remains in §2.12; cross-link to provisioners.
  - Confirm Security/Policy statements match current home-profile stance and are consistent across crates.

- `/.specs/20-orchestratord.md`
  - Promote HTTP/2 preference and buffered SSE emission to normative; define micro-batch flag semantics.
  - Update Refinements to remove items promoted.

- `orchestratord/.specs/10_orchestratord_v2_architecture.md`
  - Document SSE emitter buffering, CPU budget expectations, and micro-batch behavior. Note HTTP/2 preference.

- `/.specs/35-worker-adapters.md` and `orchestratord/.specs/22_worker_adapters.md`
  - Require usage of `worker-adapters/http-util` for HTTP adapters; standardize retries, timeouts, error taxonomy, and redaction.
  - Reference Adapter Host facade for integration points.

- `catalog-core/.specs/10_contracts.md` and `/.specs/25-catalog-core.md`
  - Promote `exists` / `locate` helpers to normative.

- `/.specs/10-orchestrator-core.md` and wiring specs listed above
  - Introduce canonical `ModelRequirements` and reference it.

- GPU-only policy (remove CPU fallback mentions)
  - `provisioners/engine-provisioner/.specs/00_engine_provisioner.md`: remove the sentence mandating fallback to CPU-only on CUDA configure failure; replace with “fail fast with actionable diagnostics; no CPU inference path.”
  - `provisioners/engine-provisioner/.specs/10_contracts.md`: remove “CUDA→CPU fallback” from Testing Expectations.
  - `provisioners/engine-provisioner/.specs/12_pool_managerd.md`: remove CPU-only normalization example (e.g., `--n-gpu-layers 0`).
  - `orchestrator-core/.specs/13_engine_provisioner.md`: remove “CPU-only fallback” from Expectations on engine-provisioner.
  - `provisioners/engine-provisioner/.specs/14_orchestrator_core.md`: remove mention of “fallback signals (e.g., CPU-only)”.

- `orchestrator-core/.specs/10_contracts.md`, `orchestratord/.specs/20_contracts.md`, and each crate’s `10_contracts.md`
  - Add a “Test Ownership” note referencing `/.specs/72-bdd-harness.md` and expected coverage.

- Proposals statuses
  - Change header status fields to “Accepted” in:
    - `/.specs/proposals/2025-09-19-human-narration-logging.md`
    - `/.specs/proposals/2025-09-19-performance-streamlining.md`
    - `/.specs/proposals/2025-09-19-adapter-host-and-http-util.md`
    - `/.specs/proposals/2025-09-19-testing-ownership-and-scope.md`
  - Set `/.specs/11_min_auth_hooks.md` to “Status: Accepted” and treat as base spec for auth seams.

## New spec files to create

- `observability/narration-core/.specs/00_narration_core.md` — minimal narration facade, fields, redaction, test capture.
- `adapter-host/.specs/00_adapter_host.md` — adapter registry/facade, narration/metrics wrappers.
- `worker-adapters/http-util/.specs/00_http_util.md` — shared HTTP client, retries/backoff/HTTP2, streaming decode helpers, redaction.

## Acceptance and proof bundles

- Ensure proof bundles include: SSE transcripts (with/without micro-batch), narration coverage excerpts, determinism outputs, metrics lints, EngineEntry snapshots.
- Update `.docs/testing/` guidance if needed to reference narration and micro-batch flows.

## Verification checklist (PRs)

- Prefix commits with ORCH-#### and include spec+contract+tests in the same change.
- Run: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --workspace --all-features -- --nocapture`, `cargo xtask regen-openapi`, `cargo xtask regen-schema`, `cargo run -p tools-spec-extract --quiet`.
- `cargo xtask dev:loop` should pass locally; include links to updated specs and artifacts.

## Ownership & next steps

- Choose ORCH-ID normalization (recommend: root-only) and migrate catalog-core local IDs.
- Implement proposal migrations as above; remove duplicate Refinements when promoting to normative.
- Create missing `.specs/` for new crates (`observability/narration-core`, `adapter-host`, `worker-adapters/http-util`).
- After edits, re-run link checks and the spec-extract tool to refresh any indexes.
