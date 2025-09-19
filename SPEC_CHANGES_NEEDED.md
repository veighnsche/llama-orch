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

- `orchestrator-core/.specs/10_contracts.md`, `orchestratord/.specs/20_contracts.md`, and each crate’s `10_contracts.md`
  - Add a “Test Ownership” note referencing `/.specs/72-bdd-harness.md` and expected coverage.

- Proposals statuses
  - Change header status fields to “Accepted” in:
    - `/.specs/proposals/2025-09-19-human-narration-logging.md`
    - `/.specs/proposals/2025-09-19-performance-streamlining.md`
    - `/.specs/proposals/2025-09-19-adapter-host-and-http-util.md`
    - `/.specs/proposals/2025-09-19-testing-ownership-and-scope.md`

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
