# model-provisioner — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Staging flows, id normalization, catalog updates, digest verification hooks.

## Test Catalog

- Unit
  - `ModelId`/`ModelRef` normalization and validation
  - Staging planner (file-only) produces atomic move/copy steps under a temp root
  - Digest computation hooks: pass/warn/fail outcomes, no panics on missing digests

- Integration
  - `FsCatalog` update on successful stage: index round-trip, lifecycle transitions
  - Deletion best-effort: missing file deletion yields warning, not panic
  - Concurrency: single-flight guards prevent duplicate staging for same id

## Execution & Tooling

- Run: `cargo test -p model-provisioner -- --nocapture`
- Keep tests hermetic; simulate network fetchers behind feature gates (disabled by default)

## Traceability

- Aligns with `catalog-core` helpers (`exists/locate`) and index schema

## Refinement Opportunities

- Feature-gated network fetchers and single-flight locks.
