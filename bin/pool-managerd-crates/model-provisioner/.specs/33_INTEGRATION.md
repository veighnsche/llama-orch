# model-provisioner — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Catalog integration with `FsCatalog` and deletion best-effort semantics.

## Test Catalog

- Stage Model (happy path)
  - GIVEN a source file under temp root
  - WHEN staged with metadata and optional digest
  - THEN the artifact is placed atomically and `FsCatalog` index updated with lifecycle/state

- Delete Model (best-effort)
  - GIVEN a catalog entry
  - WHEN deletion is requested
  - THEN file removal errors produce warnings but index removal succeeds; index remains consistent

- Concurrency / Single-Flight
  - GIVEN two concurrent stage requests for the same id
  - THEN only one performs I/O; the other observes the staged result (or waits), avoiding duplicates

## Fixtures & Mocks

- Temp directory roots for staging and catalog storage
- Fake digests and small test files to validate hashing without heavy I/O

## Execution

- `cargo test -p model-provisioner -- --nocapture`

## Traceability

- Aligns with `catalog-core` index round-trip and helpers

## Refinement Opportunities

- Concurrency tests for atomic writes.
