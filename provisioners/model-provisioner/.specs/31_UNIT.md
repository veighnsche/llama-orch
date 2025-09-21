# model-provisioner — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- File-only staging, id normalization, digest verification plumbing.

## Test Catalog

- ID/Ref Normalization
  - Accept supported forms (hf:, file:, relative, https) → normalized `ModelRef`
  - Reject invalid forms with typed errors; no panics

- Staging Planner (file-only)
  - Atomic write pattern: write temp → rename; simulate interruptions safely
  - No network calls in unit; paths resolved under temp root

- Digest Verification Plumbing
  - Compute supported digests; unknown algorithms rejected with a typed error
  - Distinguish pass/warn/fail; never panic on missing digests

## Structure & Conventions

- Keep tests pure and hermetic; use temp dirs via `tempfile`
- Table-driven tests for normalization matrices and digest cases

## Execution

- `cargo test -p model-provisioner -- --nocapture`

## Traceability

- Aligns with `catalog-core` index and helpers; see `catalog-core/.specs/30_TESTING.md`

## Refinement Opportunities

- Edge cases for path normalization.
