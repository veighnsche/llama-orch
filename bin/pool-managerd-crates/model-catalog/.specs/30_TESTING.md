# catalog-core — Testing Overview (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `catalog-core/`

## 0) Scope

Validate index round‑trip, atomic writes, schema versioning, `ModelRef` parsing, digest verification, and read‑only helpers (`exists/locate`). Tests MUST avoid network I/O; file system effects MAY be performed under a temporary root.

## 1) Test Matrix (normative)

- [CAT-TEST-3001] Index Round‑Trip
  - GIVEN entries with `id`, `local_path`, `lifecycle`, optional `digest`
  - WHEN persisted and reloaded via `FsCatalog`
  - THEN the entries MUST be equal modulo ordering; schema `version` MUST be present

- [CAT-TEST-3002] Atomic Writes & Crash Safety
  - GIVEN an interrupted write simulation (write temp; crash before rename)
  - WHEN reopening the index
  - THEN the catalog MUST remain readable and either reflect the previous committed state or the new state after a successful atomic rename

- [CAT-TEST-3003] Schema Version Rejection
  - GIVEN an index with an incompatible `version`
  - WHEN opening
  - THEN operations MUST fail with a typed error and MUST NOT partially load entries

- [CAT-TEST-3004] `ModelRef::parse` Coverage
  - GIVEN `hf:org/repo[/path]`, `file:/abs/path`, `relative/path`, `https://...`
  - THEN parse MUST succeed with normalized forms; invalid forms MUST return typed errors

- [CAT-TEST-3005] Digest Verification
  - GIVEN a file and `Digest{algo,value}`
  - WHEN `verify_digest` runs
  - THEN outcomes MUST distinguish pass/warn/fail and NEVER panic

- [CAT-TEST-3006] Read‑Only Helpers (`exists/locate`)
  - GIVEN a populated catalog root
  - WHEN calling `exists(id|ref)` and `locate(ModelRef)`
  - THEN helpers MUST perform no writes and MUST return correct presence and normalized paths

## 2) Traceability

- Root spec: `/.specs/25-catalog-core.md`
- Contracts: `catalog-core/.specs/10_contracts.md`
- Code: `catalog-core/src/`

## Refinement Opportunities

- Crash-safety simulations and fsync behavior audits.
