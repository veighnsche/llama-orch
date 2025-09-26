# catalog-core — Unit Tests (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `catalog-core/`

## 0) Scope

Validate `CatalogStore` trait operations, `ModelRef` parsing, `Digest` helper, and read‑only helpers (`exists/locate`). Tests MUST NOT perform network I/O and MUST use a temporary catalog root.

## 1) Test Matrix (normative)

- [CAT-UNIT-3101] `CatalogStore` CRUD
  - `put`/`get` round‑trip MUST preserve all fields
  - `list` MUST include inserted entries and be stable modulo ordering
  - `set_state` MUST update lifecycle to `Active|Retired` deterministically
  - `delete` MUST remove the entry and return `Ok(true)`; deleting a missing entry MUST return `Ok(false)`

- [CAT-UNIT-3102] Atomic Write Discipline
  - Index MUST be written via temp file + atomic rename (verify by intercepting file names or by simulating interruption)

- [CAT-UNIT-3103] Schema Version Handling
  - Loading an incompatible `version` MUST yield a typed error and MUST NOT partially load state

- [CAT-UNIT-3104] `ModelRef::parse`
  - Valid inputs (`hf:org/repo[/path]`, `file:/abs/path`, `relative/path`, `https://...`) MUST normalize as specified
  - Invalid inputs MUST return typed errors (no panics)

- [CAT-UNIT-3105] `verify_digest`
  - Correct digest → PASS; incorrect → FAIL; missing digest → WARN or bypass per helper semantics; MUST NOT panic

- [CAT-UNIT-3106] Read‑Only Helpers (`exists/locate`)
  - `exists(id|ref)` MUST return presence without writes; `locate(ModelRef)` MUST resolve normalized paths when present and return `None` otherwise

- [CAT-UNIT-3107] Path Normalization & Traversal Safety
  - Attempts to escape the catalog root MUST be rejected deterministically

## 2) Test Catalog

- CatalogStore Operations
  - Create/open/close with minimal schema; in-memory mocks where possible
  - Insert/update/delete entries and verify invariants (unique ids, valid paths)

- `ModelRef::parse`
  - Accept forms: `hf:org/repo[/path]`, `file:/abs/path`, `relative/path`, `https://...`
  - Reject invalid forms with typed errors; normalization of path components

- Digest Helper
  - Compute SHA256 (and supported algos); invalid algorithm string → typed error
  - Distinguish pass/warn/fail outcomes; no panics on empty/missing files (unit-level uses temp files)

- Read-only Helpers
  - `exists(id|ref)` returns accurate presence without side effects
  - `locate(ModelRef)` resolves normalized filesystem path under catalog root

## 3) Execution

- `cargo test -p catalog-core -- --nocapture`

## 4) Traceability

- Root spec: `/.specs/25-catalog-core.md`
- Contracts: `catalog-core/.specs/10_contracts.md`
- Code: `catalog-core/src/`
- Aligns with `catalog-core/.specs/30_TESTING.md`

## Refinement Opportunities

- Add path traversal negative cases and normalize edge cases.
