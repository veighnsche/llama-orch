# catalog-core — Integration Tests (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `catalog-core/`

## 0) Scope

Validate `FsCatalog` behavior under integration scenarios: crash‑safety simulations, engine entry integration with `engine-provisioner`, and (optional) multi‑process advisory safety. Network I/O is out of scope.

## 1) Scenarios (normative)

- [CAT-INT-3301] Crash‑Safety Atomicity
  - GIVEN writes that simulate a crash after writing a temp file but before atomic rename
  - WHEN the process restarts and reopens the catalog
  - THEN the index MUST be readable and reflect either the prior committed state or the fully committed new state; partial/garbled states MUST NOT load

- [CAT-INT-3302] Engine Entry Integration
  - GIVEN `engine-provisioner` writes an `EngineEntry` via `FsCatalog`
  - WHEN reading engine metadata from the catalog
  - THEN fields MUST match the write (`engine`, `version`, `build_ref?`, `digest?`, `artifacts[]`, `created_ms`) and MUST be atomic

- [CAT-INT-3303] Concurrent Readers (advisory)
  - GIVEN a long‑running reader while a writer performs atomic rename
  - WHEN the rename occurs
  - THEN the reader MUST continue to observe a consistent snapshot (old file), and subsequent opens MUST observe the new file

## 2) Test Catalog

- Atomic Writes & Crash Simulation
  - GIVEN a temp root and an index write in progress
  - WHEN the process is interrupted between temp write and rename
  - THEN reopening MUST yield a valid index (previous or new committed state)

- Engine Entry Integration
  - GIVEN an engine entry added by provisioners
  - WHEN reading via `FsCatalog`
  - THEN fields (`engine`, `version`, `binary_path`, `sampler_profile_version?`) appear as expected

- Concurrency Guards
  - GIVEN concurrent writers under the same root
  - WHEN performing writes
  - THEN advisory mechanisms prevent corruption and readers see a consistent view

## 3) Fixtures & Mocks

- Temp directory roots; fake engine entries
- Helpers to simulate interrupted writes

## 4) Execution

- `cargo test -p catalog-core -- --nocapture`

## 5) Traceability

- Root spec: `/.specs/25-catalog-core.md`
- Engine Catalog: `/.specs/56-engine-catalog.md`
- Code: `catalog-core/src/`
- Aligns with `provisioners/*` integration tests and `orchestrator-core` snapshot consumers

## Refinement Opportunities

- Add multi-process advisory lock tests (future).
