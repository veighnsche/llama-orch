# catalog-core — Component Specification (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## 0) Purpose & Scope

`catalog-core` provides model reference parsing, local catalog/indexing, lifecycle state, and digest verification for llama-orch. It is engine-agnostic and does not perform network downloads or process execution.

In scope:
- ModelRef parsing and normalization.
- Catalog storage trait and the default filesystem implementation.
- Lifecycle (Active/Retired) and trust state recording.
- Digest verification helpers.

Out of scope:
- Network fetching (`hf:`, `http(s)`, `s3`, `oci`) and policy gating — delegated to `model-provisioner`.
- Engine preparation — delegated to `engine-provisioner` and adapters.

## 1) Normative Requirements (RFC-2119)

ORCH-ID policy: requirement IDs are normative at `/.specs/00_llama-orch.md`. This document references those requirements and does not mint local `ORCH-####` IDs.

- The crate MUST expose a `CatalogStore` trait with operations: `get`, `put`, `set_state`, `list`, and `delete`.
- The crate MUST ship a default `FsCatalog` implementation rooted under a directory (`root`).
- `FsCatalog` MUST write the index atomically (temp file + atomic rename) and SHOULD fsync the file and directory to be crash-safe.
- The index MUST include a schema version field; incompatible versions MUST be rejected with a typed error.
- Entries MUST record: `id`, `local_path`, `lifecycle` (Active|Retired), optional `digest`, optional `last_verified_ms`.
- `delete(id)` MUST remove the entry and SHOULD best-effort remove on-disk artifacts (file or directory). Failures MUST be reported but MUST NOT corrupt the index.
- The crate MUST provide helpers for `default_model_cache_dir()` and `verify_digest()`; verification outcomes MUST distinguish pass/warn/fail.
- The crate MUST NOT perform network I/O; ensure-present for network schemes is out of scope.
- Paths MUST be normalized; traversal outside of `root` MUST be rejected when staging via callers.

## 2) Data Types & Semantics

- `ModelRef`
  - Variants: `Hf { org, repo, path? }`, `File { path }`, `Url { url }`.
  - Parse from string; normalization rules documented (trim, case where applicable, no trailing slash changes semantics).
- `CatalogEntry`
  - See normative requirements above. `local_path` is absolute or `root`-relative normalized path.
- `Digest`
  - `algo` (e.g., sha256), `value` (hex). Hex must be lowercased.

## 3) Interfaces & Contracts

- `CatalogStore` trait stability is required pre-1.0 within this repo. Backwards compatibility is not guaranteed across minor versions pre-1.0, but call sites MUST be updated in the same PR per repo policy.
- `FsCatalog` index file layout: JSON lines or single JSON; MUST include `version` and entry map/list.

## 4) Observability

- Expose structured errors; callers emit logs/metrics.
- Suggested counters (emitted by callers): `catalog_verifications_total{result}`; `catalog_entries_total`.

## 5) Security

- No secrets; no network.
- Path normalization and denial of traversal outside `root`.

## 6) Testing & Proof Bundle

- Unit tests: index round-trip, delete semantics, parse edge-cases, digest verification.
- Concurrency/corrosion tests: interrupted writes simulation; index readability.
- Include pact/snapshots only where it interacts with other crates through stable interfaces.

## 7) Open Questions

- Should `FsCatalog` adopt a content-addressable layout by default (digest-first) or remain id-first with digest side metadata?
- Introduce `SqliteCatalog` behind the same trait for larger catalogs?

## 8) Refinement Opportunities

- Content-addressable storage (CAS) with GC/eviction and byte-reuse across IDs.
- Add `exists(id|ref)` and `locate(ModelRef)` helpers (see `/.specs/25-catalog-core.md` for root normative requirements).
- Trust state promotion/demotion rules and hooks (verified/warned/unverified) with provenance.
- Advisory locking for multi-process safety.
