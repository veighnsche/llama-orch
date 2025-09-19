# catalog-core — TODO (Spec → Contract → Tests → Code)

This TODO reflects the current state of `catalog-core` and the next steps needed to reach production readiness. Verified against the existing code in `src/lib.rs` and project specs. Current implementation includes:

- `ModelRef::parse()` for `hf:org/repo[/path]`, `file:` (and relative paths), and generic `Url` parsing
- Filesystem catalog `FsCatalog` with JSON index and APIs: `get/put/set_state/list/delete`
- `FileFetcher` (local files only); `Hf/Url` fetchers are not implemented here (separation: see model‑provisioner)
- Digest verification helper and a default cache path helper

Traceability: `.specs/00_llama-orch.md` (§2.6 Catalog, §2.11 Model Selection & Auto‑Fetch), `.specs/71-metrics-contract.md`

## High (must)

- Index & concurrency safety
  - [ ] Atomic index writes (temp file + fsync) to avoid corruption
  - [ ] Reader/writer guard for multi‑process access (advisory lock file)
  - [ ] Validate on load (schema/version tag); repair strategy for minor inconsistencies
- Content addressability & layout rules
  - [ ] Add content‑addressable IDs (by digest) alongside logical IDs
  - [ ] Enforce canonical on‑disk layout (documented) under `~/.cache/models`
  - [ ] Disallow path traversal; normalize `local_path` within root when staging via provisioners
- API surface hardening
  - [ ] Typed error reasons for `delete` (e.g., NotFound vs Busy)
  - [ ] `evict(id)` and `gc()` APIs (return report: bytes reclaimed, entries removed)
  - [ ] `exists(id|ref)` and `locate(ModelRef)` helpers
- Lifecycle & verification
  - [ ] Track `trust_state` (verified | warned | unverified) separately from `lifecycle`
  - [ ] Wire `verify_digest` into a `verify(entry, expected)` helper; produce structured outcome

## Mid (should)

- Optional Sqlite catalog
  - [ ] `SqliteCatalog` implementation behind `CatalogStore` trait (same semantics as `FsCatalog`)
  - [ ] Migration versioning for index schema
- Observability signals
  - [ ] Counters for verifications and evictions (`catalog_verifications_total{result}`)
  - [ ] Structured log messages for CRUD and verification outcomes
- ModelRef enhancements
  - [ ] Normalization (lowercasing where applicable, removing trailing slashes)
  - [ ] `Display` implementations with redaction for logs

## Low (nice‑to‑have)

- Provenance attachments
  - [ ] Optional SBOM/signature references on entries (carried through from daemons)
- Integrity checks
  - [ ] Periodic scrubbing job (optional) to re‑hash and validate digests
- Performance
  - [ ] Batch operations on index (list range, prefix search)

## Separation of Concerns (do not pull into this crate)

- Network fetchers (HF/HTTP/S3/OCI) and policy‑gated outbound access → `model‑provisioner`
- Engine‑specific preparation and runtime metadata → `engine‑provisioner` and adapters
- HTTP endpoints → `orchestratord`

## Tests

- [ ] Unit: index round‑trip under concurrent reads/writes (use temp dirs)
- [ ] Unit: `delete` removes files/dirs and index entries (best‑effort semantics)
- [ ] Unit: `ModelRef::parse()` coverage (already present; extend for edge cases)
- [ ] Property: index never corrupts on crash during write (simulate interrupted rename)

## Acceptance Criteria

- [ ] All new APIs documented and tested
- [ ] Clippy/fmt clean
- [ ] Deterministic behavior under concurrent access and crash‑recovery scenarios
- [ ] README updated with High/Mid/Low behavior summary and examples
