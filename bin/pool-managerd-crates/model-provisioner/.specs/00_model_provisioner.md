# model-provisioner — Component Specification (v0)
Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19
## 0) Purpose & Scope
`model-provisioner` ensures models referenced by users or configs are present locally and registered in the catalog. It delegates parsing/indexing to `catalog-core` and supports multiple schemes via fetchers. The default in-tree fetcher handles local files; network fetching is pluggable.
In scope:
- Parsing model refs (via `catalog-core`) and staging artifacts locally.
- Registering/updating `CatalogEntry` with lifecycle state and expected digest.
- Providing a `ResolvedModel { id, local_path }` for downstream consumers.
Out of scope:
- Engine preparation and runtime flags (engine‑provisioner/adapters).
## 1) Normative Requirements (RFC‑2119)
- [ORCH‑3700] The crate MUST expose a generic `ModelProvisioner<C: CatalogStore, F: ModelFetcher>` with:
  - `fn ensure_present_str(&self, model_ref: &str, expected_digest: Option<Digest>) -> Result<ResolvedModel>`
  - `fn ensure_present(&self, mr: &ModelRef, expected_digest: Option<Digest>) -> Result<ResolvedModel>`
- [ORCH‑3701] The default constructor `ModelProvisioner::file_only(cache_dir)` MUST use `FsCatalog` and `FileFetcher` and MUST NOT perform network I/O.
- [ORCH‑3702] On success, the provisioner MUST insert/update a `CatalogEntry` to `Active` with `local_path` set to the staged artifact path and MUST set the expected `digest` when provided.
- [ORCH‑3703] If digest is provided, the provisioner SHOULD verify it and MUST record verification outcome; if absent, it SHOULD log a warning (at call sites) that the artifact is unverified.
- [ORCH‑3704] The provisioner MUST return a stable `id` for the resolved model; for local files, `id` SHOULD be a normalized representation of the path; for `hf:` scheme, `id` SHOULD include org/repo[/path].
- [ORCH‑3705] The crate MUST NOT install packages; optional support for `hf:` scheme MAY shell out to `huggingface-cli` when present, but MUST error with instructions if not available.
## 2) Data Types & Semantics
- `ResolvedModel { id: String, local_path: PathBuf }` — canonical id and absolute path.
- `Digest { algo, value }` — as in `catalog-core`.
## 3) Interfaces & Contracts
- Stability: The two ensure APIs are stable within the repo pre‑1.0; changing semantics requires spec update and call site changes.
- Integration: `engine-provisioner` and `pool-managerd` call `ensure_present*` before starting engines or flipping Ready. Callers SHOULD first consult `catalog-core` read-only helpers `exists(id|ref)` and `locate(ModelRef)` to avoid redundant staging, as promoted in `/.specs/25-catalog-core.md`.
## 4) Observability
- Callers SHOULD emit logs/metrics for staged bytes, duration, and verification outcomes.
## 5) Security
- No secrets stored; do not persist tokens/keys; avoid logging full URLs with tokens.
## 6) Testing & Proof Bundle
- Unit tests SHOULD cover: file staging, id normalization, catalog registration/update, and digest verification plumbing.
- Integration tests SHOULD exercise `hf:` shell‑out path when the CLI is present (feature‑gated) and assert graceful error messaging when absent.
## 7) Open Questions
- Pluggable fetchers for `hf/http/s3/oci` as feature crates?
- Single‑flight locking to avoid duplicate downloads across processes?
## 8) Refinement Opportunities
- Add `exists(id|ref)` integration with the catalog to short‑circuit fetch.
- Parallel fetch of multi‑file repos (Transformers) with integrity checks.
- Provenance capture (SBOM, signature references) for trust pipelines.
