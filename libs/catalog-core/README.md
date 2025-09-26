# catalog-core — Model Catalog, Resolution & Verification

## 1. Name & Purpose

`catalog-core` centralizes model reference parsing, local catalog/indexing, lifecycle (Active/Retired), and digest verification for llama-orch. It is engine-agnostic and shared by:

- `orchestratord` for catalog HTTP endpoints and reload/drain flows
- `pool-managerd` to ensure models are present before marking pools Ready
- `provisioners/engine-provisioner` and `provisioners/model-provisioner` to stage artifacts

This crate does not install packages and does not spawn processes.

## 2. Spec Traceability

- Catalog & artifacts: `.specs/00_llama-orch.md` (§2.6 Catalog, §2.11 Model Selection & Auto‑Fetch)
- Provisioning separation of concerns: `.specs/00_llama-orch.md` (§2.12 Engine Provisioning & Preflight)
- Metrics contracts (future emission sites live in daemons): `.specs/71-metrics-contract.md`, `.specs/metrics/otel-prom.md`

## 3. Public API Surface (current)

- `ModelRef` — user‑supplied model reference (parse from string):
  - `hf:org/repo[/path]`, `file:/abs/path` or relative path, `http(s)://`, `s3://`, `oci://`
  - Current implementation: parsing only; fetching for `file:` and relative paths is supported via `FileFetcher`.
- `CatalogStore` trait — storage/indexing interface:
  - `get`, `put`, `set_state`, `list`, `delete`
  - `FsCatalog` implementation: JSON index at a root directory
- `ModelFetcher` trait — ensure artifacts are present locally:
  - `FileFetcher` implementation for local files (absolute/relative). Networked fetchers are not in this crate yet.
- `verify_digest(actual, expected)` — pass/warn/fail outcome
- `default_model_cache_dir()` — `~/.cache/models`

## 4. Data Types

```rust
pub enum ModelRef {
    Hf { org: String, repo: String, path: Option<String> },
    File { path: PathBuf },
    Url  { url: String },
}

pub enum LifecycleState { Active, Retired }

pub struct CatalogEntry {
    pub id: String,
    pub local_path: PathBuf,
    pub lifecycle: LifecycleState,
    pub digest: Option<Digest>,
    pub last_verified_ms: Option<u64>, // unix epoch millis
}

pub trait CatalogStore {
    fn get(&self, id: &str) -> Result<Option<CatalogEntry>>;
    fn put(&self, entry: &CatalogEntry) -> Result<()>;
    fn set_state(&self, id: &str, state: LifecycleState) -> Result<()>;
    fn list(&self) -> Result<Vec<CatalogEntry>>;
    fn delete(&self, id: &str) -> Result<bool>;
}
```

## 5. Current Capabilities vs Planned

- High (implemented)
  - `ModelRef::parse()` covering `hf:`, `file:`, relative path, and general URL schemes (parse only)
  - Filesystem catalog with JSON index: `FsCatalog`
  - Local file ensure‑present via `FileFetcher`
  - Digest verification helper with pass/warn/fail semantics
- Mid (planned, not in this crate yet)
  - Network fetchers: Hugging Face Hub, HTTP(S), S3, OCI
  - Single‑flight/concurrency locks around ensure‑present
  - Eviction and GC policies (content‑addressable layout)
- Low (future)
  - SBOM/signature verification hooks
  - Extended provenance metadata (container digests, build flags), emitted by daemons

## 6. Example

```rust
use catalog_core::{FsCatalog, CatalogStore, FileFetcher, ModelFetcher, ModelRef, LifecycleState};

fn demo() -> anyhow::Result<()> {
    // Create a catalog rooted at ~/.cache/models
    let cat = FsCatalog::new(catalog_core::default_model_cache_dir())?;

    // Parse a local model reference (absolute or relative)
    let mr = ModelRef::parse("./models/llama3-8b.gguf")?;

    // Ensure the model is present locally (file fetcher only)
    let fetcher = FileFetcher;
    let resolved = fetcher.ensure_present(&mr)?;

    // Register in the catalog
    let entry = catalog_core::CatalogEntry {
        id: resolved.id.clone(),
        local_path: resolved.local_path.clone(),
        lifecycle: LifecycleState::Active,
        digest: None,
        last_verified_ms: None,
    };
    cat.put(&entry)?;

    Ok(())
}
```

## 7. Build & Test

- Format & lint (workspace): `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features -- -D warnings`
- Tests: `cargo test -p catalog-core`

## 8. Security & Policy

- No package installation and no process execution here
- Outbound network access for model downloads is delegated to provisioners and daemons, behind policy gates

## 9. Notes on Separation of Concerns

- `catalog-core` owns parsing, indexing, lifecycle, and digest verification
- `model-provisioner` orchestrates cross‑scheme fetching and registers into the catalog
- `engine-provisioner` prepares engines and depends on `model-provisioner` for models
- `orchestratord` exposes catalog CRUD endpoints backed by this crate

## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## Detailed behavior (High / Mid / Low)

- High-level
  - Provides `ModelRef` parsing, a `CatalogStore` trait with a default `FsCatalog` implementation, and helpers like `verify_digest` and `default_model_cache_dir()`. It does not perform network fetches or spawn processes; provisioners own those.

- Mid-level
  - Entries persisted by `FsCatalog` include `id`, `local_path`, `lifecycle`, and optional `digest/last_verified_ms`. Read-only helpers `exists(id|ref)` and `locate(ModelRef)` are promoted to normative behavior to avoid redundant staging in callers.

- Low-level
  - Index writes are atomic (temp file + atomic rename). Paths are normalized and must remain under the catalog root. `delete(id)` removes the index entry and best-effort deletes artifacts without corrupting the index.
