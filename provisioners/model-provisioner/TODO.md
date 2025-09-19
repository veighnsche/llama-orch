# model-provisioner — Exhaustive TODO

This provisioner is responsible for ensuring model artifacts referenced by `ModelRef` are present locally, verified, and registered in the catalog, so engines can consume them deterministically. It builds on `catalog-core/` for parsing, indexing, lifecycle, and verification.

References:
- `.specs/00_llama-orch.md` §2.6 Catalog, §2.11 Model Selection & Auto-Fetch, §2.12 Engine Provisioning & Preflight
- `.specs/71-metrics-contract.md`, `.specs/metrics/otel-prom.md`
- Engines: `worker-adapters/llamacpp-http`, `worker-adapters/vllm-http`, `worker-adapters/tgi-http`, `worker-adapters/triton`

Principles:
- Determinism-first: local cache is keyed by content (digests) and stable IDs.
- Policy-gated IO: outbound network/tooling must honor allowlists and disable switches.
- Additive-only user-visible data: keep catalog/index forward compatible.
- Minimal privileges: no package installs here (engine/tooling installs belong to engine-provisioner).

Cross-cutting work items
- [ ] High-level `ModelProvisioner` interface
  - [ ] `ensure_present(ModelRef, VerificationSpec) -> ResolvedModel`
  - [ ] `evict(id|glob)` (optional), `gc()` refcounts and orphan sweeps
  - [ ] `exists(ModelRef) -> bool`, `locate(ModelRef) -> Option<ResolvedModel>`
- [ ] ModelRef schemes mapping (ORCH-3090..3092)
  - [ ] hf:org/repo/path.gguf (llama.cpp)
  - [ ] hf:org/repo (vLLM/TGI)
  - [ ] file: and relative paths
  - [ ] https:// URL
  - [ ] s3:// bucket/key
  - [ ] oci:// registry/repo[:tag]
- [ ] Concurrency & single-flight
  - [ ] Per-model ID lock to avoid duplicate downloads/builds
  - [ ] Global backpressure (max parallel fetches)
- [ ] Verification (ORCH-3037)
  - [ ] Required when digest provided (fail on mismatch)
  - [ ] Advisory when missing (warn, log audit)
  - [ ] SBOM/signature hooks (optional)
- [ ] Catalog registration
  - [ ] Create/update entry with lifecycle `Active` by default
  - [ ] `last_verified_ms` bookkeeping
  - [ ] `model_state` metric updates (optional now)
- [ ] Storage layout
  - [ ] Default: `~/.cache/models` (configurable)
  - [ ] Layout rules per engine (see below)
  - [ ] Index strategy (reuse `catalog-core` FsCatalog index.json)
- [ ] Observability & metrics
  - [ ] `catalog_verifications_total{result,reason}`
  - [ ] Logs: `model_id`, `model_ref`, `digest`, `bytes`, `duration_ms`, `policy_label`, `correlation_id`
- [ ] Policy gates (ORCH-3080, ORCH-3206)
  - [ ] Allow/deny outbound network per domain/scheme
  - [ ] Offline mode honored
- [ ] CLI integration (future)
  - [ ] `orchctl model ensure <ref>`
  - [ ] `orchctl model ls`, `rm`, `verify` (advisory), `gc`

Per-engine provisioners (each needs a dedicated path/flow)

## llama.cpp (GGUF)
- High (must)
  - [ ] Support `hf:org/repo/path.gguf` and `file:` refs
  - [ ] Verify digest when provided; warn otherwise
  - [ ] Place artifact under `<cache>/gguf/<org>/<repo>/<file>@<digest-or-sha>`
  - [ ] Concurrency-safe ensure-present
- Mid (should)
  - [ ] Optional `https://` direct GGUF URL
  - [ ] Map legacy `ggml` extensions to `gguf` when safe
  - [ ] Partial resume for interrupted downloads
- Low (nice-to-have)
  - [ ] Mirror/alt registry fallback
  - [ ] Local de-duplication (hard links) when multiple IDs point to same digest

Open items
- [ ] Very small test model path for CI (no large downloads)

## vLLM (Transformers repo)
- High (must)
  - [ ] Support `hf:org/repo` (no file path)
  - [ ] Clone/resolve minimal files: `config.json`, tokenizer files, safetensors (possibly sharded)
  - [ ] Place under `<cache>/transformers/<org>/<repo>@<revision-or-digest>`
  - [ ] Digest map: record set of file digests to compute repo digest (stable ID)
- Mid (should)
  - [ ] Handle sharded safetensors (`model-00001-of-000xx.safetensors`)
  - [ ] Trust settings: `allow_remote_code=false` by default, warn if requested
  - [ ] LFS-aware downloads
- Low (nice-to-have)
  - [ ] Convert repo revision/tag/commit to pinned revision for reproducibility
  - [ ] Thin clones (sparse checkout) to reduce bytes

Open items
- [ ] Test harness: tiny text-classification model or mocked endpoints

## HF TGI (Transformers repo)
- High (must)
  - [ ] Same as vLLM basic repo layout resolution
  - [ ] Ensure tokenizer/config parity with engine expectations
- Mid (should)
  - [ ] Optional quantized variants (if advertised)
  - [ ] Backward-compatible tokenizers (merges/BPE) normalization
- Low (nice-to-have)
  - [ ] Repo filtering for only files used by TGI runtime

Open items
- [ ] Cross-engine repo reuse (vLLM/TGI) without duplication in the cache

## NVIDIA Triton / TensorRT-LLM
- High (must)
  - [ ] Support `s3://` and `oci://` model repositories
  - [ ] Layout detection: `config.pbtxt`, `model.onnx`/`plan`, ensemble layouts
  - [ ] Place under `<cache>/triton/<org-or-bucket>/<repo-or-path>@<digest-or-tag>`
- Mid (should)
  - [ ] OCI registry authentication and digest pinning; verify image layer digests
  - [ ] S3 pre-signed URL support; region selection; retries with backoff
- Low (nice-to-have)
  - [ ] Optional conversion pipeline hooks (NOTE: likely belong to engine-provisioner)

Open items
- [ ] Tiny test repo fixture (local file-based triton repo for CI)


Integration points
- [ ] `orchestratord`: catalog endpoints call into model-provisioner (ensure, verify, lifecycle)
- [ ] `pool-managerd`: preload path calls ensure-present before Ready
- [ ] `engine-provisioner`: depends on model-provisioner to stage the model prior to engine up


API sketch (internal)
```rust
pub struct ModelProvisioner<C: CatalogStore, F: ModelFetcher> {
    catalog: C,
    fetcher: F,
}

impl<C, F> ModelProvisioner<C, F>
where
    C: CatalogStore,
    F: ModelFetcher,
{
    pub fn ensure_present(&self, mr: &ModelRef, expect: Option<Digest>) -> Result<ResolvedModel>;
    pub fn evict(&self, id: &str) -> Result<()>;
    pub fn gc(&self) -> Result<GcReport>;
}
```

Security & policy
- [ ] No package installations here; engine/tooling installs belong to `engine-provisioner`.
- [ ] Respect outbound network policy via allowlists/denylists; offline mode if disabled.
- [ ] Keep audit logs with correlation ID and bytes transferred.

Proof plan
- [ ] Unit tests per engine flow (mocked network)
- [ ] BDD: catalog ensure + pool reload exposes Active state
- [ ] Metrics lints green (emit signals where applicable)
- [ ] OpenAPI `x-examples` show model ensure flows through control-plane endpoints
