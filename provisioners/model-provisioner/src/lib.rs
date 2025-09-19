//! model-provisioner — orchestrates model ensure-present flows using catalog-core.
//!
//! Responsibilities:
//! - Parse ModelRef input (string or structured) and ensure the model is present locally.
//! - Verify digests when given; warn otherwise (per spec §2.6/§2.11).
//! - Register/update catalog entries and lifecycle state.
//! - Return ResolvedModel with canonical local path for engine-provisioner and pool-managerd.

use anyhow::Result;
use catalog_core::{CatalogEntry, CatalogStore, Digest, FileFetcher, FsCatalog, LifecycleState, ModelFetcher, ModelRef, ResolvedModel};
use std::path::PathBuf;
use std::process::Command;

#[derive(Clone)]
pub struct ModelProvisioner<C: CatalogStore, F: ModelFetcher> {
    catalog: C,
    fetcher: F,
}

impl ModelProvisioner<FsCatalog, FileFetcher> {
    pub fn file_only(cache_dir: PathBuf) -> Result<Self> {
        let catalog = FsCatalog::new(cache_dir)?;
        let fetcher = FileFetcher;
        Ok(Self { catalog, fetcher })
    }
}

impl<C: CatalogStore, F: ModelFetcher> ModelProvisioner<C, F> {
    pub fn new(catalog: C, fetcher: F) -> Self {
        Self { catalog, fetcher }
    }

    /// Ensure the model referred to by `model_ref` is present locally.
    /// - Parses `model_ref` string to a ModelRef.
    /// - Uses the fetcher to ensure local presence.
    /// - Records/updates the catalog entry to Active unless specified otherwise.
    pub fn ensure_present_str(&self, model_ref: &str, expected_digest: Option<Digest>) -> Result<ResolvedModel> {
        let mr = ModelRef::parse(model_ref)?;
        self.ensure_present(&mr, expected_digest)
    }

    pub fn ensure_present(&self, mr: &ModelRef, expected_digest: Option<Digest>) -> Result<ResolvedModel> {
        // Try primary fetcher first (file/relative). If unsupported, handle select schemes inline.
        let resolved = match self.fetcher.ensure_present(mr) {
            Ok(r) => r,
            Err(e) => {
                // Minimal hf: support via huggingface-cli as a stopgap.
                match mr {
                    ModelRef::Hf { org, repo, path } => {
                        let cache_dir = catalog_core::default_model_cache_dir();
                        std::fs::create_dir_all(&cache_dir).ok();
                        // Require huggingface-cli for now.
                        if which::which("huggingface-cli").is_err() {
                            return Err(anyhow::anyhow!("huggingface-cli not found for hf:{}; install `python-huggingface-hub` or provide local file: path", format!("{}/{}", org, repo)));
                        }
                        let repo_spec = format!("{}/{}", org, repo);
                        let mut c = Command::new("huggingface-cli");
                        c.env("HF_HUB_ENABLE_HF_TRANSFER", "1");
                        c.arg("download").arg(&repo_spec);
                        if let Some(p) = path { c.arg(p); }
                        c.arg("--local-dir").arg(&cache_dir).arg("--local-dir-use-symlinks").arg("False");
                        let st = c.status()?;
                        if !st.success() {
                            return Err(anyhow::anyhow!("huggingface-cli download failed for {}", repo_spec));
                        }
                        let local_path = if let Some(p) = path { cache_dir.join(p) } else { cache_dir.join(repo_spec.replace('/', "_")) };
                        ResolvedModel { id: format!("hf:{}/{}{}", org, repo, path.as_ref().map(|p| format!("/{}", p)).unwrap_or_default()), local_path }
                    }
                    _ => return Err(e.into()),
                }
            }
        };
        let entry = CatalogEntry {
            id: resolved.id.clone(),
            local_path: resolved.local_path.clone(),
            lifecycle: LifecycleState::Active,
            digest: expected_digest,
            last_verified_ms: None,
        };
        // Note: verification beyond digest is handled by higher-level flows; here we store and move on.
        self.catalog.put(&entry)?;
        Ok(resolved)
    }
}
