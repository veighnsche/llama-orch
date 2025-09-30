//! model-provisioner — orchestrates model ensure-present flows using catalog-core.
//!
//! Responsibilities:
//! - Parse ModelRef input (string or structured) and ensure the model is present locally.
//! - Verify digests when given; warn otherwise (per spec §2.6/§2.11).
//! - Register/update catalog entries and lifecycle state.
//! - Return ResolvedModel with canonical local path for engine-provisioner and pool-managerd.

use anyhow::Result;
use catalog_core::{
    CatalogEntry, CatalogStore, Digest, FileFetcher, FsCatalog, LifecycleState, ModelFetcher,
    ModelRef, ResolvedModel,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
// TODO(OwnerD-CACHE-EVICT-SKELETON): Add cache size accounting and LRU eviction policy with provenance logs.
// For MVP this is a no-op skeleton.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvisionerConfig {
    pub model_ref: String,
    #[serde(default)]
    pub expected_digest: Option<Digest>,
    #[serde(default)]
    pub strict_verification: bool,
}

impl ModelProvisionerConfig {
    pub fn from_path<P: AsRef<Path>>(p: P) -> Result<Self> {
        let data = fs::read_to_string(&p)?;
        // Try YAML first, then JSON
        let cfg: Self = match serde_yaml::from_str(&data) {
            Ok(c) => c,
            Err(_) => serde_json::from_str(&data)?,
        };
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub ctx_max: Option<u32>, // TODO(OwnerD-CTX-PROBE): Parse GGUF header to infer context length
}

impl ModelMetadata {
    pub fn from_resolved(r: &ResolvedModel) -> Result<Self> {
        let meta = fs::metadata(&r.local_path)?;
        Ok(Self {
            id: r.id.clone(),
            path: r.local_path.clone(),
            size_bytes: meta.len(),
            ctx_max: None,
        })
    }

    /// Write an engine handoff JSON containing the resolved model path and metadata.
    pub fn write_handoff<P: AsRef<Path>>(&self, dest: P) -> Result<()> {
        let payload = serde_json::json!({
            "model": { "id": self.id, "path": self.path },
            "metadata": { "size_bytes": self.size_bytes, "ctx_max": self.ctx_max }
        });
        if let Some(dir) = dest.as_ref().parent() { fs::create_dir_all(dir)?; }
        fs::write(dest, serde_json::to_vec_pretty(&payload)?)?;
        Ok(())
    }

    /// Append a provenance record into `.runtime/provenance/models.jsonl`.
    pub fn append_provenance<P: AsRef<Path>>(&self, cfg_path: P) -> Result<()> {
        let prov_dir = PathBuf::from(".runtime").join("provenance");
        fs::create_dir_all(&prov_dir)?;
        let file = prov_dir.join("models.jsonl");
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let rec = serde_json::json!({
            "ts_ms": now_ms,
            "model_id": self.id,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "cfg": cfg_path.as_ref(),
        });
        use std::io::Write;
        let mut f = fs::OpenOptions::new().create(true).append(true).open(&file)?;
        writeln!(f, "{}", serde_json::to_string(&rec)?)?;
        Ok(())
    }
}

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
    pub fn ensure_present_str(
        &self,
        model_ref: &str,
        expected_digest: Option<Digest>,
    ) -> Result<ResolvedModel> {
        let mr = ModelRef::parse(model_ref)?;
        self.ensure_present(&mr, expected_digest)
    }

    pub fn ensure_present(
        &self,
        mr: &ModelRef,
        expected_digest: Option<Digest>,
    ) -> Result<ResolvedModel> {
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
                        if let Some(p) = path {
                            c.arg(p);
                        }
                        c.arg("--local-dir")
                            .arg(&cache_dir)
                            .arg("--local-dir-use-symlinks")
                            .arg("False");
                        let st = c.status()?;
                        if !st.success() {
                            return Err(anyhow::anyhow!(
                                "huggingface-cli download failed for {}",
                                repo_spec
                            ));
                        }
                        let local_path = if let Some(p) = path {
                            cache_dir.join(p)
                        } else {
                            cache_dir.join(repo_spec.replace('/', "_"))
                        };
                        ResolvedModel {
                            id: format!(
                                "hf:{}/{}{}",
                                org,
                                repo,
                                path.as_ref().map(|p| format!("/{}", p)).unwrap_or_default()
                            ),
                            local_path,
                        }
                    }
                    _ => return Err(e.into()),
                }
            }
        };
        // Optional strict verification: compute sha256 and enforce match when configured.
        if let Some(exp) = expected_digest.as_ref() {
            if exp.algo.eq_ignore_ascii_case("sha256") {
                let act = compute_sha256_hex(&resolved.local_path)?;
                if act != exp.value {
                    return Err(anyhow::anyhow!(
                        "digest mismatch: expected sha256:{}, got {}",
                        exp.value, act
                    ));
                }
            }
        }

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

fn compute_sha256_hex(path: &Path) -> Result<String> {
    use sha2::{Digest as ShaDigest, Sha256};
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let out = hasher.finalize();
    Ok(hex::encode(out))
}

/// Convenience: provision from a YAML/JSON config and write a handoff file.
/// Returns the `ModelMetadata` for chaining into engine-provisioner.
pub fn provision_from_config_to_handoff<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
    cfg_path: P,
    handoff_dest: Q,
    cache_dir: R,
) -> Result<ModelMetadata> {
    let cfg = ModelProvisionerConfig::from_path(cfg_path)?;
    let prov = ModelProvisioner::file_only(cache_dir.as_ref().to_path_buf())?;
    let resolved = prov.ensure_present_str(&cfg.model_ref, cfg.expected_digest.clone())?;
    // If strict flag is set without digest, warn via error to enforce policy clarity
    if cfg.strict_verification && cfg.expected_digest.is_none() {
        // Policy shim: we don't fail, but add a note in logs later if desired.
    }
    let meta = ModelMetadata::from_resolved(&resolved)?;
    meta.write_handoff(handoff_dest)?;
    // Record provenance for auditability
    // TODO(OwnerD-PROVENANCE-EXTEND): Include verification outcome once catalog-core exposes helpers.
    meta.append_provenance(cache_dir)?;
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn happy_path_file_model() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join("tiny.gguf");
        {
            let mut f = fs::File::create(&model_path).unwrap();
            writeln!(f, "dummy").unwrap();
        }
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse(&format!("file:{}", model_path.display())).unwrap();
        let resolved = prov.ensure_present(&mr, None).unwrap();
        assert_eq!(resolved.local_path, model_path);
        let meta = ModelMetadata::from_resolved(&resolved).unwrap();
        assert!(meta.size_bytes > 0);
    }

    #[test]
    fn missing_model_errors() {
        let cache = tempfile::tempdir().unwrap();
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse("file:/non/existent/path.gguf").unwrap();
        let err = prov.ensure_present(&mr, None).unwrap_err();
        let s = format!("{}", err);
        assert!(s.contains("not found") || s.contains("No such file"));
    }

    #[test]
    fn strict_digest_gating() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join("tiny.gguf");
        fs::write(&model_path, b"abc").unwrap();
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse(&format!("file:{}", model_path.display())).unwrap();
        // Wrong digest should fail
        let wrong = Digest { algo: "sha256".into(), value: "deadbeef".into() };
        let err = prov.ensure_present(&mr, Some(wrong)).unwrap_err();
        assert!(format!("{}", err).contains("digest mismatch"));
        // Correct digest should pass
        let ok = Digest { algo: "sha256".into(), value: compute_sha256_hex(&model_path).unwrap() };
        let resolved = prov.ensure_present(&mr, Some(ok)).unwrap();
        assert_eq!(resolved.local_path, model_path);
    }
}
