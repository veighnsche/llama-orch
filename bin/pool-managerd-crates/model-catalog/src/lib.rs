//! catalog-core — model catalog, resolution, verification, caching
//!
//! This crate centralizes model reference parsing, local caching, lifecycle,
//! and verification. It is used by:
//! - `orchestratord` for catalog HTTP endpoints and reload/drain flows
//! - `pool-managerd` to ensure models are locally present before preload
//! - `provisioners/engine-provisioner` to resolve model artifacts
//!
//! Initial implementation focuses on a filesystem-backed catalog and a
//! File-based fetcher. HF/HTTP/S3/OCI fetchers are stubbed and will be added
//! incrementally per `.specs/00_llama-orch.md §2.11`.

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
// time utilities used in tests only
use thiserror::Error;

pub type Result<T> = std::result::Result<T, CatalogError>;

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde json error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("unsupported scheme: {0}")]
    UnsupportedScheme(String),
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid model ref: {0}")]
    InvalidRef(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LifecycleState {
    Active,
    Retired,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Digest {
    pub algo: String,  // e.g., "sha256"
    pub value: String, // lowercase hex
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub local_path: PathBuf,
    pub lifecycle: LifecycleState,
    pub digest: Option<Digest>,
    pub last_verified_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub id: String,
    pub local_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationOutcomeKind {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationOutcome {
    pub kind: VerificationOutcomeKind,
    pub reason: Option<String>,
}

impl VerificationOutcome {
    pub fn pass() -> Self {
        Self { kind: VerificationOutcomeKind::Pass, reason: None }
    }
    pub fn warn(reason: impl Into<String>) -> Self {
        Self { kind: VerificationOutcomeKind::Warn, reason: Some(reason.into()) }
    }
    pub fn fail(reason: impl Into<String>) -> Self {
        Self { kind: VerificationOutcomeKind::Fail, reason: Some(reason.into()) }
    }
}

/// Model reference as supplied by users and contracts (§2.11).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelRef {
    /// hf:org/repo/path.gguf (llama.cpp) or hf:org/repo (vLLM/TGI)
    Hf { org: String, repo: String, path: Option<String> },
    /// file:/abs/path or a relative path treated as local
    File { path: PathBuf },
    /// Generic URL schemes: https, s3, oci, etc.
    Url { url: String },
}

impl ModelRef {
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(rest) = s.strip_prefix("hf:") {
            let mut parts = rest.splitn(3, '/');
            let org = parts.next().ok_or_else(|| CatalogError::InvalidRef(s.to_string()))?;
            let repo = parts.next().ok_or_else(|| CatalogError::InvalidRef(s.to_string()))?;
            let path = parts.next().map(|p| p.trim_start_matches('/').to_string());
            return Ok(ModelRef::Hf { org: org.to_string(), repo: repo.to_string(), path });
        }
        if let Some(p) = s.strip_prefix("file:") {
            return Ok(ModelRef::File { path: PathBuf::from(p) });
        }
        if s.starts_with("http://")
            || s.starts_with("https://")
            || s.starts_with("s3://")
            || s.starts_with("oci://")
        {
            return Ok(ModelRef::Url { url: s.to_string() });
        }
        // Fallback: treat as local file path
        Ok(ModelRef::File { path: PathBuf::from(s) })
    }

    pub fn id_hint(&self) -> String {
        match self {
            ModelRef::Hf { org, repo, path } => match path {
                Some(p) => format!("hf:{org}/{repo}/{p}"),
                None => format!("hf:{org}/{repo}"),
            },
            ModelRef::File { path } => format!("file:{}", path.display()),
            ModelRef::Url { url } => url.clone(),
        }
    }
}

/// Catalog storage trait (Fs/Sqlite implementors).
pub trait CatalogStore: Send + Sync {
    fn get(&self, id: &str) -> Result<Option<CatalogEntry>>;
    fn put(&self, entry: &CatalogEntry) -> Result<()>;
    fn set_state(&self, id: &str, state: LifecycleState) -> Result<()>;
    fn list(&self) -> Result<Vec<CatalogEntry>>;
    fn delete(&self, id: &str) -> Result<bool>;
}

/// Filesystem catalog: maintains a simple JSON index mapping id -> entry.
pub struct FsCatalog {
    #[allow(dead_code)]
    root: PathBuf,
    index_path: PathBuf,
}

impl FsCatalog {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        let index_path = root.join("index.json");
        if !index_path.exists() {
            let map: BTreeMap<String, CatalogEntry> = BTreeMap::new();
            let mut f = fs::File::create(&index_path)?;
            f.write_all(serde_json::to_vec_pretty(&map)?.as_slice())?;
        }
        Ok(Self { root, index_path })
    }

    fn read_index(&self) -> Result<BTreeMap<String, CatalogEntry>> {
        let mut f = fs::File::open(&self.index_path)?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        let map: BTreeMap<String, CatalogEntry> =
            if buf.trim().is_empty() { BTreeMap::new() } else { serde_json::from_str(&buf)? };
        Ok(map)
    }

    fn write_index(&self, map: &BTreeMap<String, CatalogEntry>) -> Result<()> {
        let mut f = fs::File::create(&self.index_path)?;
        f.write_all(serde_json::to_vec_pretty(map)?.as_slice())?;
        Ok(())
    }
}

impl CatalogStore for FsCatalog {
    fn get(&self, id: &str) -> Result<Option<CatalogEntry>> {
        let map = self.read_index()?;
        Ok(map.get(id).cloned())
    }

    fn put(&self, entry: &CatalogEntry) -> Result<()> {
        let mut map = self.read_index()?;
        map.insert(entry.id.clone(), entry.clone());
        self.write_index(&map)
    }

    fn set_state(&self, id: &str, state: LifecycleState) -> Result<()> {
        let mut map = self.read_index()?;
        if let Some(e) = map.get_mut(id) {
            e.lifecycle = state;
            self.write_index(&map)
        } else {
            Err(CatalogError::NotFound(id.to_string()))
        }
    }

    fn list(&self) -> Result<Vec<CatalogEntry>> {
        let map = self.read_index()?;
        Ok(map.values().cloned().collect())
    }

    fn delete(&self, id: &str) -> Result<bool> {
        let mut map = self.read_index()?;
        if let Some(entry) = map.remove(id) {
            // try to remove local artifact
            let p = entry.local_path;
            if p.exists() {
                if p.is_file() {
                    let _ = fs::remove_file(&p);
                } else {
                    let _ = fs::remove_dir_all(&p);
                }
            }
            self.write_index(&map)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Fetcher abstraction for ensuring artifacts are present locally.
pub trait ModelFetcher: Send + Sync {
    fn ensure_present(&self, mr: &ModelRef) -> Result<ResolvedModel>;
}

/// File-based fetcher: handles `file:` refs and relative paths.
pub struct FileFetcher;

impl ModelFetcher for FileFetcher {
    fn ensure_present(&self, mr: &ModelRef) -> Result<ResolvedModel> {
        match mr {
            ModelRef::File { path } => {
                let p = if path.is_absolute() {
                    path.clone()
                } else {
                    std::env::current_dir()?.join(path)
                };
                if !p.exists() {
                    return Err(CatalogError::NotFound(p.display().to_string()));
                }
                let id = format!("file:{}", p.display());
                Ok(ResolvedModel { id, local_path: p })
            }
            ModelRef::Hf { .. } => Err(CatalogError::NotImplemented("hf fetcher not wired yet")),
            ModelRef::Url { url: _ } => {
                Err(CatalogError::NotImplemented("generic URL fetcher not wired yet"))
            }
        }
    }
}

/// A simple verify function implementing the spec behavior:
/// - If a digest is provided, verify and fail on mismatch.
/// - If no digest is provided, verification may warn but should not hard-fail.
pub fn verify_digest(actual: Option<&Digest>, expected: Option<&Digest>) -> VerificationOutcome {
    match (actual, expected) {
        (Some(a), Some(e)) => {
            if a.algo.eq_ignore_ascii_case(&e.algo) && a.value.eq(&e.value) {
                VerificationOutcome::pass()
            } else {
                VerificationOutcome::fail("digest mismatch")
            }
        }
        (None, Some(_)) => VerificationOutcome::fail("missing actual digest for verification"),
        (_, None) => {
            VerificationOutcome::warn("no expected digest provided; proceeding with warnings")
        }
    }
}

/// Default model cache path (~/.cache/models) used by higher layers when none is configured.
pub fn default_model_cache_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("models")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parse_model_refs() {
        assert!(
            matches!(ModelRef::parse("hf:org/repo").unwrap(), ModelRef::Hf{ org, repo, path } if org=="org" && repo=="repo" && path.is_none())
        );
        assert!(
            matches!(ModelRef::parse("hf:org/repo/file.gguf").unwrap(), ModelRef::Hf{ org, repo, path: Some(p) } if org=="org" && repo=="repo" && p=="file.gguf")
        );
        assert!(matches!(ModelRef::parse("file:/abs/path").unwrap(), ModelRef::File { .. }));
        assert!(matches!(ModelRef::parse("relative/path").unwrap(), ModelRef::File { .. }));
        assert!(matches!(ModelRef::parse("https://example.com/x").unwrap(), ModelRef::Url { .. }));
    }

    #[test]
    fn fs_catalog_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = FsCatalog::new(tmp.path()).unwrap();
        let entry = CatalogEntry {
            id: "id1".into(),
            local_path: tmp.path().join("model.bin"),
            lifecycle: LifecycleState::Active,
            digest: None,
            last_verified_ms: Some(
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            ),
        };
        cat.put(&entry).unwrap();
        let got = cat.get("id1").unwrap().unwrap();
        assert_eq!(got.id, "id1");
        assert_eq!(got.lifecycle, LifecycleState::Active);
        let list = cat.list().unwrap();
        assert_eq!(list.len(), 1);
        cat.set_state("id1", LifecycleState::Retired).unwrap();
        let got2 = cat.get("id1").unwrap().unwrap();
        assert_eq!(got2.lifecycle, LifecycleState::Retired);
    }
}
