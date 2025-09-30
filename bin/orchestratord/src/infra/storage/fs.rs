//! Filesystem ArtifactStore (stub)
//! Placeholder; real implementation will manage content-addressed files.

use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::ports::storage::{Artifact, ArtifactId, ArtifactStore};

#[derive(Debug, Clone)]
pub struct FsStore {
    root: PathBuf,
}

impl FsStore {
    pub fn new<P: AsRef<Path>>(root: P) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn path_for(&self, id: &str) -> PathBuf {
        // id is expected to be like "sha256:<hex>"
        let hex = id.strip_prefix("sha256:").unwrap_or(id);
        self.root.join(hex)
    }
}

impl Default for FsStore {
    fn default() -> Self {
        let root = std::env::var("ORCH_ARTIFACTS_FS_ROOT")
            .unwrap_or_else(|_| "/tmp/llorch-artifacts".to_string());
        FsStore::new(root).expect("create fs store root")
    }
}

impl ArtifactStore for FsStore {
    fn put(&self, doc: Artifact) -> anyhow::Result<ArtifactId> {
        let s = doc.to_string();
        let id = sha256::digest(s.clone());  // Pure hash, no prefix
        let path = self.path_for(&id);
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        let mut f = File::create(&path)?;
        f.write_all(s.as_bytes())?;
        Ok(id)
    }

    fn get(&self, id: &ArtifactId) -> anyhow::Result<Option<Artifact>> {
        let path = self.path_for(id);
        if !path.exists() {
            return Ok(None);
        }
        let mut f = File::open(&path)?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        let v: serde_json::Value = serde_json::from_str(&buf)?;
        Ok(Some(v))
    }
}

mod sha256 {
    use sha2::{Digest, Sha256};
    pub fn digest(s: String) -> String {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        let bytes = hasher.finalize();
        hex::encode(bytes)
    }
}
