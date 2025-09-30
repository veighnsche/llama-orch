use std::collections::HashMap;
use std::sync::Mutex;

use crate::ports::storage::{Artifact, ArtifactId, ArtifactStore};

#[derive(Debug, Default)]
pub struct InMemStore {
    inner: Mutex<HashMap<ArtifactId, Artifact>>,
}

impl ArtifactStore for InMemStore {
    fn put(&self, doc: Artifact) -> anyhow::Result<ArtifactId> {
        // Content-address via SHA-256 to match spec and fs backend behavior
        let id = format!("sha256:{}", sha256::digest(doc.to_string()));
        self.inner.lock().unwrap().insert(id.clone(), doc);
        Ok(id)
    }
    fn get(&self, id: &ArtifactId) -> anyhow::Result<Option<Artifact>> {
        Ok(self.inner.lock().unwrap().get(id).cloned())
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
