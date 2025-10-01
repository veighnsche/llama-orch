use anyhow::Result;
use catalog_core::ResolvedModel;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

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
        if let Some(dir) = dest.as_ref().parent() {
            fs::create_dir_all(dir)?;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_locks::CWD_LOCK;

    #[test]
    fn metadata_handoff_and_provenance() {
        let _guard = CWD_LOCK.get_or_init(Default::default).lock().unwrap();
        // Isolate CWD for .runtime writes
        let old_cwd = std::env::current_dir().unwrap();
        let wd = tempfile::tempdir().unwrap();
        std::env::set_current_dir(wd.path()).unwrap();

        // Prepare a dummy file and resolved model
        let fpath = wd.path().join("m.gguf");
        fs::write(&fpath, b"x").unwrap();
        let r = ResolvedModel { id: "file".into(), local_path: fpath.clone() };
        let meta = ModelMetadata::from_resolved(&r).unwrap();

        let handoff = wd.path().join("handoff.json");
        meta.write_handoff(&handoff).unwrap();
        assert!(handoff.exists());
        let v: serde_json::Value = serde_json::from_slice(&fs::read(&handoff).unwrap()).unwrap();
        assert!(v.get("model").is_some());

        let cfg = wd.path().join("cfg.yaml");
        fs::write(&cfg, b"model_ref: file:m.gguf\n").unwrap();
        meta.append_provenance(&cfg).unwrap();
        let prov = wd.path().join(".runtime/provenance/models.jsonl");
        assert!(prov.exists());

        std::env::set_current_dir(old_cwd).unwrap();
    }
}
