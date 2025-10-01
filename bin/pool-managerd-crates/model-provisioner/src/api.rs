use crate::{
    config::ModelProvisionerConfig, metadata::ModelMetadata, provisioner::ModelProvisioner,
};
use anyhow::Result;
use std::path::Path;
use tracing::warn;

/// Recommended default handoff path for llama.cpp engines.
pub const DEFAULT_LLAMACPP_HANDOFF_PATH: &str = ".runtime/engines/llamacpp.json";

/// Convenience: provision from a YAML/JSON config and write a handoff file.
/// Returns the `ModelMetadata` for chaining into engine-provisioner.
pub fn provision_from_config_to_handoff<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
    cfg_path: P,
    handoff_dest: Q,
    cache_dir: R,
) -> Result<ModelMetadata> {
    // Normalize cfg_path to an owned PathBuf to avoid move/borrow issues.
    let cfg_path_buf = cfg_path.as_ref().to_path_buf();
    let cfg = ModelProvisionerConfig::from_path(&cfg_path_buf)?;
    let prov = ModelProvisioner::file_only(cache_dir.as_ref().to_path_buf())?;
    let resolved = prov.ensure_present_str(&cfg.model_ref, cfg.expected_digest.clone())?;
    if cfg.strict_verification && cfg.expected_digest.is_none() {
        warn!("strict_verification=true but no expected_digest provided; skipping verification");
    }
    let meta = ModelMetadata::from_resolved(&resolved)?;
    meta.write_handoff(handoff_dest)?;
    meta.append_provenance(&cfg_path_buf)?;
    Ok(meta)
}

/// Convenience wrapper that writes the handoff to the recommended default path.
pub fn provision_from_config_to_default_handoff<P: AsRef<Path>, R: AsRef<Path>>(
    cfg_path: P,
    cache_dir: R,
) -> Result<ModelMetadata> {
    provision_from_config_to_handoff(cfg_path, DEFAULT_LLAMACPP_HANDOFF_PATH, cache_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_locks::CWD_LOCK;

    #[test]
    fn provision_from_config_to_handoff_writes_handoff_and_provenance_yaml_and_json() {
        let _guard = CWD_LOCK.get_or_init(Default::default).lock().unwrap();
        // Isolate .runtime writes
        let old_cwd = std::env::current_dir().unwrap();
        let wd = tempfile::tempdir().unwrap();
        std::env::set_current_dir(wd.path()).unwrap();

        let cache = tempfile::tempdir().unwrap();
        // Create a dummy model file
        let model_path = wd.path().join("tiny.gguf");
        std::fs::write(&model_path, b"dummy-model").unwrap();

        // YAML config
        let yaml_cfg = wd.path().join("model.yaml");
        std::fs::write(
            &yaml_cfg,
            format!("model_ref: \"file:{}\"\nstrict_verification: false\n", model_path.display()),
        )
        .unwrap();

        let handoff_yaml = wd.path().join("handoff.yaml.json");
        let meta =
            provision_from_config_to_handoff(&yaml_cfg, &handoff_yaml, cache.path()).unwrap();
        assert_eq!(meta.path, model_path);
        // Handoff exists and contains expected fields
        let handoff_val: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&handoff_yaml).unwrap()).unwrap();
        assert!(handoff_val.get("model").is_some());
        assert!(handoff_val.get("metadata").is_some());

        // Provenance JSONL contains the cfg path
        let prov_path = wd.path().join(".runtime/provenance/models.jsonl");
        assert!(prov_path.exists());
        let prov_lines = std::fs::read_to_string(&prov_path).unwrap();
        let last_line = prov_lines.lines().last().unwrap();
        let prov_val: serde_json::Value = serde_json::from_str(last_line).unwrap();
        assert_eq!(prov_val["cfg"], serde_json::Value::String(yaml_cfg.display().to_string()));

        // JSON config
        let json_cfg = wd.path().join("model.json");
        let json_body = serde_json::json!({
            "model_ref": format!("file:{}", model_path.display()),
            "strict_verification": false
        });
        std::fs::write(&json_cfg, serde_json::to_vec_pretty(&json_body).unwrap()).unwrap();
        let handoff_json = wd.path().join("handoff.json.json");
        let meta2 =
            provision_from_config_to_handoff(&json_cfg, &handoff_json, cache.path()).unwrap();
        assert_eq!(meta2.path, model_path);

        // Cleanup: restore CWD
        std::env::set_current_dir(old_cwd).unwrap();
    }
}
