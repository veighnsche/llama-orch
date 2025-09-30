use anyhow::Result;
use catalog_core::Digest;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_yaml_and_json_config() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = tmp.path().join("m.yaml");
        fs::write(
            &yaml,
            "model_ref: file:/x.gguf\nstrict_verification: false\n",
        )
        .unwrap();
        let c1 = ModelProvisionerConfig::from_path(&yaml).unwrap();
        assert_eq!(c1.model_ref, "file:/x.gguf");
        assert!(!c1.strict_verification);

        let json = tmp.path().join("m.json");
        let body = serde_json::json!({"model_ref":"file:/y.gguf","strict_verification":true});
        fs::write(&json, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        let c2 = ModelProvisionerConfig::from_path(&json).unwrap();
        assert_eq!(c2.model_ref, "file:/y.gguf");
        assert!(c2.strict_verification);
    }
}
