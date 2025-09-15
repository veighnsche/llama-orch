//! Config schema types (pre-code). Emits deterministic JSON Schema.

use schemars::{schema::RootSchema, JsonSchema};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Llamacpp,
    Vllm,
    Tgi,
    Triton,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct QueueConfig {
    pub capacity: Option<u32>,
    pub full_policy: Option<String>, // "reject" | "drop-lru" | "shed-low-priority"
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct PriorityClass {
    pub name: String, // interactive | batch
    pub queue_capacity: Option<u32>,
    pub rate_limit_rps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct AdmissionConfig {
    pub priorities: Option<Vec<PriorityClass>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct Timeouts {
    pub wall_ms: Option<u64>,
    pub idle_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PoolConfig {
    pub id: String,
    pub engine: Engine,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u32>,
    pub devices: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<Vec<f32>>, // ratios
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preload: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_same_engine_version: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler_profile_version: Option<String>,
    #[serde(default)]
    pub queue: QueueConfig,
    #[serde(default)]
    pub admission: AdmissionConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct Config {
    pub pools: Vec<PoolConfig>,
}

/// Build the JSON Schema for the top-level `Config` type.
pub fn build_schema() -> RootSchema {
    schemars::schema_for!(Config)
}

/// Emit the schema to the given path deterministically (without timestamps).
pub fn emit_schema_json(path: &std::path::Path) -> std::io::Result<()> {
    use std::fs;
    use std::io::{self, Write};
    let schema = build_schema();
    let json =
        serde_json::to_string_pretty(&schema).map_err(|e| std::io::Error::other(e.to_string()))?;
    // Write only if changed to keep mtime stable.
    let write_needed = match fs::read_to_string(path) {
        Ok(existing) => existing != json,
        Err(_) => true,
    };
    if write_needed {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let dir = path.parent().unwrap();
        let tmp = dir.join(format!(
            ".{}.tmp",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("schema")
        ));
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(json.as_bytes())?;
            f.sync_all()?;
        }
        match fs::rename(&tmp, path) {
            Ok(_) => {}
            Err(e) => {
                let is_exdev = matches!(e.raw_os_error(), Some(code) if code == 18);
                if is_exdev {
                    let mut dest = fs::File::create(path)?;
                    let mut src = fs::File::open(&tmp)?;
                    io::copy(&mut src, &mut dest)?;
                    dest.sync_all()?;
                    let _ = fs::remove_file(&tmp);
                } else {
                    let _ = fs::remove_file(&tmp);
                    return Err(e);
                }
            }
        }
        let dirf = fs::File::open(dir)?;
        dirf.sync_all()?;
        println!("wrote {}", path.display());
    } else {
        println!("unchanged {}", path.display());
    }
    Ok(())
}
