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
    let schema = build_schema();
    let json = serde_json::to_string_pretty(&schema).expect("schema serde");
    // Write only if changed to keep mtime stable.
    let write_needed = match fs::read_to_string(path) {
        Ok(existing) => existing != json,
        Err(_) => true,
    };
    if write_needed {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, json)?;
        println!("wrote {}", path.display());
    } else {
        println!("unchanged {}", path.display());
    }
    Ok(())
}
