//! Model loading utilities for `SafeTensors` and GGUF formats
//!
//! Created by: TEAM-015 (refactored from `candle_backend.rs`)
//! Original code by: TEAM-000, TEAM-009, TEAM-011

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama, LlamaEosToks};
use std::path::Path;

/// Load Llama model from `SafeTensors` or GGUF
///
/// TEAM-009: Uses candle-transformers Llama directly
pub fn load_model(model_path: &str, device: &Device) -> Result<(Llama, Config, u64)> {
    let path = Path::new(model_path);

    // Determine model format
    let is_gguf = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));

    tracing::info!(
        path = %model_path,
        format = if is_gguf { "gguf" } else { "safetensors" },
        device = ?device,
        "Loading Llama model"
    );

    // Load model based on format
    if is_gguf {
        load_gguf(model_path, device)
    } else {
        load_safetensors(model_path, device)
    }
}

/// Load GGUF format model
///
/// TEAM-009: GGUF support deferred - use `SafeTensors` for now
fn load_gguf(_path: &str, _device: &Device) -> Result<(Llama, Config, u64)> {
    bail!("GGUF support not yet implemented - use SafeTensors format instead");
}

/// Load `SafeTensors` format model
///
/// TEAM-009: Uses `VarBuilder` + candle-transformers Llama
/// TEAM-011: Fixed directory scanning bug
fn load_safetensors(path: &str, device: &Device) -> Result<(Llama, Config, u64)> {
    let path = Path::new(path);

    // TEAM-011: Fixed directory scanning bug - was using parent instead of path
    let (parent, safetensor_files) =
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            // Single file: use its parent directory
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            (parent, vec![path.to_path_buf()])
        } else if path.is_dir() {
            // Directory: scan for .safetensors files
            let mut files = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    files.push(entry_path);
                }
            }
            (path, files)
        } else {
            bail!("Path must be a .safetensors file or directory containing .safetensors files");
        };

    if safetensor_files.is_empty() {
        bail!("No safetensors files found at {}", path.display());
    }

    // Calculate total size
    let model_size_bytes: u64 =
        safetensor_files.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();

    // TEAM-011: Parse config.json to determine model architecture
    let config_path = parent.join("config.json");
    let config_json: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
    )?;

    // Parse config manually since Config doesn't implement Deserialize
    let hidden_size =
        config_json["hidden_size"].as_u64().context("config.json missing hidden_size")?;
    let intermediate_size = config_json["intermediate_size"]
        .as_u64()
        .context("config.json missing intermediate_size")?;
    let num_hidden_layers = config_json["num_hidden_layers"]
        .as_u64()
        .context("config.json missing num_hidden_layers")?;
    let num_attention_heads = config_json["num_attention_heads"]
        .as_u64()
        .context("config.json missing num_attention_heads")?;
    let num_key_value_heads =
        config_json["num_key_value_heads"].as_u64().unwrap_or(num_attention_heads); // Default to MHA if not specified
    let vocab_size =
        config_json["vocab_size"].as_u64().context("config.json missing vocab_size")?;
    let rms_norm_eps = config_json["rms_norm_eps"].as_f64().unwrap_or(1e-5);
    let rope_theta = config_json["rope_theta"].as_f64().unwrap_or(10000.0);
    let max_position_embeddings = config_json["max_position_embeddings"].as_u64().unwrap_or(2048);
    let bos_token_id = config_json["bos_token_id"].as_u64().unwrap_or(1);
    let eos_token_id = config_json["eos_token_id"].as_u64().unwrap_or(2);
    let tie_word_embeddings = config_json["tie_word_embeddings"].as_bool().unwrap_or(false);

    tracing::info!(
        hidden_size = hidden_size,
        intermediate_size = intermediate_size,
        num_layers = num_hidden_layers,
        num_heads = num_attention_heads,
        num_kv_heads = num_key_value_heads,
        vocab_size = vocab_size,
        "Parsed model config"
    );

    // Build Config from parsed values
    let config = Config {
        hidden_size: hidden_size as usize,
        intermediate_size: intermediate_size as usize,
        vocab_size: vocab_size as usize,
        num_hidden_layers: num_hidden_layers as usize,
        num_attention_heads: num_attention_heads as usize,
        num_key_value_heads: num_key_value_heads as usize,
        rms_norm_eps,
        rope_theta: rope_theta as f32,
        max_position_embeddings: max_position_embeddings as usize,
        bos_token_id: Some(bos_token_id as u32),
        eos_token_id: Some(LlamaEosToks::Single(eos_token_id as u32)),
        rope_scaling: None,
        tie_word_embeddings,
        use_flash_attn: false, // CPU doesn't support flash attention
    };

    // Create VarBuilder from safetensors
    let dtype = DType::F32; // Use F32 for CPU, can be F16 for GPU
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };

    // Load Llama model
    let model = Llama::load(vb, &config).context("Failed to load Llama model from safetensors")?;

    Ok((model, config, model_size_bytes))
}
