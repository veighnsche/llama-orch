// VRAM Pressure Tests - LT-037
//
// Tests VRAM allocation, usage, and pressure handling.
// Validates memory management for large models (Phi-3).
//
// Spec: M0-W-1221

use worker_orcd::models::{
    qwen::{QwenConfig, QwenWeightLoader},
    phi3::{Phi3Config, Phi3WeightLoader},
};

/// Test: Qwen VRAM allocation
#[test]
fn test_qwen_vram_allocation() {
    let config = QwenConfig::qwen2_5_0_5b();
    let result = QwenWeightLoader::load_to_vram("dummy.gguf", &config);
    
    assert!(result.is_ok());
    let model = result.unwrap();
    
    // Verify VRAM usage is calculated
    assert!(model.total_vram_bytes > 0);
    assert!(model.total_vram_bytes > 1_000_000_000); // >1GB
    
    eprintln!("Qwen VRAM: {} MB", model.total_vram_bytes / (1024 * 1024));
}

/// Test: Phi-3 VRAM allocation (large model)
#[test]
fn test_phi3_vram_allocation() {
    let config = Phi3Config::phi3_mini_4k();
    let result = Phi3WeightLoader::load_to_vram("dummy.gguf", &config);
    
    assert!(result.is_ok());
    let model = result.unwrap();
    
    // Verify VRAM usage for large model
    assert!(model.total_vram_bytes > 6_000_000_000); // >6GB
    
    eprintln!("Phi-3 VRAM: {} MB", model.total_vram_bytes / (1024 * 1024));
}

/// Test: VRAM usage calculation accuracy
#[test]
fn test_vram_calculation_accuracy() {
    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_vram = QwenWeightLoader::calculate_vram_usage(&qwen_config);
    
    let phi3_config = Phi3Config::phi3_mini_4k();
    let phi3_vram = Phi3WeightLoader::calculate_vram_usage(&phi3_config);
    
    // Verify calculations are reasonable
    assert!(qwen_vram > 0);
    assert!(phi3_vram > 0);
    assert!(phi3_vram > qwen_vram);
    
    // Verify against expected sizes
    let qwen_mb = qwen_vram / (1024 * 1024);
    let phi3_mb = phi3_vram / (1024 * 1024);
    
    assert!(qwen_mb > 1000 && qwen_mb < 1500, "Qwen VRAM: {} MB", qwen_mb);
    assert!(phi3_mb > 6000 && phi3_mb < 9000, "Phi-3 VRAM: {} MB", phi3_mb);
    
    eprintln!("Qwen: {} MB", qwen_mb);
    eprintln!("Phi-3: {} MB", phi3_mb);
}

/// Test: Multiple model loading (VRAM pressure)
#[test]
fn test_multiple_model_loading() {
    // Load Qwen
    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_result = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config);
    assert!(qwen_result.is_ok());
    
    // Load Phi-3
    let phi3_config = Phi3Config::phi3_mini_4k();
    let phi3_result = Phi3WeightLoader::load_to_vram("dummy.gguf", &phi3_config);
    assert!(phi3_result.is_ok());
    
    // Both models loaded successfully
    let qwen_model = qwen_result.unwrap();
    let phi3_model = phi3_result.unwrap();
    
    let total_vram = qwen_model.total_vram_bytes + phi3_model.total_vram_bytes;
    
    eprintln!("Total VRAM (both models): {} MB", total_vram / (1024 * 1024));
    
    // Should be ~8.8GB total
    assert!(total_vram > 7_000_000_000);
}

/// Test: VRAM usage breakdown
#[test]
fn test_vram_usage_breakdown() {
    let config = Phi3Config::phi3_mini_4k();
    let fp16_size = 2;
    
    // Calculate component sizes
    let embedding_size = config.vocab_size * config.hidden_dim * fp16_size;
    let layer_size = (
        config.hidden_dim + // attn_norm
        config.hidden_dim * config.hidden_dim + // Q
        config.hidden_dim * config.hidden_dim + // K
        config.hidden_dim * config.hidden_dim + // V
        config.hidden_dim * config.hidden_dim + // output
        config.hidden_dim + // ffn_norm
        config.ffn_dim * config.hidden_dim + // gate
        config.ffn_dim * config.hidden_dim + // up
        config.hidden_dim * config.ffn_dim // down
    ) * fp16_size;
    let output_size = (config.hidden_dim + config.vocab_size * config.hidden_dim) * fp16_size;
    
    let total_calculated = embedding_size + (layer_size * config.num_layers) + output_size;
    let total_from_loader = Phi3WeightLoader::calculate_vram_usage(&config);
    
    // Should match
    assert_eq!(total_calculated, total_from_loader);
    
    eprintln!("Embedding: {} MB", embedding_size / (1024 * 1024));
    eprintln!("Per-layer: {} MB", layer_size / (1024 * 1024));
    eprintln!("32 Layers: {} MB", (layer_size * config.num_layers) / (1024 * 1024));
    eprintln!("Output: {} MB", output_size / (1024 * 1024));
    eprintln!("Total: {} MB", total_from_loader / (1024 * 1024));
}

/// Test: VRAM limits validation
#[test]
fn test_vram_limits() {
    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let phi3_config = Phi3Config::phi3_mini_4k();
    
    let qwen_vram = QwenWeightLoader::calculate_vram_usage(&qwen_config);
    let phi3_vram = Phi3WeightLoader::calculate_vram_usage(&phi3_config);
    
    // Verify models fit in reasonable VRAM limits
    // Assuming 24GB GPU (e.g., RTX 3090, RTX 4090)
    let gpu_vram = 24 * 1024 * 1024 * 1024; // 24GB
    
    assert!(qwen_vram < gpu_vram, "Qwen exceeds 24GB VRAM");
    assert!(phi3_vram < gpu_vram, "Phi-3 exceeds 24GB VRAM");
    
    // Verify both can fit simultaneously (with overhead)
    let total_with_overhead = (qwen_vram + phi3_vram) * 12 / 10; // 20% overhead
    assert!(total_with_overhead < gpu_vram, "Both models exceed 24GB with overhead");
    
    eprintln!("Qwen: {} GB", qwen_vram / (1024 * 1024 * 1024));
    eprintln!("Phi-3: {} GB", phi3_vram / (1024 * 1024 * 1024));
    eprintln!("Total + overhead: {} GB", total_with_overhead / (1024 * 1024 * 1024));
}

/// Test: Memory efficiency comparison
#[test]
fn test_memory_efficiency() {
    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let phi3_config = Phi3Config::phi3_mini_4k();
    
    let qwen_vram = QwenWeightLoader::calculate_vram_usage(&qwen_config);
    let phi3_vram = Phi3WeightLoader::calculate_vram_usage(&phi3_config);
    
    // Calculate bytes per parameter
    // Qwen: ~500M params, Phi-3: ~3.8B params
    let qwen_params = 500_000_000_u64;
    let phi3_params = 3_800_000_000_u64;
    
    let qwen_bytes_per_param = qwen_vram as f64 / qwen_params as f64;
    let phi3_bytes_per_param = phi3_vram as f64 / phi3_params as f64;
    
    // Should be ~2 bytes per param (FP16)
    assert!(qwen_bytes_per_param > 1.5 && qwen_bytes_per_param < 3.0);
    assert!(phi3_bytes_per_param > 1.5 && phi3_bytes_per_param < 3.0);
    
    eprintln!("Qwen: {:.2} bytes/param", qwen_bytes_per_param);
    eprintln!("Phi-3: {:.2} bytes/param", phi3_bytes_per_param);
}
