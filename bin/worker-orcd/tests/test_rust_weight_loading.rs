//! Test Rust Weight Loading
//!
//! This example tests the complete Rust weight loading pipeline:
//! 1. Parse GGUF file
//! 2. Load and dequantize Q4_K tensors to FP16
//! 3. Upload to GPU
//! 4. Create C++ model from pointers
//!
//! Usage:
//!   cargo run --example test_rust_weight_loading --features cuda -- /path/to/model.gguf

use worker_gguf::GGUFMetadata;
use worker_orcd::cuda;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    eprintln!("🚀 Testing Rust Weight Loading");
    eprintln!("Model: {}", model_path);
    eprintln!();

    // Step 1: Parse GGUF metadata
    eprintln!("📖 Step 1: Parsing GGUF metadata...");
    let metadata = GGUFMetadata::from_file(model_path)?;

    let vocab_size = metadata.vocab_size()? as u32;
    let hidden_dim = metadata.hidden_dim()? as u32;
    let num_layers = metadata.num_layers()? as u32;
    let num_heads = metadata.num_heads()? as u32;
    let num_kv_heads = metadata.num_kv_heads()? as u32;
    let context_length = metadata.context_length()? as u32;

    eprintln!("  Architecture: {}", metadata.architecture()?);
    eprintln!("  Vocab size: {}", vocab_size);
    eprintln!("  Hidden dim: {}", hidden_dim);
    eprintln!("  Layers: {}", num_layers);
    eprintln!("  Heads: {} (KV: {})", num_heads, num_kv_heads);
    eprintln!("  Context: {}", context_length);
    eprintln!();

    // Step 2: Load weights via Rust
    eprintln!("⚙️  Step 2: Loading weights via Rust...");
    let model = unsafe {
        cuda::load_model_from_rust(
            model_path,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            context_length,
        )?
    };

    eprintln!();
    eprintln!("✅ SUCCESS: Model loaded via Rust weight loading!");
    eprintln!();
    eprintln!("Next steps:");
    eprintln!("  1. Verify VRAM usage with nvidia-smi");
    eprintln!("  2. Run inference to test correctness");
    eprintln!("  3. Compare with C++ weight loading");

    // Note: Model cleanup would happen here in real code
    // For now, we just let it leak for testing

    Ok(())
}
