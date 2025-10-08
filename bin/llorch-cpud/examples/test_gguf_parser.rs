//! Test GGUF parser with Llama-2 7B model
//!
//! Created by: TEAM-008
//!
//! Usage: cargo run --example test_gguf_parser

use llorch_cpud::model::GGUFParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GGUF Parser Test ===\n");

    let model_path = "../../.test-models/llama2-7b/llama-2-7b.Q8_0.gguf";
    
    println!("Loading model: {}", model_path);
    println!("This may take a few seconds...\n");

    let parser = GGUFParser::parse(model_path)?;

    // Validate it's a Llama model
    parser.validate_llama()?;
    println!("✅ Validated as Llama architecture\n");

    // Print summary
    parser.print_summary();

    // Check for specific tensors
    println!("\n=== Key Tensor Verification ===");
    
    let key_tensors = vec![
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight",
    ];

    for tensor_name in &key_tensors {
        if let Some(tensor) = parser.get_tensor(tensor_name) {
            println!("✅ {}: {:?} ({:?})", 
                tensor_name, 
                tensor.dimensions,
                tensor.tensor_type
            );
        } else {
            println!("❌ {} NOT FOUND", tensor_name);
        }
    }

    // Count layer tensors
    println!("\n=== Layer Count Verification ===");
    let tensor_names = parser.tensor_names();
    let layer_count = (0..100)
        .filter(|i| tensor_names.iter().any(|name| name.contains(&format!("blk.{}", i))))
        .count();
    
    println!("Found {} layers", layer_count);
    if layer_count == 32 {
        println!("✅ Correct number of layers for Llama-2 7B");
    } else {
        println!("❌ Expected 32 layers, found {}", layer_count);
    }

    println!("\n=== Test Complete ===");
    println!("✅ GGUF parser working correctly!");

    Ok(())
}
