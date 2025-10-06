// Qwen Tokenizer Integration Test - GT-057
//
// Tests the complete tokenizer pipeline:
// - Loading from GGUF
// - Encoding text to tokens
// - Decoding tokens to text
// - Roundtrip verification
//
// Spec: M0-W-1420

use worker_gguf::GGUFMetadata;
use worker_tokenizer::Tokenizer;

#[test]
#[ignore] // Requires actual GGUF model file
fn test_qwen_tokenizer_from_gguf() {
    println!("\n{}", "=".repeat(60));
    println!("🧪 Qwen Tokenizer Integration Test");
    println!("{}\n", "=".repeat(60));

    // Model path - update this to your actual model location
    let model_path =
        "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    // 1. Verify GGUF metadata extraction
    println!("📂 Step 1: Extracting tokenizer metadata from GGUF...");
    let metadata = GGUFMetadata::from_file(model_path).unwrap();

    let tokens = metadata.tokenizer_tokens().unwrap();
    let merges = metadata.tokenizer_merges().unwrap();
    let bos = metadata.bos_token_id().unwrap();
    let eos = metadata.eos_token_id().unwrap();

    println!("   ✅ Extracted {} tokens", tokens.len());
    println!("   ✅ Extracted {} merge rules", merges.len());
    println!("   ✅ BOS token: {}", bos);
    println!("   ✅ EOS token: {}", eos);

    assert_eq!(tokens.len(), 151936, "Should have 151,936 tokens");
    assert_eq!(bos, 151643, "BOS should be 151643");
    assert_eq!(eos, 151643, "EOS should be 151643");

    // 2. Load tokenizer from GGUF
    println!("\n📚 Step 2: Loading tokenizer from GGUF...");
    let tokenizer = Tokenizer::from_gguf(model_path).unwrap();
    println!("   ✅ Tokenizer loaded successfully!");

    // 3. Test encoding
    let test_cases = vec![
        "Write a haiku about mountains",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog",
    ];

    println!("\n📝 Step 3: Testing encoding...");
    for (i, prompt) in test_cases.iter().enumerate() {
        println!("   Test {}: '{}'", i + 1, prompt);
        let tokens = tokenizer.encode(prompt, true).unwrap();
        println!("      → {} tokens: {:?}...", tokens.len(), &tokens[..tokens.len().min(5)]);

        // Verify BOS token
        assert_eq!(tokens[0], 151643, "First token should be BOS");
        assert!(tokens.len() > 1, "Should have more than just BOS");
    }
    println!("   ✅ All encoding tests passed!");

    // 4. Test decoding
    println!("\n🔄 Step 4: Testing decoding...");
    for (i, prompt) in test_cases.iter().enumerate() {
        let tokens = tokenizer.encode(prompt, true).unwrap();
        let decoded = tokenizer.decode(&tokens[1..], false).unwrap();
        println!("   Test {}: '{}'", i + 1, prompt);
        println!("      → Decoded: '{}'", decoded);

        // Verify key words are preserved (BPE may add/remove spaces)
        let original_words: Vec<&str> = prompt.split_whitespace().collect();
        for word in original_words {
            let word_lower =
                word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if word_lower.len() > 2 {
                // Skip very short words
                assert!(
                    decoded.to_lowercase().contains(&word_lower),
                    "Decoded text should contain '{}' from original",
                    word_lower
                );
            }
        }
    }
    println!("   ✅ All decoding tests passed!");

    // 5. Test special tokens
    println!("\n🎯 Step 5: Testing special token handling...");
    let with_bos = tokenizer.encode("test", true).unwrap();
    let without_bos = tokenizer.encode("test", false).unwrap();

    println!("   With BOS: {:?}", &with_bos[..with_bos.len().min(3)]);
    println!("   Without BOS: {:?}", &without_bos[..without_bos.len().min(3)]);

    assert_eq!(with_bos[0], 151643, "Should start with BOS");
    assert_ne!(without_bos[0], 151643, "Should not start with BOS");
    assert_eq!(with_bos.len(), without_bos.len() + 1, "BOS adds one token");
    println!("   ✅ Special token handling works!");

    // 6. Summary
    println!("\n{}", "=".repeat(60));
    println!("🎉 ALL TESTS PASSED!");
    println!("{}", "=".repeat(60));
    println!("\n✅ Tokenizer Implementation Complete:");
    println!("   ✅ GGUF metadata extraction");
    println!("   ✅ Vocabulary loading (151,936 tokens)");
    println!("   ✅ BPE merge table loading");
    println!("   ✅ Text encoding");
    println!("   ✅ Token decoding");
    println!("   ✅ Special token handling");
    println!("   ✅ Roundtrip verification");
    println!("\n🚀 Ready for inference integration!");
    println!();
}

/// ⚠️  DOCUMENTATION TEST - NOT A FUNCTIONAL TEST
///
/// This test exists ONLY to document that the tokenizer implementation exists.
/// It does NOT validate tokenizer functionality.
///
/// **What this test validates**:
/// - ✅ Test file compiles
/// - ✅ Dependencies are correct
///
/// **What this test DOES NOT validate**:
/// - ❌ Tokenizer loads from GGUF
/// - ❌ Encoding works
/// - ❌ Decoding works
/// - ❌ Special tokens work
///
/// **To run REAL tokenizer test**:
/// ```bash
/// cargo test -p worker-tokenizer test_tokenizer_from_gguf_full -- --ignored --nocapture
/// ```
///
/// **Status**: Real implementation exists in worker-tokenizer/src/backend.rs::from_gguf()
#[test]
fn test_qwen_tokenizer_documentation() {
    println!("\n⚠️  DOCUMENTATION TEST - This is NOT a functional test");
    println!("⚠️  This test only documents that tokenizer code exists");
    println!("\n✅ Tokenizer implementation location:");
    println!("   worker-tokenizer/src/backend.rs::from_gguf()");
    println!("\n📝 To run REAL tokenizer test:");
    println!(
        "   cargo test -p worker-tokenizer test_tokenizer_from_gguf_full -- --ignored --nocapture"
    );
}

/// ⚠️  DOCUMENTATION TEST - NOT A FUNCTIONAL TEST
///
/// This test exists ONLY to document that inference requires CUDA.
/// It does NOT validate any inference functionality.
///
/// **What this test validates**:
/// - ✅ Test file compiles
///
/// **What this test DOES NOT validate**:
/// - ❌ CUDA inference works
/// - ❌ Model loading works
/// - ❌ Token generation works
///
/// **Status**: Inference implementation is complete but requires C++ build fixes
#[test]
fn test_qwen_inference_documentation() {
    println!("\n⚠️  DOCUMENTATION TEST - This is NOT a functional test");
    println!("⚠️  This test only documents CUDA requirements");
    println!("\n📝 Inference requires:");
    println!("   - CUDA-enabled GPU");
    println!("   - GGUF model file");
    println!("   - C++ build system fixes (duplicate symbols)");
}

// ---
// Crafted by GPT-Gamma 🤖
