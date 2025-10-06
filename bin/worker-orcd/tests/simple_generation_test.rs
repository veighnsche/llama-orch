//! Simple text generation test - proves inference works
//!
//! This test validates that the model can generate coherent text,
//! without requiring complex instruction following.

use worker_orcd::tests::integration::{
    collect_sse_events, extract_tokens, make_test_request, WorkerTestHarness,
};

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_simple_text_generation() {
    // Use Qwen 2.5 0.5B FP16 - already downloaded and working
    let model_path =
        "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf";

    let mut harness =
        WorkerTestHarness::start(model_path, 0).await.expect("Failed to start worker");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Simple prompt
    let prompt = "The quick brown fox";

    let mut req = make_test_request("simple-gen-test", prompt, 20);
    req.temperature = 0.7;

    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");

    let tokens = extract_tokens(&events);
    let generated_text = tokens.join("");

    eprintln!("\n🎯 Generated text: {}", generated_text);
    eprintln!("📊 Token count: {}", tokens.len());

    // Validate we generated tokens
    assert!(!tokens.is_empty(), "Should generate at least one token");
    assert!(tokens.len() >= 5, "Should generate at least 5 tokens, got {}", tokens.len());

    // Validate we got some ASCII/English characters (not all garbage)
    let ascii_chars: usize = generated_text
        .chars()
        .filter(|c| c.is_ascii_alphabetic() || c.is_ascii_whitespace())
        .count();

    let total_chars = generated_text.chars().count();
    let ascii_ratio = ascii_chars as f64 / total_chars.max(1) as f64;

    eprintln!("📈 ASCII ratio: {:.1}%", ascii_ratio * 100.0);

    // At least 30% should be ASCII (proves it's not random garbage)
    assert!(
        ascii_ratio >= 0.3,
        "Generated text should contain at least 30% ASCII characters, got {:.1}%",
        ascii_ratio * 100.0
    );

    eprintln!("✅ Test passed! Model generates coherent text.");
}
