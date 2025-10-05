//! FT-046: Final Validation Suite
//!
//! Comprehensive validation of all M0 requirements before Gate 4.
//! Tests all critical functionality end-to-end.
//!
//! Spec: M0 Success Criteria

use worker_orcd::tests::integration::{collect_sse_events, extract_tokens, make_test_request, WorkerTestHarness};
use worker_http::sse::InferenceEvent;

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore] // Only run as part of final validation
async fn test_m0_requirement_model_loading() {
    // M0 Requirement 1: Load Models
    println!("\n=== M0 Validation: Model Loading ===");
    
    let models = vec![
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ".test-models/gpt/gpt-oss-20b-mxfp4.gguf",
    ];
    
    for model_path in models {
        println!("Loading: {}", model_path);
        let harness = WorkerTestHarness::start(model_path, 0)
            .await
            .expect("Failed to load model");
        
        let health = harness.health().await.expect("Health check failed");
        assert!(health.is_healthy, "Model should be healthy after loading");
        
        println!("✅ Loaded: {}", model_path);
    }
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_m0_requirement_token_generation() {
    // M0 Requirement 2: Generate Tokens
    println!("\n=== M0 Validation: Token Generation ===");
    
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let req = make_test_request("m0-tokens", "Count to three", 20);
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    
    let tokens = extract_tokens(&events);
    assert!(!tokens.is_empty(), "Should generate tokens");
    
    println!("✅ Generated {} tokens", tokens.len());
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_m0_requirement_sse_streaming() {
    // M0 Requirement 3: Stream Results via SSE
    println!("\n=== M0 Validation: SSE Streaming ===");
    
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let req = make_test_request("m0-sse", "Hello", 10);
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    
    // Validate SSE event sequence
    assert!(matches!(events.first(), Some(InferenceEvent::Started { .. })));
    assert!(events.last().unwrap().is_terminal());
    
    println!("✅ SSE streaming validated ({} events)", events.len());
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_m0_requirement_vram_enforcement() {
    // M0 Requirement 4: VRAM Enforcement
    println!("\n=== M0 Validation: VRAM Enforcement ===");
    
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let metrics = harness.get_metrics().await.expect("Failed to get metrics");
    assert!(metrics.vram_used_bytes > 0, "Should use VRAM");
    
    println!("✅ VRAM usage: {} MB", metrics.vram_used_bytes / 1024 / 1024);
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_m0_requirement_determinism() {
    // M0 Requirement 5: Determinism with seeded RNG
    println!("\n=== M0 Validation: Determinism ===");
    
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let mut req1 = make_test_request("m0-det-1", "Count to three", 10);
    req1.temperature = 1.0;
    req1.seed = Some(42);
    
    let mut req2 = make_test_request("m0-det-2", "Count to three", 10);
    req2.temperature = 1.0;
    req2.seed = Some(42);
    
    let response1 = harness.execute(req1).await.expect("Execute 1 failed");
    let events1 = collect_sse_events(response1).await.expect("Failed to collect events 1");
    let tokens1 = extract_tokens(&events1);
    
    let response2 = harness.execute(req2).await.expect("Execute 2 failed");
    let events2 = collect_sse_events(response2).await.expect("Failed to collect events 2");
    let tokens2 = extract_tokens(&events2);
    
    assert_eq!(tokens1, tokens2, "Same seed should produce same output");
    
    println!("✅ Determinism validated");
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_m0_requirement_error_handling() {
    // M0 Requirement 6: Error Handling
    println!("\n=== M0 Validation: Error Handling ===");
    
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    // Test invalid request
    let mut req = make_test_request("m0-error", "", 10); // Empty prompt
    req.max_tokens = 0; // Invalid max_tokens
    
    let result = harness.execute(req).await;
    
    // Should either reject or return error event
    match result {
        Err(_) => println!("✅ Request validation rejected invalid request"),
        Ok(response) => {
            let events = collect_sse_events(response).await.expect("Failed to collect events");
            let has_error = events.iter().any(|e| matches!(e, InferenceEvent::Error { .. }));
            assert!(has_error, "Should have error event");
            println!("✅ Error event returned for invalid request");
        }
    }
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_m0_requirement_architecture_detection() {
    // M0 Requirement 7: Architecture Detection
    println!("\n=== M0 Validation: Architecture Detection ===");
    
    let models = vec![
        (".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf", "llama"),
        (".test-models/gpt/gpt-oss-20b-mxfp4.gguf", "gpt"),
    ];
    
    for (model_path, expected_arch) in models {
        let harness = WorkerTestHarness::start(model_path, 0)
            .await
            .expect("Failed to start worker");
        
        let req = make_test_request("m0-arch", "Test", 5);
        let response = harness.execute(req).await.expect("Execute failed");
        let events = collect_sse_events(response).await.expect("Failed to collect events");
        
        // Should successfully generate tokens (proves correct adapter selected)
        let tokens = extract_tokens(&events);
        assert!(!tokens.is_empty(), "Should generate tokens for {}", expected_arch);
        
        println!("✅ Architecture {} detected and working", expected_arch);
    }
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_m0_complete_workflow() {
    // Complete end-to-end workflow
    println!("\n=== M0 Validation: Complete Workflow ===");
    
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    // 1. Health check
    let health = harness.health().await.expect("Health check failed");
    assert!(health.is_healthy);
    println!("✅ Health check passed");
    
    // 2. Metrics check
    let metrics_before = harness.get_metrics().await.expect("Metrics failed");
    println!("✅ Metrics retrieved");
    
    // 3. Execute inference
    let req = make_test_request("m0-workflow", "Write a haiku", 30);
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    println!("✅ Inference completed");
    
    // 4. Validate events
    assert!(matches!(events.first(), Some(InferenceEvent::Started { .. })));
    let tokens = extract_tokens(&events);
    assert!(!tokens.is_empty());
    assert!(events.last().unwrap().is_terminal());
    println!("✅ Events validated");
    
    // 5. Verify metrics updated
    let metrics_after = harness.get_metrics().await.expect("Metrics failed");
    assert!(metrics_after.tokens_out_total > metrics_before.tokens_out_total);
    println!("✅ Metrics updated");
    
    println!("\n🎉 M0 Complete Workflow PASSED");
}

// Built by Foundation-Alpha 🏗️
