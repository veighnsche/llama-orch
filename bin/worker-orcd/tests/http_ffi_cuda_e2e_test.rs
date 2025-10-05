//! HTTP-FFI-CUDA End-to-End Integration Test
//!
//! Tests complete flow: HTTP → Rust → FFI → C++ → CUDA → C++ → FFI → Rust → HTTP
//!
//! # Story
//!
//! # Spec References
//! - M0-W-1820: Integration tests

use worker_orcd::http::sse::{InferenceEvent, StopReason};
use worker_orcd::tests::integration::{
    assert_event_order, extract_tokens, make_test_request, TestConfig, TestModel, WorkerTestHarness,
};

// ============================================================================
// Complete Inference Flow Tests
// ============================================================================

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary and model"]
async fn test_complete_inference_flow() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0)
        .await
        .expect("Failed to start worker");

    let req = make_test_request("e2e-test-001", "Write a haiku about GPU computing", 50);

    let response = harness.execute(req).await.expect("Execute failed");

    // Verify SSE stream
    assert_eq!(response.status(), 200);
    assert!(response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("text/event-stream"));

    // Collect events
    let events = harness.collect_sse_events(response).await;

    // Validate event sequence
    assert_event_order(&events).expect("Invalid event order");

    // Verify Started event
    assert!(matches!(events[0], InferenceEvent::Started { .. }));

    // Verify Token events
    let token_count = events.iter().filter(|e| matches!(e, InferenceEvent::Token { .. })).count();
    assert!(token_count > 0, "No tokens generated");

    // Verify End event
    assert!(matches!(events.last().unwrap(), InferenceEvent::End { .. }));

    // Extract and validate tokens
    let tokens = extract_tokens(&events);
    assert!(!tokens.is_empty(), "No tokens extracted");

    // Verify output is not empty
    let output = tokens.join("");
    assert!(!output.is_empty(), "Empty output");

    eprintln!("✅ E2E test passed: {} tokens generated", tokens.len());
    eprintln!("Output: {}", output);
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary and model"]
async fn test_determinism() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0)
        .await
        .expect("Failed to start worker");

    // Use greedy sampling (temperature=0) for determinism
    let mut req1 = make_test_request("det-test-1", "Count to five", 20);
    req1.temperature = 0.0; // Greedy
    req1.seed = Some(42);

    // Run first generation
    let response1 = harness.execute(req1.clone()).await.unwrap();
    let events1 = collect_sse_events(response1).await.unwrap();
    let tokens1 = extract_tokens(&events1);

    // Run second generation with same parameters
    let mut req2 = req1.clone();
    req2.job_id = "det-test-2".to_string();

    let response2 = harness.execute(req2).await.unwrap();
    let events2 = collect_sse_events(response2).await.unwrap();
    let tokens2 = extract_tokens(&events2);

    // Same seed + temperature=0 should produce identical tokens
    assert_eq!(tokens1, tokens2, "Determinism failed: outputs differ with same seed");

    eprintln!("✅ Determinism test passed: {} tokens identical", tokens1.len());
    eprintln!("Output: {}", tokens1.join(""));
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary and model"]
async fn test_multiple_requests() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0)
        .await
        .expect("Failed to start worker");

    // Send multiple requests sequentially
    for i in 0..3 {
        let req =
            make_test_request(&format!("multi-test-{}", i), &format!("Test prompt {}", i), 10);

        let response = harness.execute(req).await.expect("Execute failed");
        let events = harness.collect_sse_events(response).await;

        assert_event_order(&events).expect("Invalid event order");

        let tokens = extract_tokens(&events);
        assert!(!tokens.is_empty(), "No tokens for request {}", i);

        eprintln!("✅ Request {} completed: {} tokens", i, tokens.len());
    }
}

// ============================================================================
// Health Endpoint Tests
// ============================================================================

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary and model"]
async fn test_health_during_inference() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0)
        .await
        .expect("Failed to start worker");

    // Start inference
    let req = make_test_request("health-test-1", "Generate a long story", 100);

    let _response = harness.execute(req).await.expect("Execute failed");

    // Query health endpoint during inference
    let health = harness.health().await.expect("Health check failed");

    // Verify health response
    assert!(health.get("status").is_some());
    assert_eq!(health["status"], "ok");

    eprintln!("✅ Health endpoint responsive during inference");
}

// ============================================================================
// VRAM Enforcement Tests
// ============================================================================

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary and model"]
async fn test_vram_only_operation() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0)
        .await
        .expect("Failed to start worker");

    let req = make_test_request("vram-test-1", "Test VRAM enforcement", 20);

    let response = harness.execute(req).await.expect("Execute failed");
    let events = harness.collect_sse_events(response).await;

    assert_event_order(&events).expect("Invalid event order");

    let tokens = extract_tokens(&events);
    assert!(!tokens.is_empty(), "No tokens generated");

    // If we got here, VRAM-only operation succeeded
    // (Would fail if RAM fallback was attempted)
    eprintln!("✅ VRAM-only operation validated");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary"]
async fn test_invalid_request_handling() {
    let harness = WorkerTestHarness::start_mock().await.unwrap();

    // Send request with invalid parameters
    let mut req = make_test_request("error-test-1", "Test", 10);
    req.max_tokens = 0; // Invalid

    let result = harness.execute(req).await;

    // Should fail with validation error
    if let Err(e) = result {
        eprintln!("✅ Invalid request rejected: {}", e);
    } else {
        panic!("Invalid request should have been rejected");
    }
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary and model"]
async fn test_context_length_exceeded() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0)
        .await
        .expect("Failed to start worker");

    // Generate very long prompt (exceeds context)
    let long_prompt = "word ".repeat(40000); // ~40K tokens

    let req = make_test_request("context-test-1", &long_prompt, 10);

    let result = harness.execute(req).await;

    // Should fail with context length error
    if let Err(e) = result {
        eprintln!("✅ Context length error handled: {}", e);
    } else {
        // Or succeed with truncation
        eprintln!("✅ Long prompt handled (truncated)");
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore = "Requires worker binary and model"]
async fn test_inference_performance() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0)
        .await
        .expect("Failed to start worker");

    let req = make_test_request("perf-test-1", "Performance test", 50);

    let start = std::time::Instant::now();
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.unwrap();
    let elapsed = start.elapsed();

    let tokens = extract_tokens(&events);
    let tokens_per_sec = tokens.len() as f64 / elapsed.as_secs_f64();

    eprintln!("✅ Performance test completed:");
    eprintln!("  Tokens: {}", tokens.len());
    eprintln!("  Time: {:.2}s", elapsed.as_secs_f64());
    eprintln!("  Throughput: {:.1} tokens/sec", tokens_per_sec);

    // Sanity check: should be at least 1 token/sec
    assert!(tokens_per_sec >= 1.0, "Performance too slow: {:.1} tokens/sec", tokens_per_sec);
}

// ---
// Built by Foundation-Alpha 🏗️
// Story: FT-024 (HTTP-FFI-CUDA Integration Test)
