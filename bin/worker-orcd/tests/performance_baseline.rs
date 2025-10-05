//! FT-040: Performance Baseline Measurements
//!
//! Measures and documents performance baselines for M0.
//! Tracks token generation speed, latency, and throughput.
//!
//! Spec: M0-W-1230

use std::time::{Duration, Instant};
use worker_orcd::tests::integration::{collect_sse_events, extract_tokens, make_test_request, WorkerTestHarness};

#[derive(Debug, serde::Serialize)]
struct PerformanceMetrics {
    model: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    total_time_ms: u128,
    tokens_per_second: f64,
    time_to_first_token_ms: u128,
    avg_token_latency_ms: f64,
    timestamp: String,
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore] // Only run with real model
async fn test_qwen_baseline_performance() {
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let mut req = make_test_request(
        "perf-baseline-qwen",
        "Write a short story about a robot.",
        100
    );
    req.temperature = 0.7;
    req.seed = Some(42);
    
    let start = Instant::now();
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    let total_time = start.elapsed();
    
    let tokens = extract_tokens(&events);
    let generated_tokens = tokens.len();
    
    let metrics = PerformanceMetrics {
        model: "qwen2.5-0.5b-instruct-q4_k_m".to_string(),
        prompt_tokens: 8, // Approximate
        generated_tokens,
        total_time_ms: total_time.as_millis(),
        tokens_per_second: generated_tokens as f64 / total_time.as_secs_f64(),
        time_to_first_token_ms: 0, // Would need event timestamps
        avg_token_latency_ms: total_time.as_millis() as f64 / generated_tokens as f64,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    println!("\n=== Qwen Performance Baseline ===");
    println!("Generated tokens: {}", metrics.generated_tokens);
    println!("Total time: {} ms", metrics.total_time_ms);
    println!("Tokens/sec: {:.2}", metrics.tokens_per_second);
    println!("Avg latency: {:.2} ms/token", metrics.avg_token_latency_ms);
    
    // Save baseline
    save_baseline(&metrics, "qwen-baseline.json");
    
    // Basic performance assertions
    assert!(metrics.tokens_per_second > 1.0, "Should generate at least 1 token/sec");
    assert!(metrics.avg_token_latency_ms < 5000.0, "Avg latency should be < 5s/token");
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore] // Only run with real model
async fn test_gpt_baseline_performance() {
    let harness = WorkerTestHarness::start(
        ".test-models/gpt/gpt-oss-20b-mxfp4.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let mut req = make_test_request(
        "perf-baseline-gpt",
        "Write a short story about a robot.",
        100
    );
    req.temperature = 0.7;
    req.seed = Some(42);
    
    let start = Instant::now();
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    let total_time = start.elapsed();
    
    let tokens = extract_tokens(&events);
    let generated_tokens = tokens.len();
    
    let metrics = PerformanceMetrics {
        model: "gpt-oss-20b-mxfp4".to_string(),
        prompt_tokens: 8,
        generated_tokens,
        total_time_ms: total_time.as_millis(),
        tokens_per_second: generated_tokens as f64 / total_time.as_secs_f64(),
        time_to_first_token_ms: 0,
        avg_token_latency_ms: total_time.as_millis() as f64 / generated_tokens as f64,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    println!("\n=== GPT Performance Baseline ===");
    println!("Generated tokens: {}", metrics.generated_tokens);
    println!("Total time: {} ms", metrics.total_time_ms);
    println!("Tokens/sec: {:.2}", metrics.tokens_per_second);
    println!("Avg latency: {:.2} ms/token", metrics.avg_token_latency_ms);
    
    save_baseline(&metrics, "gpt-baseline.json");
    
    assert!(metrics.tokens_per_second > 0.5, "Should generate at least 0.5 token/sec");
    assert!(metrics.avg_token_latency_ms < 10000.0, "Avg latency should be < 10s/token");
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore]
async fn test_batch_performance() {
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let prompts = vec![
        "Count to five",
        "List three colors",
        "Name two animals",
    ];
    
    let mut total_tokens = 0;
    let start = Instant::now();
    
    for (i, prompt) in prompts.iter().enumerate() {
        let mut req = make_test_request(
            &format!("batch-{}", i),
            prompt,
            20
        );
        req.temperature = 0.7;
        
        let response = harness.execute(req).await.expect("Execute failed");
        let events = collect_sse_events(response).await.expect("Failed to collect events");
        let tokens = extract_tokens(&events);
        total_tokens += tokens.len();
    }
    
    let total_time = start.elapsed();
    let throughput = total_tokens as f64 / total_time.as_secs_f64();
    
    println!("\n=== Batch Performance ===");
    println!("Total requests: {}", prompts.len());
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:?}", total_time);
    println!("Throughput: {:.2} tokens/sec", throughput);
    
    assert!(throughput > 1.0, "Batch throughput should be > 1 token/sec");
}

#[test]
fn test_performance_calculation() {
    // Test performance metric calculations
    let tokens = 100;
    let time_ms = 5000;
    
    let tokens_per_sec = tokens as f64 / (time_ms as f64 / 1000.0);
    let avg_latency = time_ms as f64 / tokens as f64;
    
    assert_eq!(tokens_per_sec, 20.0);
    assert_eq!(avg_latency, 50.0);
}

#[test]
fn test_throughput_calculation() {
    let total_tokens = 300;
    let total_secs = 10.0;
    
    let throughput = total_tokens as f64 / total_secs;
    assert_eq!(throughput, 30.0);
}

fn save_baseline(metrics: &PerformanceMetrics, filename: &str) {
    let dir = std::path::PathBuf::from(".test-results/performance");
    std::fs::create_dir_all(&dir).ok();
    
    let path = dir.join(filename);
    let json = serde_json::to_string_pretty(metrics).unwrap();
    std::fs::write(&path, json).ok();
    
    println!("Baseline saved to: {}", path.display());
}

// Built by Foundation-Alpha 🏗️
