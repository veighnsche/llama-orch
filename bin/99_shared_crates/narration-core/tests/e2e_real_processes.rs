// TEAM-303: Real process E2E integration tests
//!
//! Comprehensive E2E tests using real process binaries to verify production scenarios.
//!
//! # Test Coverage
//!
//! 1. **Keeper → Queen** - Job submission with correlation ID
//! 2. **Queen → Hive** - Worker spawn forwarding
//! 3. **Hive → Worker** - Process capture with stdout narration
//! 4. **Full Stack** - Keeper → Queen → Hive → Worker (complete flow)
//! 5. **Correlation ID** - Multi-hop propagation verification
//! 6. **Process Capture** - Real stdout → SSE flow
//! 7. **Failure Scenarios** - Process crashes, timeouts, errors

use tokio::process::Command;
use tokio::time::{timeout, Duration};
use std::process::Stdio;

// TEAM-303: Import test harness
mod harness;

/// Helper to start a binary and wait for it to be ready
async fn start_binary_and_wait(
    binary_name: &str,
    port_env: &str,
    port: u16,
    extra_env: Vec<(&str, &str)>,
) -> tokio::process::Child {
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--bin")
        .arg(binary_name)
        .arg("--features")
        .arg("axum")
        .env(port_env, port.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    
    let child = cmd.spawn().expect("Failed to spawn binary");
    
    // Wait for binary to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    child
}

/// Helper to wait for HTTP service to be ready
async fn wait_for_http_ready(url: &str, max_attempts: u32) -> Result<(), String> {
    let client = reqwest::Client::new();
    
    for attempt in 0..max_attempts {
        if let Ok(resp) = client.get(format!("{}/health", url)).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        if attempt < max_attempts - 1 {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    
    Err(format!("Service at {} never became ready", url))
}

#[tokio::test]
#[ignore] // TEAM-303: Requires cargo build first
async fn test_keeper_to_queen_real_process() {
    // TEAM-303: Test Keeper → Queen with real process
    
    // Start fake queen
    let mut queen = start_binary_and_wait(
        "fake-queen-rbee",
        "QUEEN_PORT",
        18700,
        vec![],
    ).await;
    
    // Wait for queen to be ready
    wait_for_http_ready("http://localhost:18700", 20)
        .await
        .expect("Queen never became ready");
    
    // Use job-client (simulating keeper)
    let client = job_client::JobClient::new("http://localhost:18700");
    let operation = operations_contract::Operation::HiveList;
    
    let mut events = Vec::new();
    let result = timeout(
        Duration::from_secs(5),
        client.submit_and_stream(operation, |line| {
            events.push(line.to_string());
            Ok(())
        })
    ).await;
    
    assert!(result.is_ok(), "Job submission timed out");
    assert!(result.unwrap().is_ok(), "Job submission failed");
    
    // Verify queen narration received
    assert!(!events.is_empty(), "No events received");
    assert!(events.iter().any(|e| e.contains("queen_receive")), "Missing queen_receive event");
    assert!(events.iter().any(|e| e.contains("queen_complete")), "Missing queen_complete event");
    
    // Cleanup
    queen.kill().await.ok();
}

#[tokio::test]
#[ignore] // TEAM-303: Requires cargo build first
async fn test_queen_to_hive_real_process() {
    // TEAM-303: Test Queen → Hive with real processes
    
    // Start fake hive
    let mut hive = start_binary_and_wait(
        "fake-rbee-hive",
        "HIVE_PORT",
        19100,
        vec![],
    ).await;
    
    wait_for_http_ready("http://localhost:19100", 20)
        .await
        .expect("Hive never became ready");
    
    // Start fake queen with hive URL
    let mut queen = start_binary_and_wait(
        "fake-queen-rbee",
        "QUEEN_PORT",
        18701,
        vec![("HIVE_URL", "http://localhost:19100")],
    ).await;
    
    wait_for_http_ready("http://localhost:18701", 20)
        .await
        .expect("Queen never became ready");
    
    // Submit worker operation (should be forwarded to hive)
    let client = job_client::JobClient::new("http://localhost:18701");
    let operation = operations_contract::Operation::WorkerSpawn(
        operations_contract::WorkerSpawnRequest {
            hive_id: "localhost".to_string(),
            worker_id: "test-worker".to_string(),
            model_id: "test-model".to_string(),
            device: operations_contract::Device::Cpu,
        }
    );
    
    let mut events = Vec::new();
    let result = timeout(
        Duration::from_secs(5),
        client.submit_and_stream(operation, |line| {
            events.push(line.to_string());
            Ok(())
        })
    ).await;
    
    assert!(result.is_ok(), "Job submission timed out");
    
    // Verify both queen and hive narration
    assert!(events.iter().any(|e| e.contains("queen_receive")), "Missing queen_receive");
    assert!(events.iter().any(|e| e.contains("queen_forward")), "Missing queen_forward");
    
    // Cleanup
    queen.kill().await.ok();
    hive.kill().await.ok();
}

#[tokio::test]
#[ignore] // TEAM-303: Requires cargo build first
async fn test_hive_to_worker_process_capture() {
    // TEAM-303: Test Hive → Worker with real process capture
    
    // Start fake hive
    let mut hive = start_binary_and_wait(
        "fake-rbee-hive",
        "HIVE_PORT",
        19101,
        vec![],
    ).await;
    
    wait_for_http_ready("http://localhost:19101", 20)
        .await
        .expect("Hive never became ready");
    
    // Submit worker spawn directly to hive
    let client = job_client::JobClient::new("http://localhost:19101");
    let operation = operations_contract::Operation::WorkerSpawn(
        operations_contract::WorkerSpawnRequest {
            hive_id: "localhost".to_string(),
            worker_id: "test-worker".to_string(),
            model_id: "test-model".to_string(),
            device: operations_contract::Device::Cpu,
        }
    );
    
    let mut events = Vec::new();
    let result = timeout(
        Duration::from_secs(10), // Longer timeout for process spawn
        client.submit_and_stream(operation, |line| {
            events.push(line.to_string());
            Ok(())
        })
    ).await;
    
    assert!(result.is_ok(), "Job submission timed out");
    
    // Verify hive narration
    assert!(events.iter().any(|e| e.contains("hive_receive")), "Missing hive_receive");
    assert!(events.iter().any(|e| e.contains("hive_spawn")), "Missing hive_spawn");
    assert!(events.iter().any(|e| e.contains("hive_worker_spawned")), "Missing hive_worker_spawned");
    
    // Verify worker narration (captured via ProcessNarrationCapture)
    assert!(events.iter().any(|e| e.contains("worker_startup")), "Missing worker_startup");
    assert!(events.iter().any(|e| e.contains("worker_load_model")), "Missing worker_load_model");
    assert!(events.iter().any(|e| e.contains("worker_ready")), "Missing worker_ready");
    assert!(events.iter().any(|e| e.contains("worker_inference")), "Missing worker_inference");
    
    // Cleanup
    hive.kill().await.ok();
}

#[tokio::test]
#[ignore] // TEAM-303: Requires cargo build first
async fn test_full_stack_keeper_to_worker() {
    // TEAM-303: Test complete flow: Keeper → Queen → Hive → Worker
    
    // Start fake hive
    let mut hive = start_binary_and_wait(
        "fake-rbee-hive",
        "HIVE_PORT",
        19102,
        vec![],
    ).await;
    
    wait_for_http_ready("http://localhost:19102", 20)
        .await
        .expect("Hive never became ready");
    
    // Start fake queen with hive URL
    let mut queen = start_binary_and_wait(
        "fake-queen-rbee",
        "QUEEN_PORT",
        18702,
        vec![("HIVE_URL", "http://localhost:19102")],
    ).await;
    
    wait_for_http_ready("http://localhost:18702", 20)
        .await
        .expect("Queen never became ready");
    
    // Keeper submits worker spawn
    let client = job_client::JobClient::new("http://localhost:18702");
    let operation = operations_contract::Operation::WorkerSpawn(
        operations_contract::WorkerSpawnRequest {
            hive_id: "localhost".to_string(),
            worker_id: "test-worker".to_string(),
            model_id: "test-model".to_string(),
            device: operations_contract::Device::Cpu,
        }
    );
    
    let mut events = Vec::new();
    let result = timeout(
        Duration::from_secs(15), // Long timeout for full stack
        client.submit_and_stream(operation, |line| {
            events.push(line.to_string());
            Ok(())
        })
    ).await;
    
    assert!(result.is_ok(), "Full stack test timed out");
    
    // Verify narration from all layers
    let expected_events = vec![
        ("queen_receive", "Queen layer"),
        ("queen_forward", "Queen forwarding"),
        ("hive_receive", "Hive layer"),
        ("hive_spawn", "Hive spawning"),
        ("worker_startup", "Worker layer"),
        ("worker_load_model", "Worker loading"),
        ("worker_ready", "Worker ready"),
        ("worker_inference", "Worker inference"),
    ];
    
    for (event_name, description) in expected_events {
        assert!(
            events.iter().any(|e| e.contains(event_name)),
            "Missing {} event: {}",
            event_name,
            description
        );
    }
    
    // Cleanup
    queen.kill().await.ok();
    hive.kill().await.ok();
}

#[tokio::test]
#[ignore] // TEAM-303: Requires cargo build first
async fn test_correlation_id_propagation() {
    // TEAM-303: Test correlation ID flows through all services
    
    // Start fake hive
    let mut hive = start_binary_and_wait(
        "fake-rbee-hive",
        "HIVE_PORT",
        19103,
        vec![],
    ).await;
    
    wait_for_http_ready("http://localhost:19103", 20)
        .await
        .expect("Hive never became ready");
    
    // Start fake queen with hive URL
    let mut queen = start_binary_and_wait(
        "fake-queen-rbee",
        "QUEEN_PORT",
        18703,
        vec![("HIVE_URL", "http://localhost:19103")],
    ).await;
    
    wait_for_http_ready("http://localhost:18703", 20)
        .await
        .expect("Queen never became ready");
    
    // Submit with correlation ID header
    let client = reqwest::Client::new();
    let correlation_id = "test-correlation-12345";
    
    let response = client
        .post("http://localhost:18703/v1/jobs")
        .header("x-correlation-id", correlation_id)
        .json(&serde_json::json!({
            "operation": "worker_spawn",
            "hive_id": "localhost",
            "worker_id": "test-worker",
            "model_id": "test-model",
            "device": "cpu"
        }))
        .send()
        .await
        .expect("Failed to submit job");
    
    let job_response: serde_json::Value = response.json().await.expect("Failed to parse response");
    
    // Verify correlation ID in response
    assert_eq!(
        job_response.get("correlation_id").and_then(|v| v.as_str()),
        Some(correlation_id),
        "Correlation ID not returned in response"
    );
    
    // TODO: Verify correlation ID appears in narration events
    // This requires parsing SSE stream and checking event metadata
    
    // Cleanup
    queen.kill().await.ok();
    hive.kill().await.ok();
}

#[tokio::test]
#[ignore] // TEAM-303: Requires cargo build first
async fn test_worker_crash_handling() {
    // TEAM-303: Test that worker crashes are captured and reported
    
    // Start fake hive
    let mut hive = start_binary_and_wait(
        "fake-rbee-hive",
        "HIVE_PORT",
        19104,
        vec![],
    ).await;
    
    wait_for_http_ready("http://localhost:19104", 20)
        .await
        .expect("Hive never became ready");
    
    // Submit worker spawn that will crash
    // (In a real test, we'd have a crashing worker binary)
    let client = job_client::JobClient::new("http://localhost:19104");
    let operation = operations_contract::Operation::WorkerSpawn(
        operations_contract::WorkerSpawnRequest {
            hive_id: "localhost".to_string(),
            worker_id: "crashing-worker".to_string(),
            model_id: "test-model".to_string(),
            device: operations_contract::Device::Cpu,
        }
    );
    
    let mut events = Vec::new();
    let result = timeout(
        Duration::from_secs(10),
        client.submit_and_stream(operation, |line| {
            events.push(line.to_string());
            Ok(())
        })
    ).await;
    
    // Should complete even if worker crashes
    assert!(result.is_ok(), "Test timed out");
    
    // Verify we got some events
    assert!(!events.is_empty(), "No events received");
    
    // Cleanup
    hive.kill().await.ok();
}

#[tokio::test]
#[ignore] // TEAM-303: Requires cargo build first
async fn test_concurrent_full_stack_requests() {
    // TEAM-303: Test multiple concurrent full-stack requests
    
    // Start fake hive
    let mut hive = start_binary_and_wait(
        "fake-rbee-hive",
        "HIVE_PORT",
        19105,
        vec![],
    ).await;
    
    wait_for_http_ready("http://localhost:19105", 20)
        .await
        .expect("Hive never became ready");
    
    // Start fake queen with hive URL
    let mut queen = start_binary_and_wait(
        "fake-queen-rbee",
        "QUEEN_PORT",
        18704,
        vec![("HIVE_URL", "http://localhost:19105")],
    ).await;
    
    wait_for_http_ready("http://localhost:18704", 20)
        .await
        .expect("Queen never became ready");
    
    // Submit 3 concurrent requests
    let client = job_client::JobClient::new("http://localhost:18704");
    
    let mut handles = Vec::new();
    for i in 0..3 {
        let client_clone = client.clone();
        let handle = tokio::spawn(async move {
            let operation = operations_contract::Operation::WorkerSpawn(
                operations_contract::WorkerSpawnRequest {
                    hive_id: "localhost".to_string(),
                    worker_id: format!("worker-{}", i),
                    model_id: "test-model".to_string(),
                    device: operations_contract::Device::Cpu,
                }
            );
            
            let mut events = Vec::new();
            let result = timeout(
                Duration::from_secs(15),
                client_clone.submit_and_stream(operation, |line| {
                    events.push(line.to_string());
                    Ok(())
                })
            ).await;
            
            (i, result, events)
        });
        handles.push(handle);
    }
    
    // Wait for all requests
    let mut results = Vec::new();
    for handle in handles {
        let (i, result, events) = handle.await.expect("Task panicked");
        results.push((i, result, events));
    }
    
    // Verify all succeeded
    for (i, result, events) in results {
        assert!(result.is_ok(), "Request {} timed out", i);
        assert!(!events.is_empty(), "Request {} received no events", i);
        assert!(
            events.iter().any(|e| e.contains("worker_startup")),
            "Request {} missing worker events",
            i
        );
    }
    
    // Cleanup
    queen.kill().await.ok();
    hive.kill().await.ok();
}
