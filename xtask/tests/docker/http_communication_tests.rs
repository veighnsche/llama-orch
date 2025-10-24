// HTTP communication tests
// Purpose: Test all HTTP communication patterns between queen-rbee and rbee-hive
// Run with: cargo test --package xtask --test http_communication_tests --ignored

use std::time::Duration;
use xtask::integration::docker_harness::{DockerTestHarness, Topology};

#[tokio::test]
#[ignore]
async fn test_queen_to_hive_health_check() {
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Queen calls hive health endpoint
    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(response.into_string().unwrap(), "ok");

    println!("✅ Queen → Hive health check passed");
}

#[tokio::test]
#[ignore]
async fn test_queen_to_hive_capabilities() {
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Queen fetches capabilities from hive
    let response = ureq::get("http://localhost:9000/capabilities")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();

    assert_eq!(response.status(), 200);

    let json: serde_json::Value = response.into_json().unwrap();
    assert!(json["devices"].is_array());

    let devices = json["devices"].as_array().unwrap();
    assert!(devices.len() > 0, "Should have at least CPU device");

    // Verify device structure
    let first_device = &devices[0];
    assert!(first_device["id"].is_string());
    assert!(first_device["name"].is_string());
    assert!(first_device["device_type"].is_string());

    println!("✅ Queen → Hive capabilities check passed ({} devices)", devices.len());
}

#[tokio::test]
#[ignore]
async fn test_http_connection_timeout() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Kill hive to simulate timeout
    harness.kill("rbee-hive-localhost").await.unwrap();

    // Wait a bit for container to die
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Test: Connection should timeout/fail
    let result = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(2))
        .call();

    assert!(result.is_err(), "Connection should fail when hive is down");

    println!("✅ HTTP timeout handling passed");
}

#[tokio::test]
#[ignore]
async fn test_http_connection_refused() {
    // Don't start harness - no containers running
    // WARNING: This test is flaky if port 9000 is already in use!
    // Consider using a random port or checking if port is free first.

    // Test: Connection should be refused
    let result = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(2))
        .call();

    assert!(result.is_err(), "Connection should be refused when no hive running");

    println!("✅ HTTP connection refused test passed");
}

#[tokio::test]
#[ignore]
async fn test_http_concurrent_requests() {
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Send 10 concurrent requests
    let mut handles = vec![];

    for i in 0..10 {
        let handle = tokio::spawn(async move {
            let response = ureq::get("http://localhost:9000/health")
                .timeout(Duration::from_secs(5))
                .call()
                .unwrap_or_else(|e| panic!("Request {} failed: {}", i, e));

            assert_eq!(response.status(), 200);
            i
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }

    println!("✅ HTTP concurrent requests test passed (10 requests)");
}

#[tokio::test]
#[ignore]
async fn test_http_large_response() {
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Capabilities response can be large with many devices
    let response = ureq::get("http://localhost:9000/capabilities")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();

    let body = response.into_string().unwrap();
    assert!(body.len() > 0, "Response should not be empty");
    assert!(body.len() < 1_000_000, "Response should be reasonable size");

    println!("✅ HTTP large response test passed ({} bytes)", body.len());
}
