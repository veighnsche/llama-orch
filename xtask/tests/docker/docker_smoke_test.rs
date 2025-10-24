// Docker smoke tests
// Purpose: Basic connectivity and health checks for queen-rbee → rbee-hive communication
// Run with: cargo test --package xtask --test docker_smoke_test --ignored

use std::time::Duration;
use xtask::integration::docker_harness::{DockerTestHarness, Topology};

#[tokio::test]
#[ignore]
async fn test_docker_queen_health() {
    // Setup
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Queen health check
    let response = ureq::get("http://localhost:8500/health")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Queen health check failed");

    assert_eq!(response.status(), 200);
    assert_eq!(response.into_string().unwrap(), "ok");

    println!("✅ Queen health check passed");
}

#[tokio::test]
#[ignore]
async fn test_docker_hive_health() {
    // Setup
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Hive health check
    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Hive health check failed");

    assert_eq!(response.status(), 200);
    assert_eq!(response.into_string().unwrap(), "ok");

    println!("✅ Hive health check passed");
}

#[tokio::test]
#[ignore]
async fn test_docker_hive_capabilities() {
    // Setup
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Hive capabilities endpoint
    let response = ureq::get("http://localhost:9000/capabilities")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Hive capabilities check failed");

    assert_eq!(response.status(), 200);

    let json: serde_json::Value = response.into_json().unwrap();
    assert!(json["devices"].is_array(), "Response should have devices array");

    let devices = json["devices"].as_array().unwrap();
    assert!(devices.len() > 0, "Should have at least one device (CPU)");

    println!("✅ Hive capabilities check passed ({} devices)", devices.len());
}

#[tokio::test]
#[ignore]
async fn test_docker_ssh_connection() {
    // Setup
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Execute command via docker exec (simulates SSH)
    let output = harness
        .exec("rbee-hive-localhost", &["echo", "test"])
        .await
        .expect("Failed to execute command in container");

    assert_eq!(output.trim(), "test");

    println!("✅ SSH connection test passed");
}

#[tokio::test]
#[ignore]
async fn test_docker_hive_binary_exists() {
    // Setup
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Check if rbee-hive binary exists and is executable
    let output = harness
        .exec("rbee-hive-localhost", &["ls", "-la", "/home/rbee/.local/bin/rbee-hive"])
        .await
        .expect("Failed to check binary");

    assert!(output.contains("rbee-hive"), "Binary should exist");
    assert!(output.contains("-rwx"), "Binary should be executable");

    println!("✅ Hive binary check passed");
}

#[tokio::test]
#[ignore]
async fn test_docker_all_services() {
    // Setup
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    println!("🧪 Testing all services...");

    // Test 1: Queen health
    let response = ureq::get("http://localhost:8500/health")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Queen health check failed");
    assert_eq!(response.status(), 200);
    println!("  ✅ Queen health: OK");

    // Test 2: Hive health
    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Hive health check failed");
    assert_eq!(response.status(), 200);
    println!("  ✅ Hive health: OK");

    // Test 3: Hive capabilities
    let response = ureq::get("http://localhost:9000/capabilities")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Hive capabilities check failed");
    assert_eq!(response.status(), 200);
    let json: serde_json::Value = response.into_json().unwrap();
    assert!(json["devices"].is_array());
    println!("  ✅ Hive capabilities: OK ({} devices)", json["devices"].as_array().unwrap().len());

    println!("✅ All services test passed");
}
