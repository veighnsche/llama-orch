// Failure scenario tests
// Purpose: Test failure handling and recovery in queen-rbee → rbee-hive communication
// Run with: cargo test --package xtask --test failure_tests --ignored

use std::time::Duration;
use xtask::integration::docker_harness::{DockerTestHarness, Topology};

#[tokio::test]
#[ignore]
async fn test_hive_crash_during_operation() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Verify hive is healthy
    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();
    assert_eq!(response.status(), 200);

    // Kill hive mid-operation
    harness.kill("rbee-hive-localhost").await.unwrap();

    // Wait for container to die
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Verify connection fails
    let result = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(2))
        .call();

    assert!(result.is_err(), "Connection should fail after hive crash");

    println!("✅ Hive crash test passed");
}

#[tokio::test]
#[ignore]
async fn test_hive_restart_recovery() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Verify hive is healthy
    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();
    assert_eq!(response.status(), 200);

    // Restart hive
    harness.restart("rbee-hive-localhost").await.unwrap();

    // Wait for hive to restart
    println!("⏳ Waiting for hive to restart...");
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Verify hive is healthy again
    DockerTestHarness::wait_for_http("http://localhost:9000/health", Duration::from_secs(30))
        .await
        .unwrap();

    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();
    assert_eq!(response.status(), 200);

    println!("✅ Hive restart recovery test passed");
}

#[tokio::test]
#[ignore]
async fn test_queen_restart_recovery() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Verify queen is healthy
    let response = ureq::get("http://localhost:8500/health")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();
    assert_eq!(response.status(), 200);

    // Restart queen
    harness.restart("rbee-queen-localhost").await.unwrap();

    // Wait for queen to restart
    println!("⏳ Waiting for queen to restart...");
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Verify queen is healthy again
    DockerTestHarness::wait_for_http("http://localhost:8500/health", Duration::from_secs(30))
        .await
        .unwrap();

    let response = ureq::get("http://localhost:8500/health")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();
    assert_eq!(response.status(), 200);

    println!("✅ Queen restart recovery test passed");
}

#[tokio::test]
#[ignore]
async fn test_concurrent_operations_on_same_hive() {
    let _harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Send 10 concurrent requests to same hive
    let mut handles = vec![];

    for i in 0..10 {
        let handle = tokio::spawn(async move {
            let response = ureq::get("http://localhost:9000/capabilities")
                .timeout(Duration::from_secs(5))
                .call()
                .unwrap_or_else(|e| panic!("Request {} failed: {}", i, e));

            assert_eq!(response.status(), 200);
            let json: serde_json::Value = response.into_json()
                .unwrap_or_else(|e| panic!("Failed to parse JSON for request {}: {}", i, e));
            assert!(json["devices"].is_array());
            i
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }

    println!("✅ Concurrent operations test passed (10 operations)");
}

#[tokio::test]
#[ignore]
async fn test_rapid_restart_cycle() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Test: Restart hive 3 times rapidly
    for i in 1..=3 {
        println!("🔄 Restart cycle {}/3", i);

        harness.restart("rbee-hive-localhost").await.unwrap();

        // Wait for hive to restart
        tokio::time::sleep(Duration::from_secs(10)).await;

        // Verify hive is healthy
        DockerTestHarness::wait_for_http("http://localhost:9000/health", Duration::from_secs(30))
            .await
            .unwrap();

        let response = ureq::get("http://localhost:9000/health")
            .timeout(Duration::from_secs(5))
            .call()
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    println!("✅ Rapid restart cycle test passed (3 cycles)");
}

#[tokio::test]
#[ignore]
async fn test_service_logs_after_failure() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Kill hive
    harness.kill("rbee-hive-localhost").await.unwrap();

    // Wait a bit
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Get logs
    let logs = harness.logs("rbee-hive-localhost").await.unwrap();

    // Verify logs contain startup messages
    assert!(logs.len() > 0, "Logs should not be empty");
    assert!(logs.contains("rbee") || logs.contains("Starting") || logs.contains("Listening"), 
            "Logs should contain service startup messages");

    println!("✅ Service logs test passed ({} bytes)", logs.len());
}
