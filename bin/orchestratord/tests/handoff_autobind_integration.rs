//! Integration tests for handoff autobind watcher.
//!
//! These tests verify that orchestratord can:
//! 1. Watch for handoff JSON files written by engine-provisioner
//! 2. Parse and bind adapters from handoff files
//! 3. Update pool-managerd registry with readiness info
//! 4. Serve SSE streams after autobind completes
//!
//! Note: These tests use sleep-based synchronization and may be flaky under load.
//! Consider using a notification mechanism for production-grade testing.

use orchestratord::services::handoff;
use orchestratord::state::AppState;
use std::fs;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

/// Integration test: Start orchestrator with file watcher, write a valid handoff JSON,
/// assert GET /control/pools/{id}/health reflects Live+Ready and engine_version.
#[tokio::test]
async fn test_handoff_autobind_integration_stub_engine() {
    let state = AppState::new();
    let temp = TempDir::new().unwrap();
    let runtime_dir = temp.path().to_str().unwrap();

    // Set environment for watcher
    std::env::set_var("ORCHD_RUNTIME_DIR", runtime_dir);
    std::env::set_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS", "100");

    // Spawn watcher
    handoff::spawn_handoff_autobind_watcher(state.clone());

    // Give watcher time to start
    sleep(Duration::from_millis(150)).await;

    // Write a handoff file
    let handoff = serde_json::json!({
        "url": "http://127.0.0.1:19999",
        "pool_id": "integration-pool",
        "replica_id": "r1",
        "engine_version": "llamacpp-integration-v1",
        "device_mask": "GPU0",
        "slots_total": 8,
        "slots_free": 8
    });
    let handoff_path = temp.path().join("integration-handoff.json");
    fs::write(&handoff_path, serde_json::to_string(&handoff).unwrap()).unwrap();

    // Wait for watcher to pick up the file
    sleep(Duration::from_millis(250)).await;

    // Verify pool registry updated
    {
        let reg = state.pool_manager.lock().unwrap();
        let health = reg.get_health("integration-pool");
        assert!(health.is_some(), "pool not registered after handoff");
        let h = health.unwrap();
        assert!(h.live, "pool not live");
        assert!(h.ready, "pool not ready");
        assert_eq!(
            reg.get_engine_version("integration-pool").as_deref(),
            Some("llamacpp-integration-v1"),
            "engine version mismatch"
        );
        assert_eq!(reg.get_slots_total("integration-pool"), Some(8));
        assert_eq!(reg.get_slots_free("integration-pool"), Some(8));
        assert!(reg.get_heartbeat("integration-pool").is_some(), "heartbeat not set");
    }

    // Verify bound_pools tracking
    {
        let bound = state.bound_pools.lock().unwrap();
        assert!(bound.contains("integration-pool:r1"), "pool not marked as bound");
    }

    // Clean up env vars
    std::env::remove_var("ORCHD_RUNTIME_DIR");
    std::env::remove_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS");
}

/// Integration test: Multiple handoff files should all be processed
#[tokio::test]
async fn test_handoff_autobind_multiple_pools() {
    let state = AppState::new();
    let temp = TempDir::new().unwrap();
    let runtime_dir = temp.path().to_str().unwrap();

    std::env::set_var("ORCHD_RUNTIME_DIR", runtime_dir);
    std::env::set_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS", "100");

    handoff::spawn_handoff_autobind_watcher(state.clone());
    sleep(Duration::from_millis(150)).await;

    // Write multiple handoff files
    for i in 0..3 {
        let handoff = serde_json::json!({
            "url": format!("http://127.0.0.1:{}", 20000 + i),
            "pool_id": format!("pool-{}", i),
            "replica_id": "r0",
            "engine_version": format!("v{}", i),
            "slots_total": 4,
            "slots_free": 4
        });
        let path = temp.path().join(format!("handoff-{}.json", i));
        fs::write(&path, serde_json::to_string(&handoff).unwrap()).unwrap();
    }

    // Wait for watcher to process all files
    sleep(Duration::from_millis(500)).await;

    // Verify all pools registered
    {
        let reg = state.pool_manager.lock().unwrap();
        for i in 0..3 {
            let pool_id = format!("pool-{}", i);
            let health = reg.get_health(&pool_id);
            assert!(health.is_some(), "pool {} not registered", i);
            assert!(health.unwrap().ready, "pool {} not ready", i);
            assert_eq!(
                reg.get_engine_version(&pool_id).as_deref(),
                Some(&format!("v{}", i) as &str),
                "pool {} version mismatch",
                i
            );
        }
    }

    std::env::remove_var("ORCHD_RUNTIME_DIR");
    std::env::remove_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS");
}

/// Integration test: Non-JSON files should be ignored
#[tokio::test]
async fn test_handoff_autobind_ignores_non_json() {
    let state = AppState::new();
    let temp = TempDir::new().unwrap();
    let runtime_dir = temp.path().to_str().unwrap();

    std::env::set_var("ORCHD_RUNTIME_DIR", runtime_dir);
    std::env::set_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS", "100");

    handoff::spawn_handoff_autobind_watcher(state.clone());
    sleep(Duration::from_millis(150)).await;

    // Write a non-JSON file
    let txt_path = temp.path().join("not-a-handoff.txt");
    fs::write(&txt_path, "this is not json").unwrap();

    // Write a valid handoff
    let handoff = serde_json::json!({
        "url": "http://127.0.0.1:21000",
        "pool_id": "valid-pool",
        "slots_total": 2,
        "slots_free": 2
    });
    let json_path = temp.path().join("valid.json");
    fs::write(&json_path, serde_json::to_string(&handoff).unwrap()).unwrap();

    sleep(Duration::from_millis(400)).await;

    // Only the valid JSON pool should be registered
    {
        let reg = state.pool_manager.lock().unwrap();
        assert!(reg.get_health("valid-pool").is_some(), "valid pool not registered");
        // No pool should exist for the txt file
    }

    std::env::remove_var("ORCHD_RUNTIME_DIR");
    std::env::remove_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS");
}

/// Integration test: Idempotency - processing same handoff twice should not error
#[tokio::test]
async fn test_handoff_autobind_idempotent() {
    let state = AppState::new();
    let temp = TempDir::new().unwrap();
    let runtime_dir = temp.path().to_str().unwrap();

    std::env::set_var("ORCHD_RUNTIME_DIR", runtime_dir);
    std::env::set_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS", "100");

    handoff::spawn_handoff_autobind_watcher(state.clone());
    sleep(Duration::from_millis(150)).await;

    let handoff = serde_json::json!({
        "url": "http://127.0.0.1:22000",
        "pool_id": "idempotent-pool",
        "replica_id": "r0",
        "engine_version": "v1",
        "slots_total": 4,
        "slots_free": 4
    });
    let path = temp.path().join("idempotent.json");
    fs::write(&path, serde_json::to_string(&handoff).unwrap()).unwrap();

    // First processing
    sleep(Duration::from_millis(250)).await;

    let first_heartbeat = {
        let reg = state.pool_manager.lock().unwrap();
        reg.get_heartbeat("idempotent-pool")
    };
    assert!(first_heartbeat.is_some(), "first processing failed");

    // Wait and let watcher process again (should be idempotent)
    sleep(Duration::from_millis(250)).await;

    // Should still be registered and healthy
    {
        let reg = state.pool_manager.lock().unwrap();
        let health = reg.get_health("idempotent-pool");
        assert!(health.is_some(), "pool disappeared after re-processing");
        assert!(health.unwrap().ready, "pool not ready after re-processing");
    }

    std::env::remove_var("ORCHD_RUNTIME_DIR");
    std::env::remove_var("ORCHD_HANDOFF_WATCH_INTERVAL_MS");
}
