// TEAM-243: Concurrent access tests for hive-registry
// Purpose: Verify thread-safe concurrent operations on hive registry
// Scale: Reasonable for NUC (5-10 concurrent, 100 hives total)
// Historical Context: TEAM-243 implemented Priority 1 critical tests for hive lifecycle

use queen_rbee_hive_registry::HiveRegistry;
use rbee_heartbeat::{HiveHeartbeatPayload, WorkerState};
use std::sync::Arc;

/// Helper to create a test heartbeat payload
fn create_test_heartbeat(hive_id: &str, worker_count: usize) -> HiveHeartbeatPayload {
    let workers = (0..worker_count)
        .map(|i| WorkerState {
            worker_id: format!("worker-{}-{}", hive_id, i),
            state: "ready".to_string(),
            last_heartbeat: chrono::Utc::now().to_rfc3339(),
            health_status: "healthy".to_string(),
            url: format!("http://localhost:800{}", i),
            model_id: Some("llama2".to_string()),
            backend: Some("llama-cpp".to_string()),
            device_id: Some(0),
            vram_bytes: Some(8_000_000_000),
            ram_bytes: Some(16_000_000_000),
            cpu_percent: Some(10.0),
            gpu_percent: Some(50.0),
        })
        .collect();

    HiveHeartbeatPayload {
        hive_id: hive_id.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        workers,
    }
}

/// Test concurrent hive state updates (10 concurrent)
#[tokio::test]
async fn test_concurrent_hive_state_updates() {
    let registry = Arc::new(HiveRegistry::new());
    let mut handles = vec![];

    // Update 10 different hives concurrently
    for i in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            let hive_id = format!("hive-{}", i);
            let payload = create_test_heartbeat(&hive_id, 3);
            registry_clone.update_hive_state(&hive_id, payload);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all hives were updated
    assert_eq!(registry.hive_count(), 10);
    println!("✓ 10 concurrent hive state updates completed successfully");
}

/// Test concurrent updates to same hive
#[tokio::test]
async fn test_concurrent_updates_same_hive() {
    let registry = Arc::new(HiveRegistry::new());
    let hive_id = "localhost";
    let mut handles = vec![];

    // Update same hive 5 times concurrently
    for i in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let hive_id_clone = hive_id.to_string();
        let handle = tokio::spawn(async move {
            let payload = create_test_heartbeat(&hive_id_clone, i + 1);
            registry_clone.update_hive_state(&hive_id_clone, payload);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify hive exists and has valid state
    assert_eq!(registry.hive_count(), 1);
    assert!(registry.get_hive_state(hive_id).is_some());
    println!("✓ 5 concurrent updates to same hive completed successfully");
}

/// Test concurrent list_active_hives queries
#[tokio::test]
async fn test_concurrent_list_active_hives() {
    let registry = Arc::new(HiveRegistry::new());

    // Create 10 hives
    for i in 0..10 {
        let hive_id = format!("hive-{}", i);
        let payload = create_test_heartbeat(&hive_id, 2);
        registry.update_hive_state(&hive_id, payload);
    }

    let mut handles = vec![];

    // Query active hives 10 times concurrently
    for _ in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            registry_clone.list_active_hives(30_000) // 30 second timeout
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut results = vec![];
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    // All queries should return 10 hives
    assert!(results.iter().all(|r| r.len() == 10));
    println!("✓ 10 concurrent list_active_hives queries completed successfully");
}

/// Test memory efficiency with 100 hives
#[tokio::test]
async fn test_memory_efficiency_100_hives() {
    let registry = Arc::new(HiveRegistry::new());

    // Create 100 hives
    for i in 0..100 {
        let hive_id = format!("hive-{}", i);
        let payload = create_test_heartbeat(&hive_id, 2);
        registry.update_hive_state(&hive_id, payload);
    }

    assert_eq!(registry.hive_count(), 100);

    // Query all hives
    let all_hives = registry.list_all_hives();
    assert_eq!(all_hives.len(), 100);

    // Remove all hives
    for hive_id in all_hives {
        registry.remove_hive(&hive_id);
    }

    // Verify memory was freed
    assert_eq!(registry.hive_count(), 0);
    println!("✓ 100 hives created, queried, and removed successfully");
}
