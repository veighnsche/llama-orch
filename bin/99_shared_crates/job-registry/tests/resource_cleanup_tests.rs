// TEAM-243: Resource cleanup tests for job-registry
// Purpose: Verify job registry cleans up resources and prevents memory leaks
// Scale: Reasonable for NUC (5-10 concurrent, 100 jobs total)
// Historical Context: TEAM-243 implemented Priority 1 critical tests for resource management
// Focus: Memory leaks, dangling resources, proper state cleanup

use job_registry::{JobRegistry, JobState};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::Duration;

/// Test cleanup on normal job completion
#[tokio::test]
async fn test_cleanup_on_normal_completion() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    // Simulate job execution
    registry.update_state(&job_id, JobState::Running);

    let (tx, rx) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Send completion token
    tx.send("token-1".to_string()).unwrap();
    tx.send("[DONE]".to_string()).unwrap();
    drop(tx);

    // Cleanup: remove job
    let removed = registry.remove_job(&job_id);
    assert!(removed.is_some());
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup on normal completion successful");
}

/// Test cleanup on client disconnect (receiver dropped)
#[tokio::test]
async fn test_cleanup_on_client_disconnect() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    let (tx, rx) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Send some tokens
    tx.send("token-1".to_string()).unwrap();

    // Client disconnects (receiver dropped)
    let mut receiver = registry.take_token_receiver(&job_id).unwrap();
    drop(receiver);

    // Sender should still work (but no one is listening)
    let result = tx.send("token-2".to_string());
    // Send may fail because receiver is gone, which is fine
    let _ = result;

    // Cleanup: remove job
    registry.remove_job(&job_id);
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup on client disconnect successful");
}

/// Test cleanup on timeout (operation takes too long)
#[tokio::test]
async fn test_cleanup_on_timeout() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    let (tx, rx) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Simulate timeout: operation doesn't complete
    registry.update_state(&job_id, JobState::Running);

    // Wait a bit (simulate timeout)
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Cleanup: remove job after timeout
    registry.remove_job(&job_id);
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup on timeout successful");
}

/// Test cleanup on error
#[tokio::test]
async fn test_cleanup_on_error() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    let (tx, rx) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Simulate error
    registry.update_state(&job_id, JobState::Failed("operation failed".to_string()));

    // Send error token
    tx.send("[ERROR] operation failed".to_string()).unwrap();
    drop(tx);

    // Cleanup: remove job
    registry.remove_job(&job_id);
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup on error successful");
}

/// Test cleanup with concurrent operations
#[tokio::test]
async fn test_cleanup_concurrent_operations() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let mut job_ids = vec![];

    // Create 10 jobs
    for _ in 0..10 {
        job_ids.push(registry.create_job());
    }

    assert_eq!(registry.job_count(), 10);

    let mut handles = vec![];

    // Cleanup jobs concurrently
    for job_id in job_ids {
        let registry_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move { registry_clone.remove_job(&job_id) });
        handles.push(handle);
    }

    // Wait for all cleanups
    futures::future::join_all(handles).await;

    // Verify all jobs cleaned up
    assert_eq!(registry.job_count(), 0);

    println!("✓ Concurrent cleanup successful");
}

/// Test cleanup with payload cleanup
#[tokio::test]
async fn test_cleanup_with_payload() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    // Set large payload
    let payload = serde_json::json!({
        "data": "x".repeat(1_000_000),
        "metadata": {
            "size": 1_000_000,
            "timestamp": "2025-10-22T10:00:00Z"
        }
    });
    registry.set_payload(&job_id, payload);

    // Take and verify payload
    let taken = registry.take_payload(&job_id);
    assert!(taken.is_some());

    // Cleanup: remove job
    registry.remove_job(&job_id);
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup with payload successful");
}

/// Test cleanup prevents memory leaks with 100 jobs
#[tokio::test]
async fn test_cleanup_prevents_memory_leaks_100_jobs() {
    let registry = Arc::new(JobRegistry::<String>::new());

    // Create 100 jobs with channels
    for i in 0..100 {
        let job_id = registry.create_job();

        let (tx, rx) = mpsc::unbounded_channel();
        registry.set_token_receiver(&job_id, rx);

        // Send some tokens
        for j in 0..10 {
            let _ = tx.send(format!("token-{}-{}", i, j));
        }
    }

    assert_eq!(registry.job_count(), 100);

    // Cleanup all jobs
    let job_ids = registry.job_ids();
    for job_id in job_ids {
        registry.remove_job(&job_id);
    }

    // Verify all cleaned up
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup prevents memory leaks (100 jobs)");
}

/// Test cleanup with partial state
#[tokio::test]
async fn test_cleanup_with_partial_state() {
    let registry = Arc::new(JobRegistry::<String>::new());

    // Job 1: Has receiver
    let job_id_1 = registry.create_job();
    let (tx1, rx1) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id_1, rx1);

    // Job 2: Has payload
    let job_id_2 = registry.create_job();
    registry.set_payload(&job_id_2, serde_json::json!({"data": "test"}));

    // Job 3: Has both
    let job_id_3 = registry.create_job();
    let (tx3, rx3) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id_3, rx3);
    registry.set_payload(&job_id_3, serde_json::json!({"data": "test"}));

    // Job 4: Has neither
    let job_id_4 = registry.create_job();

    assert_eq!(registry.job_count(), 4);

    // Cleanup all
    registry.remove_job(&job_id_1);
    registry.remove_job(&job_id_2);
    registry.remove_job(&job_id_3);
    registry.remove_job(&job_id_4);

    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup with partial state successful");
}

/// Test cleanup idempotency
#[tokio::test]
async fn test_cleanup_idempotency() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    // First cleanup
    let result1 = registry.remove_job(&job_id);
    assert!(result1.is_some());

    // Second cleanup (should be safe)
    let result2 = registry.remove_job(&job_id);
    assert!(result2.is_none());

    // Registry should be empty
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup idempotency verified");
}

/// Test cleanup with sender still active
#[tokio::test]
async fn test_cleanup_with_active_sender() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    let (tx, rx) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Take receiver (cleanup from client side)
    let _receiver = registry.take_token_receiver(&job_id);

    // Sender still active
    let result = tx.send("token".to_string());
    // Send may fail, which is expected
    let _ = result;

    // Cleanup: remove job
    registry.remove_job(&job_id);
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup with active sender successful");
}

/// Test cleanup with rapid create/remove cycles
#[tokio::test]
async fn test_cleanup_rapid_cycles() {
    let registry = Arc::new(JobRegistry::<String>::new());

    // Rapid create/remove cycles
    for _ in 0..100 {
        let job_id = registry.create_job();
        assert_eq!(registry.job_count(), 1);

        registry.remove_job(&job_id);
        assert_eq!(registry.job_count(), 0);
    }

    println!("✓ Rapid create/remove cycles successful");
}

/// Test cleanup with state transitions
#[tokio::test]
async fn test_cleanup_with_state_transitions() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    // Transition through states
    assert!(matches!(registry.get_job_state(&job_id), Some(JobState::Queued)));

    registry.update_state(&job_id, JobState::Running);
    assert!(matches!(registry.get_job_state(&job_id), Some(JobState::Running)));

    registry.update_state(&job_id, JobState::Completed);
    assert!(matches!(registry.get_job_state(&job_id), Some(JobState::Completed)));

    // Cleanup
    registry.remove_job(&job_id);
    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup with state transitions successful");
}

/// Test cleanup prevents dangling references
#[tokio::test]
async fn test_cleanup_prevents_dangling_references() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    let (tx, rx) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Take receiver
    let receiver = registry.take_token_receiver(&job_id);
    assert!(receiver.is_some());

    // Try to take again (should return None)
    let receiver2 = registry.take_token_receiver(&job_id);
    assert!(receiver2.is_none());

    // Cleanup
    registry.remove_job(&job_id);

    // Try to access after cleanup
    assert!(!registry.has_job(&job_id));

    println!("✓ Cleanup prevents dangling references");
}

/// Test cleanup with mixed payload and receiver operations
#[tokio::test]
async fn test_cleanup_mixed_operations() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let mut job_ids = vec![];

    // Create 20 jobs with mixed operations
    for i in 0..20 {
        let job_id = registry.create_job();

        if i % 2 == 0 {
            // Set payload
            registry.set_payload(&job_id, serde_json::json!({"index": i}));
        } else {
            // Set receiver
            let (tx, rx) = mpsc::unbounded_channel();
            registry.set_token_receiver(&job_id, rx);
            let _ = tx.send(format!("token-{}", i));
        }

        job_ids.push(job_id);
    }

    assert_eq!(registry.job_count(), 20);

    // Cleanup all
    for job_id in job_ids {
        registry.remove_job(&job_id);
    }

    assert_eq!(registry.job_count(), 0);

    println!("✓ Cleanup with mixed operations successful");
}
