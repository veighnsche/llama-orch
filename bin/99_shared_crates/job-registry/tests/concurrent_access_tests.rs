// TEAM-243: Concurrent access tests for job-registry
// Purpose: Verify thread-safe concurrent operations on job registry
// Scale: Reasonable for NUC (5-10 concurrent, 100 jobs total)
// Historical Context: TEAM-243 implemented Priority 1 critical tests for job lifecycle

use job_registry::{JobRegistry, JobState};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task;

/// Test concurrent job creation (10 concurrent)
#[tokio::test]
async fn test_concurrent_job_creation() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let mut handles = vec![];

    // Create 10 concurrent job creation tasks
    for _ in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = task::spawn(async move {
            registry_clone.create_job()
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let job_ids: Vec<String> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Verify all jobs were created
    assert_eq!(job_ids.len(), 10);
    assert_eq!(registry.job_count(), 10);

    // Verify all job IDs are unique
    let unique_ids: std::collections::HashSet<_> = job_ids.into_iter().collect();
    assert_eq!(unique_ids.len(), 10);

    println!("✓ 10 concurrent job creations completed successfully");
}

/// Test concurrent state updates on same job
#[tokio::test]
async fn test_concurrent_state_updates_same_job() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    let mut handles = vec![];

    // Create 5 concurrent state update tasks
    for i in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let job_id_clone = job_id.clone();
        let handle = task::spawn(async move {
            if i % 2 == 0 {
                registry_clone.update_state(&job_id_clone, JobState::Running);
            } else {
                registry_clone.update_state(&job_id_clone, JobState::Completed);
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    futures::future::join_all(handles).await;

    // Verify job still exists and has a valid state
    assert!(registry.has_job(&job_id));
    let state = registry.get_job_state(&job_id).unwrap();
    assert!(matches!(state, JobState::Running | JobState::Completed));

    println!("✓ 5 concurrent state updates on same job completed successfully");
}

/// Test concurrent state updates on different jobs
#[tokio::test]
async fn test_concurrent_state_updates_different_jobs() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let mut job_ids = vec![];

    // Create 10 jobs
    for _ in 0..10 {
        job_ids.push(registry.create_job());
    }

    let mut handles = vec![];

    // Update state on each job concurrently
    for job_id in job_ids.clone() {
        let registry_clone = Arc::clone(&registry);
        let handle = task::spawn(async move {
            registry_clone.update_state(&job_id, JobState::Running);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    futures::future::join_all(handles).await;

    // Verify all jobs have Running state
    for job_id in job_ids {
        let state = registry.get_job_state(&job_id).unwrap();
        assert!(matches!(state, JobState::Running));
    }

    println!("✓ 10 concurrent state updates on different jobs completed successfully");
}

/// Test concurrent reads during writes
#[tokio::test]
async fn test_concurrent_reads_during_writes() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    let mut read_handles = vec![];
    let mut write_handles = vec![];

    // Spawn 5 readers
    for _ in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let job_id_clone = job_id.clone();
        let handle = task::spawn(async move {
            registry_clone.get_job_state(&job_id_clone)
        });
        read_handles.push(handle);
    }

    // Spawn 5 writers
    for i in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let job_id_clone = job_id.clone();
        let handle = task::spawn(async move {
            if i % 2 == 0 {
                registry_clone.update_state(&job_id_clone, JobState::Running);
            } else {
                registry_clone.update_state(&job_id_clone, JobState::Completed);
            }
        });
        write_handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in read_handles {
        handle.await.unwrap();
    }
    for handle in write_handles {
        handle.await.unwrap();
    }

    println!("✓ 5 concurrent reads and 5 concurrent writes completed successfully");
}

/// Test concurrent token receiver operations
#[tokio::test]
async fn test_concurrent_token_receiver_operations() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    let mut handles = vec![];

    // Create and set token receiver
    let (tx, rx) = mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Spawn 5 tasks that send tokens
    for i in 0..5 {
        let tx_clone = tx.clone();
        let handle = task::spawn(async move {
            tx_clone.send(format!("token-{}", i)).unwrap();
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    futures::future::join_all(handles).await;

    // Take receiver and verify tokens
    let mut receiver = registry.take_token_receiver(&job_id).unwrap();
    let mut tokens = vec![];
    while let Ok(Some(token)) = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        receiver.recv()
    ).await {
        tokens.push(token);
    }

    assert_eq!(tokens.len(), 5);
    println!("✓ 5 concurrent token sends completed successfully");
}

/// Test concurrent payload operations
#[tokio::test]
async fn test_concurrent_payload_operations() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let mut job_ids = vec![];

    // Create 10 jobs
    for _ in 0..10 {
        job_ids.push(registry.create_job());
    }

    let mut handles = vec![];

    // Set payloads concurrently
    for (i, job_id) in job_ids.clone().into_iter().enumerate() {
        let registry_clone = Arc::clone(&registry);
        let handle = task::spawn(async move {
            let payload = serde_json::json!({
                "index": i,
                "data": format!("payload-{}", i)
            });
            registry_clone.set_payload(&job_id, payload);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    futures::future::join_all(handles).await;

    // Take payloads concurrently
    let mut handles = vec![];
    for job_id in job_ids {
        let registry_clone = Arc::clone(&registry);
        let handle = task::spawn(async move {
            registry_clone.take_payload(&job_id)
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let payloads: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Verify all payloads were retrieved
    assert_eq!(payloads.len(), 10);
    assert!(payloads.iter().all(|p| p.is_some()));

    println!("✓ 10 concurrent payload set/take operations completed successfully");
}

/// Test concurrent job removal
#[tokio::test]
async fn test_concurrent_job_removal() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let mut job_ids = vec![];

    // Create 10 jobs
    for _ in 0..10 {
        job_ids.push(registry.create_job());
    }

    assert_eq!(registry.job_count(), 10);

    let mut handles = vec![];

    // Remove jobs concurrently
    for job_id in job_ids {
        let registry_clone = Arc::clone(&registry);
        let handle = task::spawn(async move {
            registry_clone.remove_job(&job_id)
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    futures::future::join_all(handles).await;

    // Verify all jobs were removed
    assert_eq!(registry.job_count(), 0);

    println!("✓ 10 concurrent job removals completed successfully");
}

/// Test memory efficiency with 100 jobs
#[tokio::test]
async fn test_memory_efficiency_100_jobs() {
    let registry = Arc::new(JobRegistry::<String>::new());

    // Create 100 jobs
    for _ in 0..100 {
        registry.create_job();
    }

    assert_eq!(registry.job_count(), 100);

    // Update state on all jobs
    let job_ids = registry.job_ids();
    for job_id in job_ids {
        registry.update_state(&job_id, JobState::Running);
    }

    // Remove all jobs
    let job_ids = registry.job_ids();
    for job_id in job_ids {
        registry.remove_job(&job_id);
    }

    // Verify memory was freed
    assert_eq!(registry.job_count(), 0);

    println!("✓ 100 jobs created, updated, and removed successfully");
}

/// Test job_ids() with concurrent modifications
#[tokio::test]
async fn test_job_ids_with_concurrent_modifications() {
    let registry = Arc::new(JobRegistry::<String>::new());

    // Create initial jobs
    for _ in 0..5 {
        registry.create_job();
    }

    let mut handles = vec![];

    // Spawn tasks that create and remove jobs
    for _ in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let handle = task::spawn(async move {
            let job_id = registry_clone.create_job();
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            registry_clone.remove_job(&job_id);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    futures::future::join_all(handles).await;

    // Verify job_ids() returns consistent results
    let ids = registry.job_ids();
    assert_eq!(ids.len(), registry.job_count());

    println!("✓ job_ids() returned consistent results with concurrent modifications");
}

/// Test has_job() with concurrent operations
#[tokio::test]
async fn test_has_job_with_concurrent_operations() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    let mut handles = vec![];

    // Spawn tasks that check job existence
    for _ in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let job_id_clone = job_id.clone();
        let handle = task::spawn(async move {
            registry_clone.has_job(&job_id_clone)
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // All checks should return true
    assert!(results.iter().all(|&r| r));

    println!("✓ 10 concurrent has_job() checks all returned true");
}

/// Test concurrent access with mixed operations
#[tokio::test]
async fn test_concurrent_mixed_operations() {
    let registry = Arc::new(JobRegistry::<String>::new());
    let mut handles = vec![];

    // Create 10 jobs
    let job_ids: Vec<_> = (0..10)
        .map(|_| registry.create_job())
        .collect();

    // Spawn mixed operations
    for (i, job_id) in job_ids.iter().enumerate() {
        let registry_clone = Arc::clone(&registry);
        let job_id_clone = job_id.clone();
        
        let handle = task::spawn(async move {
            match i % 4 {
                0 => {
                    // Update state
                    registry_clone.update_state(&job_id_clone, JobState::Running);
                }
                1 => {
                    // Get state
                    registry_clone.get_job_state(&job_id_clone);
                }
                2 => {
                    // Set payload
                    registry_clone.set_payload(&job_id_clone, serde_json::json!({"data": i}));
                }
                _ => {
                    // Check existence
                    registry_clone.has_job(&job_id_clone);
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    futures::future::join_all(handles).await;

    // Verify registry is still consistent
    assert_eq!(registry.job_count(), 10);

    println!("✓ 10 concurrent mixed operations completed successfully");
}
