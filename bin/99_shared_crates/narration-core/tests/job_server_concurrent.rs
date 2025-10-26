// TEAM-302: Concurrent job testing
//!
//! Tests for concurrent job execution and narration isolation.
//!
//! # Coverage
//!
//! - Concurrent job creation (10 jobs)
//! - High-frequency narration (100 events)
//! - Nested task context propagation
//!
//! # Scale
//!
//! Tests use NUC-friendly limits (10 jobs, 100 events) for fast execution.

use observability_narration_core::{n, with_narration_context, NarrationContext};

// TEAM-302: Import test harness
mod harness;
use harness::NarrationTestHarness;

#[tokio::test]
async fn test_10_concurrent_jobs() {
    // TEAM-302: Test concurrent job creation and isolation
    
    let harness = NarrationTestHarness::start().await;
    
    // Create 10 jobs
    let mut job_ids = Vec::new();
    for _ in 0..10 {
        let op = serde_json::json!({"operation": "test"});
        let job_id = harness.submit_job(op).await;
        job_ids.push(job_id);
    }
    
    // Emit narration to each job concurrently
    let mut handles = Vec::new();
    for (i, job_id) in job_ids.iter().enumerate() {
        let job_id = job_id.clone();
        let handle = tokio::spawn(async move {
            let ctx = NarrationContext::new().with_job_id(&job_id);
            let msg = format!("Job {} message", i);
            
            with_narration_context(ctx, async move {
                n!("job_test", "{}", msg);
            }).await;
        });
        handles.push(handle);
    }
    
    // Wait for all emissions
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify each job received only its own message
    for (i, job_id) in job_ids.iter().enumerate() {
        let mut stream = harness.get_sse_stream(job_id);
        let event = stream.next_event().await
            .unwrap_or_else(|| panic!("Job {} didn't receive event", i));
        
        assert!(
            event.human.contains(&format!("Job {} message", i)),
            "Job {} received wrong message: {}",
            i,
            event.human
        );
        
        // Verify no cross-contamination
        stream.assert_no_more_events().await;
    }
}

#[tokio::test]
async fn test_high_frequency_narration() {
    // TEAM-302: Test that channels handle high-frequency events
    
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::json!({"operation": "test"})
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    // Emit 100 events rapidly
    with_narration_context(ctx, async {
        for i in 0..100 {
            n!("rapid_test", "Event {}", i);
        }
    }).await;
    
    // Verify all received
    let mut stream = harness.get_sse_stream(&job_id);
    let mut count = 0;
    
    while let Some(event) = stream.next_event().await {
        assert_eq!(event.action, "rapid_test");
        assert!(event.human.starts_with("Event "));
        count += 1;
        
        if count >= 100 {
            break;
        }
    }
    
    assert_eq!(count, 100, "Not all events received");
}

#[tokio::test]
async fn test_job_context_in_nested_tasks() {
    // TEAM-302: Test that context propagates to nested tasks
    // NOTE: tokio::spawn does NOT inherit task-local context
    // This test verifies that we can manually pass job_id to spawned tasks
    
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::json!({"operation": "test"})
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        // Outer task
        n!("outer", "Outer task");
    }).await;
    
    // Nested task needs explicit context (tokio::spawn doesn't inherit task-local)
    let job_id_clone = job_id.clone();
    tokio::spawn(async move {
        let ctx = NarrationContext::new().with_job_id(&job_id_clone);
        with_narration_context(ctx, async {
            n!("inner", "Inner task");
        }).await;
    }).await.unwrap();
    
    // Verify both events received
    let mut stream = harness.get_sse_stream(&job_id);
    stream.assert_next("outer", "Outer task").await;
    stream.assert_next("inner", "Inner task").await;
}

#[tokio::test]
async fn test_concurrent_narration_same_job() {
    // TEAM-302: Test multiple tasks emitting to same job concurrently
    // NOTE: Each spawned task needs explicit context
    
    let harness = NarrationTestHarness::start().await;
    let job_id = harness.submit_job(serde_json::json!({"operation": "test"})).await;
    
    // Spawn 5 concurrent tasks all emitting to same job
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let job_id_clone = job_id.clone();
        let handle = tokio::spawn(async move {
            let ctx = NarrationContext::new().with_job_id(&job_id_clone);
            with_narration_context(ctx, async move {
                n!("concurrent", "Task {} emitting", i);
            }).await;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all 5 events received (order may vary)
    let mut stream = harness.get_sse_stream(&job_id);
    let mut count = 0;
    
    while let Some(event) = stream.next_event().await {
        assert_eq!(event.action, "concurrent");
        assert!(event.human.contains("Task"));
        assert!(event.human.contains("emitting"));
        count += 1;
        
        if count >= 5 {
            break;
        }
    }
    
    assert_eq!(count, 5, "Not all concurrent events received");
}

#[tokio::test]
async fn test_job_registry_concurrent_access() {
    // TEAM-302: Test that job registry handles concurrent job creation
    
    let harness = NarrationTestHarness::start().await;
    
    // Create 10 jobs concurrently
    let mut handles = Vec::new();
    for i in 0..10 {
        let harness_clone = harness.registry().clone();
        let handle = tokio::spawn(async move {
            let job_id = harness_clone.create_job();
            let payload = serde_json::json!({"operation": format!("test{}", i)});
            harness_clone.set_payload(&job_id, payload);
            job_id
        });
        handles.push(handle);
    }
    
    // Collect all job IDs
    let mut job_ids = Vec::new();
    for handle in handles {
        let job_id = handle.await.unwrap();
        job_ids.push(job_id);
    }
    
    // Verify all jobs created successfully
    assert_eq!(job_ids.len(), 10);
    
    // Verify all job IDs are unique
    let mut unique_ids = job_ids.clone();
    unique_ids.sort();
    unique_ids.dedup();
    assert_eq!(unique_ids.len(), 10, "Job IDs are not unique");
    
    // Verify all jobs exist in registry
    for job_id in &job_ids {
        assert!(harness.registry().has_job(job_id), "Job {} not found in registry", job_id);
    }
}
