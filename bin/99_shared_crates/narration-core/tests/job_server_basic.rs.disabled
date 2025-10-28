// TEAM-302: Job-server basic integration tests
//!
//! Tests for basic job creation and narration flow through SSE channels.
//!
//! # Coverage
//!
//! - Job creation with narration
//! - Job isolation (no cross-contamination)
//! - SSE channel cleanup
//!
//! # Scale
//!
//! Tests are designed for NUC-friendly execution (small scale, fast).

use observability_narration_core::{n, with_narration_context, NarrationContext};

// TEAM-302: Import test harness
mod harness;
use harness::NarrationTestHarness;

#[tokio::test]
async fn test_job_creation_with_narration() {
    // TEAM-302: Test basic job creation and narration flow
    
    // Start test harness
    let harness = NarrationTestHarness::start().await;
    
    // Create job
    let operation = serde_json::json!({"operation": "test"});
    let job_id = harness.submit_job(operation).await;
    
    assert!(!job_id.is_empty());
    assert!(job_id.starts_with("job-"));
    
    // Get SSE stream
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Emit test narration (simulating service)
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("test_action", "Test message from service");
    }).await;
    
    // Verify received via SSE
    stream.assert_next("test_action", "Test message").await;
}

#[tokio::test]
async fn test_job_narration_isolation() {
    // TEAM-302: Test that jobs have isolated SSE channels
    
    let harness = NarrationTestHarness::start().await;
    
    // Create two jobs
    let job1_id = harness.submit_job(
        serde_json::json!({"operation": "test1"})
    ).await;
    
    let job2_id = harness.submit_job(
        serde_json::json!({"operation": "test2"})
    ).await;
    
    // Emit to job 1
    let ctx1 = NarrationContext::new().with_job_id(&job1_id);
    with_narration_context(ctx1, async {
        n!("job1", "Message for job 1");
    }).await;
    
    // Emit to job 2
    let ctx2 = NarrationContext::new().with_job_id(&job2_id);
    with_narration_context(ctx2, async {
        n!("job2", "Message for job 2");
    }).await;
    
    // Verify isolation - each job only receives its own message
    let mut stream1 = harness.get_sse_stream(&job1_id);
    stream1.assert_next("job1", "Message for job 1").await;
    
    let mut stream2 = harness.get_sse_stream(&job2_id);
    stream2.assert_next("job2", "Message for job 2").await;
}

#[tokio::test]
async fn test_sse_channel_cleanup() {
    // TEAM-302: Test that SSE channels work correctly after receiver drop
    
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::json!({"operation": "test"})
    ).await;
    
    {
        // Create stream in limited scope
        let mut stream = harness.get_sse_stream(&job_id);
        
        let ctx = NarrationContext::new().with_job_id(&job_id);
        with_narration_context(ctx, async {
            n!("test", "Before drop");
        }).await;
        
        stream.assert_next("test", "Before drop").await;
        
        // Stream dropped here
    }
    
    // Note: After receiver is taken once, it cannot be taken again
    // This test verifies that the channel was properly consumed
    
    // Verify channel was taken (receiver should be None now)
    let receiver_opt = observability_narration_core::output::sse_sink::take_job_receiver(&job_id);
    assert!(receiver_opt.is_none(), "Receiver should have been consumed");
}

#[tokio::test]
async fn test_multiple_events_same_job() {
    // TEAM-302: Test that multiple narration events flow through same channel
    
    let harness = NarrationTestHarness::start().await;
    let job_id = harness.submit_job(serde_json::json!({"operation": "test"})).await;
    
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Emit multiple events
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("step1", "First step");
        n!("step2", "Second step");
        n!("step3", "Third step");
    }).await;
    
    // Verify all events received in order
    stream.assert_next("step1", "First step").await;
    stream.assert_next("step2", "Second step").await;
    stream.assert_next("step3", "Third step").await;
}

#[tokio::test]
async fn test_narration_without_job_id_dropped() {
    // TEAM-302: Test that narration without job_id is dropped (security)
    
    let harness = NarrationTestHarness::start().await;
    let job_id = harness.submit_job(serde_json::json!({"operation": "test"})).await;
    
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Emit narration WITHOUT job_id (should be dropped)
    n!("no_job", "This should not appear in SSE");
    
    // Emit narration WITH job_id (should appear)
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("with_job", "This should appear in SSE");
    }).await;
    
    // Verify only the event with job_id was received
    stream.assert_next("with_job", "This should appear").await;
    
    // Verify no more events
    stream.assert_no_more_events().await;
}
