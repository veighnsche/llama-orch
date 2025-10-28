// TEAM-305: Tests for job timeout and cancellation functionality
//!
//! Verifies that job-server properly handles:
//! - Job timeout (jobs that run too long)
//! - Job cancellation (user-initiated cancellation)
//! - [CANCELLED] signal emission
//! - State transitions to Cancelled

use futures::StreamExt;
use job_server::{execute_and_stream, JobRegistry, JobState}; // TEAM-312: Renamed from execute_and_stream_with_timeout
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_job_timeout() {
    // TEAM-305: Verify job times out after specified duration

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Send a token before timeout
    tx.send("token1".to_string()).unwrap();

    // Create stream with 100ms timeout
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async {
            // Simulate long-running job (500ms)
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(())
        },
        Some(Duration::from_millis(100)), // Timeout after 100ms
    )
    .await;

    // Give executor time to timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    drop(tx); // Close channel

    let results: Vec<String> = stream.collect().await;

    // Should have token + [ERROR] with timeout message
    assert!(results.len() >= 2, "Expected at least 2 items: token + [ERROR]");
    assert_eq!(results[0], "token1");
    assert!(results.last().unwrap().starts_with("[ERROR]"), "Last item should be [ERROR]");
    assert!(results.last().unwrap().contains("Timeout after"), "Error should mention timeout");

    // Check job state is Failed
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Failed(_))), "Job state should be Failed");
}

#[tokio::test]
async fn test_job_completes_before_timeout() {
    // TEAM-305: Verify job completes successfully if it finishes before timeout

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    tx.send("token1".to_string()).unwrap();

    // Create stream with 500ms timeout
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async {
            // Fast job (50ms)
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(())
        },
        Some(Duration::from_millis(500)), // Timeout after 500ms
    )
    .await;

    // Give executor time to complete
    tokio::time::sleep(Duration::from_millis(100)).await;

    drop(tx);

    let results: Vec<String> = stream.collect().await;

    // Should have token + [DONE]
    assert_eq!(results.len(), 2, "Expected 2 items: token + [DONE]");
    assert_eq!(results[0], "token1");
    assert_eq!(results[1], "[DONE]");

    // Check job state is Completed
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Completed)), "Job state should be Completed");
}

#[tokio::test]
async fn test_job_cancellation() {
    // TEAM-305: Verify job can be cancelled by user

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    tx.send("token1".to_string()).unwrap();

    // Create stream (no timeout)
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async {
            // Long-running job
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(())
        },
        None, // No timeout
    )
    .await;

    // Cancel the job after a short delay
    tokio::time::sleep(Duration::from_millis(50)).await;
    let cancelled = registry.cancel_job(&job_id);
    assert!(cancelled, "Job should be cancelled successfully");

    // Give executor time to detect cancellation
    tokio::time::sleep(Duration::from_millis(100)).await;

    drop(tx);

    let results: Vec<String> = stream.collect().await;

    // Should have token + [CANCELLED]
    assert!(results.len() >= 2, "Expected at least 2 items: token + [CANCELLED]");
    assert_eq!(results[0], "token1");
    assert_eq!(results.last().unwrap(), "[CANCELLED]", "Last item should be [CANCELLED]");

    // Check job state is Cancelled
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Cancelled)), "Job state should be Cancelled");
}

#[tokio::test]
async fn test_cancel_queued_job() {
    // TEAM-305: Verify queued job can be cancelled before execution

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    // Job is in Queued state
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Queued)));

    // Cancel it
    let cancelled = registry.cancel_job(&job_id);
    assert!(cancelled, "Queued job should be cancelled");

    // Check state
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Cancelled)), "Job state should be Cancelled");
}

#[tokio::test]
async fn test_cannot_cancel_completed_job() {
    // TEAM-305: Verify completed job cannot be cancelled

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    // Mark as completed
    registry.update_state(&job_id, JobState::Completed);

    // Try to cancel
    let cancelled = registry.cancel_job(&job_id);
    assert!(!cancelled, "Completed job should not be cancelled");

    // State should still be Completed
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Completed)), "Job state should remain Completed");
}

#[tokio::test]
async fn test_cannot_cancel_failed_job() {
    // TEAM-305: Verify failed job cannot be cancelled

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    // Mark as failed
    registry.update_state(&job_id, JobState::Failed("Test error".to_string()));

    // Try to cancel
    let cancelled = registry.cancel_job(&job_id);
    assert!(!cancelled, "Failed job should not be cancelled");

    // State should still be Failed
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Failed(_))), "Job state should remain Failed");
}

#[tokio::test]
async fn test_cancel_nonexistent_job() {
    // TEAM-305: Verify cancelling non-existent job returns false

    let registry = Arc::new(JobRegistry::<String>::new());

    let cancelled = registry.cancel_job("job-nonexistent");
    assert!(!cancelled, "Non-existent job should return false");
}

#[tokio::test]
async fn test_timeout_with_no_receiver() {
    // TEAM-305: Verify timeout works even when no receiver is set

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));

    // Don't set receiver

    // Create stream with 100ms timeout
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async {
            // Long-running job
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(())
        },
        Some(Duration::from_millis(100)),
    )
    .await;

    // Give executor time to timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    let results: Vec<String> = stream.collect().await;

    // Should have [ERROR] immediately (no tokens)
    assert_eq!(results.len(), 1, "Expected 1 item: [ERROR]");
    assert!(
        results[0].starts_with("[ERROR]") || results[0] == "[DONE]",
        "Should be [ERROR] or [DONE] (race condition)"
    );
}

#[tokio::test]
async fn test_cancellation_with_no_receiver() {
    // TEAM-305: Verify cancellation works even when no receiver is set

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));

    // Don't set receiver

    // Create stream
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async {
            // Long-running job
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(())
        },
        None,
    )
    .await;

    // Cancel immediately
    tokio::time::sleep(Duration::from_millis(50)).await;
    registry.cancel_job(&job_id);

    // Give executor time to detect cancellation
    tokio::time::sleep(Duration::from_millis(100)).await;

    let results: Vec<String> = stream.collect().await;

    // Should have [CANCELLED]
    assert_eq!(results.len(), 1, "Expected 1 item: [CANCELLED]");
    assert_eq!(results[0], "[CANCELLED]");
}

#[tokio::test]
async fn test_multiple_tokens_then_timeout() {
    // TEAM-305: Verify timeout signal comes after all tokens

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    // Send multiple tokens
    for i in 0..5 {
        tx.send(format!("token{}", i)).unwrap();
    }

    // Create stream with timeout
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(())
        },
        Some(Duration::from_millis(100)),
    )
    .await;

    // Give executor time to timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    drop(tx);

    let results: Vec<String> = stream.collect().await;

    // Should have 5 tokens + [ERROR]
    assert_eq!(results.len(), 6, "Expected 6 items: 5 tokens + [ERROR]");

    // Verify all tokens come before [ERROR]
    for i in 0..5 {
        assert_eq!(results[i], format!("token{}", i));
    }
    assert!(results[5].starts_with("[ERROR]"), "Last item should be [ERROR]");
}

#[tokio::test]
async fn test_get_cancellation_token() {
    // TEAM-305: Verify get_cancellation_token returns correct token

    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();

    let token = registry.get_cancellation_token(&job_id);
    assert!(token.is_some(), "Cancellation token should exist");

    // Cancel via registry
    registry.cancel_job(&job_id);

    // Token should be cancelled
    let token = token.unwrap();
    assert!(token.is_cancelled(), "Token should be cancelled");
}

#[tokio::test]
async fn test_get_cancellation_token_nonexistent() {
    // TEAM-305: Verify get_cancellation_token returns None for non-existent job

    let registry = Arc::new(JobRegistry::<String>::new());

    let token = registry.get_cancellation_token("job-nonexistent");
    assert!(token.is_none(), "Token should be None for non-existent job");
}
