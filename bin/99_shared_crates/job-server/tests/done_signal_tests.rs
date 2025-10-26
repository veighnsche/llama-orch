// TEAM-304: Tests for [DONE] and [ERROR] signals
//!
//! Verifies that job-server emits proper lifecycle signals:
//! - [DONE] when job completes successfully
//! - [ERROR] when job fails
//! - Signals sent exactly once per job

use job_server::{JobRegistry, execute_and_stream, JobState};
use std::sync::Arc;
use futures::StreamExt;

#[tokio::test]
async fn test_execute_and_stream_sends_done_on_success() {
    // TEAM-304: Verify [DONE] signal is sent when job completes successfully
    
    let registry = Arc::new(JobRegistry::new());
    let job_id = registry.create_job();
    
    // Set up payload
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    // Set up channel
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    // Send some tokens
    tx.send("token1".to_string()).unwrap();
    tx.send("token2".to_string()).unwrap();
    drop(tx);  // Close channel to trigger [DONE]
    
    // Stream should include [DONE]
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Ok(()) }  // Successful execution
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    assert_eq!(results.len(), 3, "Expected 3 items: 2 tokens + [DONE]");
    assert_eq!(results[0], "token1");
    assert_eq!(results[1], "token2");
    assert_eq!(results[2], "[DONE]");
}

#[tokio::test]
async fn test_execute_and_stream_sends_error_on_failure() {
    // TEAM-304: Verify [ERROR] signal is sent when job fails
    
    let registry = Arc::new(JobRegistry::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    // Stream should include [ERROR]
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { 
            // Fail immediately
            Err(anyhow::anyhow!("Test error"))
        }
    ).await;
    
    tx.send("token1".to_string()).unwrap();
    
    // Give executor time to fail and update state
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    drop(tx);  // Close channel after executor has failed
    
    let results: Vec<String> = stream.collect().await;
    
    assert!(results.len() >= 2, "Expected at least 2 items: token + [ERROR]");
    assert_eq!(results[0], "token1");
    assert!(results.last().unwrap().starts_with("[ERROR]"), "Last item should be [ERROR]");
    assert!(results.last().unwrap().contains("Test error"), "Error message should be included");
}

#[tokio::test]
async fn test_done_sent_only_once() {
    // TEAM-304: Verify [DONE] is sent exactly once
    
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    drop(tx);  // Close immediately
    
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Ok(()) }
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    // Should only have one [DONE]
    assert_eq!(results.len(), 1, "Expected exactly 1 item: [DONE]");
    assert_eq!(results[0], "[DONE]");
}

#[tokio::test]
async fn test_no_receiver_sends_done_immediately() {
    // TEAM-304: Verify [DONE] is sent even when no receiver is set
    
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    // Don't set a receiver - simulate missing channel
    
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Ok(()) }
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    // Should send [DONE] immediately
    assert_eq!(results.len(), 1, "Expected exactly 1 item: [DONE]");
    assert_eq!(results[0], "[DONE]");
}

#[tokio::test]
async fn test_job_state_updated_on_success() {
    // TEAM-304: Verify job state is updated to Completed on success
    
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    drop(tx);
    
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Ok(()) }
    ).await;
    
    // Consume stream
    let _results: Vec<String> = stream.collect().await;
    
    // Give executor time to update state
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Check job state
    let state = registry.get_job_state(&job_id);
    assert!(matches!(state, Some(JobState::Completed)), "Job state should be Completed");
}

#[tokio::test]
async fn test_job_state_updated_on_failure() {
    // TEAM-304: Verify job state is updated to Failed on error
    
    let registry = Arc::new(JobRegistry::<String>::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    drop(tx);
    
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Err(anyhow::anyhow!("Test failure")) }
    ).await;
    
    // Consume stream
    let _results: Vec<String> = stream.collect().await;
    
    // Give executor time to update state
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Check job state
    let state = registry.get_job_state(&job_id);
    match state {
        Some(JobState::Failed(err)) => {
            assert!(err.contains("Test failure"), "Error message should be preserved");
        }
        _ => panic!("Job state should be Failed"),
    }
}

#[tokio::test]
async fn test_multiple_tokens_then_done() {
    // TEAM-304: Verify [DONE] comes after all tokens
    
    let registry = Arc::new(JobRegistry::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    // Send multiple tokens
    for i in 0..10 {
        tx.send(format!("token{}", i)).unwrap();
    }
    drop(tx);
    
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Ok(()) }
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    assert_eq!(results.len(), 11, "Expected 10 tokens + [DONE]");
    
    // Verify all tokens come before [DONE]
    for i in 0..10 {
        assert_eq!(results[i], format!("token{}", i));
    }
    assert_eq!(results[10], "[DONE]");
}
