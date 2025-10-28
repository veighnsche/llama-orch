//! Tests for #[with_timeout] attribute macro
//!
//! Created by: TEAM-330
//!
//! Verifies that the macro correctly wraps async functions with timeout enforcement.

use anyhow::Result;
use std::time::Duration;
use timeout_enforcer::with_timeout;

// TEAM-330: Basic macro usage
#[with_timeout(secs = 1)]
async fn quick_operation() -> Result<String> {
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok("success".to_string())
}

// TEAM-330: With label
#[with_timeout(secs = 1, label = "Labeled operation")]
async fn labeled_operation() -> Result<String> {
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok("success".to_string())
}

// TEAM-330: With countdown (won't show in tests due to TTY detection)
#[with_timeout(secs = 1, label = "Countdown operation", countdown = true)]
async fn countdown_operation() -> Result<String> {
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok("success".to_string())
}

// TEAM-330: Operation that should timeout
#[with_timeout(secs = 1)]
async fn slow_operation() -> Result<String> {
    tokio::time::sleep(Duration::from_secs(10)).await;
    Ok("should not reach here".to_string())
}

// TEAM-330: With parameters
#[with_timeout(secs = 1, label = "Operation with params")]
async fn operation_with_params(x: i32, y: i32) -> Result<i32> {
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(x + y)
}

// TEAM-330: With mutable parameters
#[with_timeout(secs = 1)]
async fn operation_with_mut_params(mut x: i32) -> Result<i32> {
    x += 10;
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(x)
}

// TEAM-330: With reference parameters
#[with_timeout(secs = 1)]
async fn operation_with_ref_params(s: &str) -> Result<String> {
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(s.to_uppercase())
}

#[tokio::test]
async fn test_macro_basic() {
    let result = quick_operation().await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
}

#[tokio::test]
async fn test_macro_with_label() {
    let result = labeled_operation().await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
}

#[tokio::test]
async fn test_macro_with_countdown() {
    let result = countdown_operation().await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
}

#[tokio::test]
async fn test_macro_timeout_occurs() {
    let result = slow_operation().await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("timed out"));
}

#[tokio::test]
async fn test_macro_with_params() {
    let result = operation_with_params(5, 3).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 8);
}

#[tokio::test]
async fn test_macro_with_mut_params() {
    let result = operation_with_mut_params(5).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 15);
}

#[tokio::test]
async fn test_macro_with_ref_params() {
    let result = operation_with_ref_params("hello").await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "HELLO");
}

// TEAM-330: Test with context propagation
#[tokio::test]
async fn test_macro_with_context() {
    use observability_narration_core::{with_narration_context, NarrationContext};

    let ctx = NarrationContext::new().with_job_id("test-job-123");

    let result = with_narration_context(ctx, async {
        // Timeout narration automatically includes job_id!
        quick_operation().await
    })
    .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
}

// TEAM-330: Test multiple operations in sequence
#[tokio::test]
async fn test_macro_sequential_operations() {
    let result1 = quick_operation().await;
    assert!(result1.is_ok());

    let result2 = labeled_operation().await;
    assert!(result2.is_ok());

    let result3 = operation_with_params(10, 20).await;
    assert!(result3.is_ok());
    assert_eq!(result3.unwrap(), 30);
}
