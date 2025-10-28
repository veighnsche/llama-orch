//! Unit tests for TimeoutEnforcer
//!
//! Created by: TEAM-163
//! Updated by: TEAM-330

use crate::TimeoutEnforcer;
use anyhow::Result;
use std::time::Duration;

#[tokio::test]
async fn test_successful_operation() {
    async fn quick_op() -> Result<String> {
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok("success".to_string())
    }

    let result = TimeoutEnforcer::new(Duration::from_secs(1))
        .with_label("Quick operation")
        .silent()
        .enforce(quick_op())
        .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
}

#[tokio::test]
async fn test_timeout_occurs() {
    async fn slow_op() -> Result<String> {
        tokio::time::sleep(Duration::from_secs(10)).await;
        Ok("should not reach here".to_string())
    }

    let result = TimeoutEnforcer::new(Duration::from_secs(1))
        .with_label("Slow operation")
        .silent()
        .enforce(slow_op())
        .await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("timed out"));
    assert!(err_msg.contains("1 second"));
}

#[tokio::test]
async fn test_operation_failure() {
    async fn failing_op() -> Result<String> {
        anyhow::bail!("operation failed")
    }

    let result = TimeoutEnforcer::new(Duration::from_secs(1)).silent().enforce(failing_op()).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("operation failed"));
}
