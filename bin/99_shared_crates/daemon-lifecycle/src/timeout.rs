//! Timeout enforcement for daemon operations
//!
//! TEAM-276: Added timeout wrapper using TimeoutEnforcer

use anyhow::Result;
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;

/// Configuration for timeout enforcement
///
/// TEAM-276: Wrapper around TimeoutEnforcer for daemon operations
pub struct TimeoutConfig {
    /// Operation name for narration
    pub operation_name: String,

    /// Timeout duration
    pub timeout: Duration,

    /// Optional job_id for narration routing
    pub job_id: Option<String>,
}

impl TimeoutConfig {
    /// Create a new timeout config
    pub fn new(operation_name: impl Into<String>, timeout: Duration) -> Self {
        Self {
            operation_name: operation_name.into(),
            timeout,
            job_id: None,
        }
    }

    /// Set the job_id for narration routing
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set timeout in seconds (convenience method)
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout = Duration::from_secs(secs);
        self
    }
}

/// Wrap an async operation with timeout enforcement
///
/// TEAM-276: Uses TimeoutEnforcer for consistent timeout behavior with narration
///
/// # Arguments
/// * `config` - Timeout configuration
/// * `operation` - Async operation to execute with timeout
///
/// # Returns
/// * `Ok(T)` - Operation completed successfully
/// * `Err` - Operation timed out or failed
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{with_timeout, TimeoutConfig};
/// use std::time::Duration;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = TimeoutConfig::new("fetch_data", Duration::from_secs(30))
///     .with_job_id("job-123");
///
/// let result = with_timeout(config, async {
///     // Your operation here
///     Ok(42)
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub async fn with_timeout<F, T>(config: TimeoutConfig, operation: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let mut enforcer = TimeoutEnforcer::new(config.timeout)
        .with_label(&config.operation_name);

    if let Some(ref job_id) = config.job_id {
        enforcer = enforcer.with_job_id(job_id);
    }

    enforcer.enforce(operation).await
}

/// Convenience function for simple timeout without config
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::timeout_after;
/// use std::time::Duration;
///
/// # async fn example() -> anyhow::Result<()> {
/// let result = timeout_after(Duration::from_secs(5), async {
///     // Your operation here
///     Ok(42)
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub async fn timeout_after<F, T>(timeout: Duration, operation: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    with_timeout(
        TimeoutConfig::new("operation", timeout),
        operation,
    )
    .await
}
