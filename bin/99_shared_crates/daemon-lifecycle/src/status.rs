//! Check daemon status
//!
//! TEAM-259: Extracted common status check pattern
//!
//! Provides generic daemon status checking functionality for:
//! - hive-lifecycle (check hive status)
//! - worker-lifecycle (check worker status)

use anyhow::Result;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use serde::Serialize;
use std::time::Duration;

/// Request to check daemon status
#[derive(Debug, Clone)]
pub struct StatusRequest {
    /// ID of the daemon instance (e.g., alias, worker ID)
    pub id: String,

    /// Health check URL (e.g., "http://localhost:8081/health")
    pub health_url: String,

    /// Optional: Daemon type name for narration (e.g., "hive", "worker")
    pub daemon_type: Option<String>,
}

/// Response from daemon status check
#[derive(Debug, Clone, Serialize)]
pub struct StatusResponse {
    /// ID of the daemon instance
    pub id: String,

    /// Whether the daemon is running
    pub running: bool,

    /// Health check URL that was checked
    pub health_url: String,
}

/// Check daemon status via HTTP health check
///
/// Performs HTTP GET to /health endpoint with 5-second timeout.
/// Returns whether the daemon is running and healthy.
///
/// # Arguments
/// * `request` - Status request with ID and health URL
/// * `job_id` - Optional job ID for narration routing
///
/// # Returns
/// * `Ok(StatusResponse)` - Status check result
/// * `Err` - HTTP client error
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{StatusRequest, check_daemon_status};
///
/// # async fn example() -> anyhow::Result<()> {
/// let request = StatusRequest {
///     id: "my-hive".to_string(),
///     health_url: "http://localhost:8081/health".to_string(),
///     daemon_type: Some("hive".to_string()),
/// };
///
/// let status = check_daemon_status(request, Some("job_123")).await?;
/// println!("Running: {}", status.running);
/// # Ok(())
/// # }
/// ```
pub async fn check_daemon_status(
    request: StatusRequest,
    job_id: Option<&str>,
) -> Result<StatusResponse> {
    // TEAM-311: Migrated to n!() macro
    let ctx = job_id.map(|jid| NarrationContext::new().with_job_id(jid));
    let daemon_type = request.daemon_type.as_deref().unwrap_or("daemon");

    let check_impl = async {
        n!("daemon_check", "Checking {} status at {}", daemon_type, request.health_url);

        let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;

        let running = match client.get(&request.health_url).send().await {
            Ok(response) if response.status().is_success() => {
                n!("daemon_check", "✅ {} '{}' is running on {}", daemon_type, request.id, request.health_url);
                true
            }
            Ok(response) => {
                n!("daemon_check", "⚠️  {} '{}' responded with status: {}", daemon_type, request.id, response.status());
                false
            }
            Err(_) => {
                n!("daemon_check", "❌ {} '{}' is not running on {}", daemon_type, request.id, request.health_url);
                false
            }
        };

        Ok(StatusResponse { id: request.id, running, health_url: request.health_url })
    };
    
    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, check_impl).await
    } else {
        check_impl.await
    }
}
