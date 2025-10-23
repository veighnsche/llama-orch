//! Check daemon status
//!
//! TEAM-259: Extracted common status check pattern
//!
//! Provides generic daemon status checking functionality for:
//! - hive-lifecycle (check hive status)
//! - worker-lifecycle (check worker status)

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use serde::Serialize;
use std::time::Duration;

const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");

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
    let daemon_type = request.daemon_type.as_deref().unwrap_or("daemon");

    let mut narration =
        NARRATE.action("daemon_check").context(daemon_type).context(&request.health_url);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human(&format!("Checking {} status at {{}}", daemon_type)).emit();

    let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;

    let running = match client.get(&request.health_url).send().await {
        Ok(response) if response.status().is_success() => {
            let mut narration = NARRATE
                .action("daemon_check")
                .context(daemon_type)
                .context(&request.id)
                .context(&request.health_url);
            if let Some(jid) = job_id {
                narration = narration.job_id(jid);
            }
            narration.human(&format!("✅ {} '{{0}}' is running on {{1}}", daemon_type)).emit();
            true
        }
        Ok(response) => {
            let mut narration = NARRATE
                .action("daemon_check")
                .context(daemon_type)
                .context(&request.id)
                .context(response.status().to_string());
            if let Some(jid) = job_id {
                narration = narration.job_id(jid);
            }
            narration
                .human(&format!("⚠️  {} '{{0}}' responded with status: {{1}}", daemon_type))
                .emit();
            false
        }
        Err(_) => {
            let mut narration = NARRATE
                .action("daemon_check")
                .context(daemon_type)
                .context(&request.id)
                .context(&request.health_url);
            if let Some(jid) = job_id {
                narration = narration.job_id(jid);
            }
            narration.human(&format!("❌ {} '{{0}}' is not running on {{1}}", daemon_type)).emit();
            false
        }
    };

    Ok(StatusResponse { id: request.id, running, health_url: request.health_url })
}
