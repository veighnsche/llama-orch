//! Check daemon status
//!
//! TEAM-259: Extracted common status check pattern
//! TEAM-316: Use types from daemon-contract
//!
//! Provides generic daemon status checking functionality for:
//! - hive-lifecycle (check hive status)
//! - worker-lifecycle (check worker status)

use anyhow::Result;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use std::time::Duration;

// TEAM-316: Use status types from daemon-contract
pub use daemon_contract::{StatusRequest, StatusResponse};

/// Check daemon status via HTTP health check
///
/// TEAM-316: Updated to use daemon-contract types with health_url and daemon_type as parameters
///
/// Performs HTTP GET to /health endpoint with 5-second timeout.
/// Returns whether the daemon is running and healthy.
///
/// # Arguments
/// * `id` - ID of the daemon instance (e.g., alias, worker ID)
/// * `health_url` - Health check URL (e.g., "http://localhost:8081/health")
/// * `daemon_type` - Optional daemon type for narration (e.g., "hive", "worker")
/// * `job_id` - Optional job ID for narration routing
///
/// # Returns
/// * `Ok(StatusResponse)` - Status check result
/// * `Err` - HTTP client error
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::check_daemon_status;
///
/// # async fn example() -> anyhow::Result<()> {
/// let status = check_daemon_status(
///     "my-hive",
///     "http://localhost:8081/health",
///     Some("hive"),
///     Some("job_123"),
/// ).await?;
/// println!("Running: {}", status.is_running);
/// # Ok(())
/// # }
/// ```
pub async fn check_daemon_status(
    id: &str,
    health_url: &str,
    daemon_type: Option<&str>,
    job_id: Option<&str>,
) -> Result<StatusResponse> {
    // TEAM-311: Migrated to n!() macro
    // TEAM-316: Updated to use daemon-contract types
    let ctx = job_id.map(|jid| NarrationContext::new().with_job_id(jid));
    let daemon_type = daemon_type.unwrap_or("daemon");
    let id = id.to_string();
    let health_url = health_url.to_string();

    let check_impl = async {
        n!("daemon_check", "Checking {} status at {}", daemon_type, health_url);

        let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;

        let running = match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                n!("daemon_check", "✅ {} '{}' is running on {}", daemon_type, id, health_url);
                true
            }
            Ok(response) => {
                n!("daemon_check", "⚠️  {} '{}' responded with status: {}", daemon_type, id, response.status());
                false
            }
            Err(_) => {
                n!("daemon_check", "❌ {} '{}' is not running on {}", daemon_type, id, health_url);
                false
            }
        };

        Ok(StatusResponse {
            id,
            is_running: running,
            health_status: Some(health_url),
            metadata: None,
        })
    };
    
    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, check_impl).await
    } else {
        check_impl.await
    }
}
