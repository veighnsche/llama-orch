// TEAM-211: Check if hive is running
// TEAM-220: Investigated - Health check with 5s timeout documented
// TEAM-259: Refactored to use daemon-lifecycle::check_daemon_status

use anyhow::Result;
use daemon_lifecycle::{check_daemon_status, StatusRequest};
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveStatusRequest, HiveStatusResponse};
use crate::validation::validate_hive_exists;

/// Check if hive is running
///
/// Performs HTTP health check to hive endpoint.
/// Timeout: 5 seconds.
///
/// # Arguments
/// * `request` - Status request with hive alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveStatusResponse)` - Health check result
/// * `Err` - Configuration error
pub async fn execute_hive_status(
    request: HiveStatusRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveStatusResponse> {
    let hive_config = validate_hive_exists(&config, &request.alias)?;

    let health_url = format!("http://{}:{}/health", hive_config.hostname, hive_config.hive_port);

    // Use daemon-lifecycle's generic check_daemon_status function
    let status_request = StatusRequest {
        id: request.alias.clone(),
        health_url: health_url.clone(),
        daemon_type: Some("hive".to_string()),
    };

    let status = check_daemon_status(status_request, Some(job_id)).await?;

    Ok(HiveStatusResponse {
        alias: request.alias,
        running: status.running,
        health_url,
    })
}
