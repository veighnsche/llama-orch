// TEAM-211: Check if hive is running
// TEAM-220: Investigated - Health check with 5s timeout documented

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;
use tokio::time::Duration;

use crate::types::{HiveStatusRequest, HiveStatusResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

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

    NARRATE
        .action("hive_check")
        .job_id(job_id)
        .context(&health_url)
        .human("Checking hive status at {}")
        .emit();

    let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;

    let running = match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE
                .action("hive_check")
                .job_id(job_id)
                .context(&request.alias)
                .context(&health_url)
                .human("✅ Hive '{0}' is running on {1}")
                .emit();
            true
        }
        Ok(response) => {
            NARRATE
                .action("hive_check")
                .job_id(job_id)
                .context(&request.alias)
                .context(response.status().to_string())
                .human("⚠️  Hive '{0}' responded with status: {1}")
                .emit();
            false
        }
        Err(_) => {
            NARRATE
                .action("hive_check")
                .job_id(job_id)
                .context(&request.alias)
                .context(&health_url)
                .human("❌ Hive '{0}' is not running on {1}")
                .emit();
            false
        }
    };

    Ok(HiveStatusResponse { alias: request.alias, running, health_url })
}
