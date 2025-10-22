// TEAM-211: Get details for a single hive
// TEAM-220: Investigated - Single hive query documented

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveGetRequest, HiveGetResponse, HiveInfo};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Get details for a single hive
///
/// Returns hive configuration from hives.conf.
/// Validates that hive exists.
///
/// # Arguments
/// * `request` - Get request with hive alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveGetResponse)` - Hive details
/// * `Err` - Hive not found or configuration error
pub async fn execute_hive_get(
    request: HiveGetRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveGetResponse> {
    let hive_config = validate_hive_exists(&config, &request.alias)?;

    NARRATE
        .action("hive_get")
        .job_id(job_id)
        .context(&request.alias)
        .human("Hive '{}' details:")
        .emit();

    // Log details (matches original behavior)
    println!("Alias: {}", request.alias);
    println!("Host: {}", hive_config.hostname);
    println!("Port: {}", hive_config.hive_port);
    if let Some(ref bp) = hive_config.binary_path {
        println!("Binary: {}", bp);
    }

    let hive = HiveInfo {
        alias: request.alias.clone(),
        hostname: hive_config.hostname.clone(),
        hive_port: hive_config.hive_port,
        binary_path: hive_config.binary_path.clone(),
    };

    Ok(HiveGetResponse { hive })
}
