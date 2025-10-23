// TEAM-211: Get details for a single hive
// TEAM-220: Investigated - Single hive query documented
// TEAM-259: Refactored to use daemon-lifecycle::get_daemon

use anyhow::Result;
use daemon_lifecycle::{get_daemon, GettableConfig};
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveGetRequest, HiveGetResponse, HiveInfo};

// Wrapper struct to implement GettableConfig (avoids orphan rule)
struct HiveConfigWrapper<'a>(&'a RbeeConfig);

impl<'a> GettableConfig for HiveConfigWrapper<'a> {
    type Info = HiveInfo;

    fn get_by_id(&self, id: &str) -> Option<Self::Info> {
        self.0.hives.hives.iter().find(|h| h.alias == id).map(|h| HiveInfo {
            alias: h.alias.clone(),
            hostname: h.hostname.clone(),
            hive_port: h.hive_port,
            binary_path: h.binary_path.clone(),
        })
    }

    fn daemon_type_name(&self) -> &'static str {
        "hive"
    }
}

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
    // Use daemon-lifecycle's generic get_daemon function
    let wrapper = HiveConfigWrapper(&config);
    let hive = get_daemon(&wrapper, &request.alias, Some(job_id)).await?;

    Ok(HiveGetResponse { hive })
}
