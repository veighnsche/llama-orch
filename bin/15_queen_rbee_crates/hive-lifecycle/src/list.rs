// TEAM-211: List all configured hives
// TEAM-220: Investigated - Simple query operation documented
// TEAM-259: Refactored to use daemon-lifecycle::list_daemons

use anyhow::Result;
use daemon_lifecycle::{list_daemons, ListableConfig};
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveInfo, HiveListRequest, HiveListResponse};

// Wrapper struct to implement ListableConfig (avoids orphan rule)
struct HiveConfigWrapper<'a>(&'a RbeeConfig);

impl<'a> ListableConfig for HiveConfigWrapper<'a> {
    type Info = HiveInfo;

    fn list_all(&self) -> Vec<Self::Info> {
        self.0
            .hives
            .hives
            .iter()
            .map(|h| HiveInfo {
                alias: h.alias.clone(),
                hostname: h.hostname.clone(),
                hive_port: h.hive_port,
                binary_path: h.binary_path.clone(),
            })
            .collect()
    }

    fn daemon_type_name(&self) -> &'static str {
        "hive"
    }
}

/// List all configured hives
///
/// Returns all hives from config (hives.conf).
/// If no hives configured, returns empty list.
///
/// # Arguments
/// * `request` - List request (no parameters)
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveListResponse)` - List of hives
/// * `Err` - Configuration error
pub async fn execute_hive_list(
    _request: HiveListRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveListResponse> {
    // Use daemon-lifecycle's generic list_daemons function
    let wrapper = HiveConfigWrapper(&config);
    let hives = list_daemons(&wrapper, Some(job_id)).await?;

    Ok(HiveListResponse { hives })
}
