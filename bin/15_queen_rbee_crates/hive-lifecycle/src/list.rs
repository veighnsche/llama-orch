// TEAM-211: List all configured hives

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveListRequest, HiveListResponse, HiveInfo};

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

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
    NARRATE
        .action("hive_list")
        .job_id(job_id)
        .human("📊 Listing all hives")
        .emit();

    let hives: Vec<HiveInfo> = config
        .hives
        .all()
        .iter()
        .map(|h| HiveInfo {
            alias: h.alias.clone(),
            hostname: h.hostname.clone(),
            hive_port: h.hive_port,
            binary_path: h.binary_path.clone(),
        })
        .collect();

    if hives.is_empty() {
        NARRATE
            .action("hive_empty")
            .job_id(job_id)
            .human(
                "No hives registered.\n\
                 \n\
                 To install a hive:\n\
                 \n\
                   ./rbee hive install",
            )
            .emit();
    } else {
        // Convert to JSON for table display
        let hives_json: Vec<serde_json::Value> = hives
            .iter()
            .map(|h| {
                serde_json::json!({
                    "alias": h.alias,
                    "host": h.hostname,
                    "port": h.hive_port,
                    "binary_path": h.binary_path.as_ref().unwrap_or(&"-".to_string()),
                })
            })
            .collect();

        NARRATE
            .action("hive_result")
            .job_id(job_id)
            .context(hives.len().to_string())
            .human("Found {} hive(s):")
            .table(&serde_json::Value::Array(hives_json))
            .emit();
    }

    Ok(HiveListResponse { hives })
}
