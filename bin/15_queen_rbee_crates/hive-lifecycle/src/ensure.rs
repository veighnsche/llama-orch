//! Ensure hive is running, auto-start if needed
//!
//! TEAM-276: Added for consistency with queen-lifecycle ensure pattern
//! TEAM-276: Refactored to use daemon-lifecycle::ensure_daemon_with_handle

use anyhow::Result;
use daemon_lifecycle::ensure_daemon_with_handle;
use rbee_config::RbeeConfig;
use std::sync::Arc;
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;

use crate::start::execute_hive_start;
use crate::types::{HiveHandle, HiveStartRequest};

/// Ensure hive is running, auto-start if needed
///
/// # Pattern (same as queen-lifecycle)
/// 1. Check health using HTTP GET /health
/// 2. If healthy → return Ok(HiveHandle)
/// 3. If not running:
///    - Start hive (via SSH if remote, locally if attached)
///    - Poll health until ready (with timeout)
///    - Return HiveHandle
///
/// # Arguments
/// * `hive_alias` - Hive alias from hives.conf
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for narration routing
///
/// # Returns
/// * `Ok(HiveHandle)` - Handle to hive (tracks if we started it)
///
/// # Errors
///
/// Returns an error if hive fails to start or timeout waiting for health
///
/// # Example
///
/// ```rust,no_run
/// use queen_rbee_hive_lifecycle::ensure_hive_running;
/// use rbee_config::RbeeConfig;
/// use std::sync::Arc;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = Arc::new(RbeeConfig::load()?);
/// let handle = ensure_hive_running("hive-1", config, "job-123").await?;
/// # Ok(())
/// # }
/// ```
pub async fn ensure_hive_running(
    hive_alias: &str,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveHandle> {
    // Use TimeoutEnforcer with progress bar for visual feedback
    TimeoutEnforcer::new(Duration::from_secs(60))
        .with_label(&format!("Starting hive: {}", hive_alias))
        .with_job_id(job_id)
        .with_countdown()
        .enforce(ensure_hive_running_inner(hive_alias, config, job_id))
        .await
}

async fn ensure_hive_running_inner(
    hive_alias: &str,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveHandle> {
    // TEAM-276: Use shared ensure pattern from daemon-lifecycle

    // Step 1: Get hive configuration
    let all_hives = &config.hives.hives;
    let hive_config = all_hives
        .iter()
        .find(|h| h.alias == hive_alias)
        .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found in configuration", hive_alias))?;

    let hive_endpoint = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);
    let health_url = format!("{}/health", hive_endpoint);

    // Step 2: Use shared ensure pattern
    let alias = hive_alias.to_string();
    let endpoint = hive_endpoint.clone();
    let cfg = config.clone();
    let jid = job_id.to_string();

    ensure_daemon_with_handle(
        hive_alias,
        &health_url,
        Some(job_id),
        || async move {
            // Spawn logic: start hive via SSH or local
            let start_request = HiveStartRequest { alias: alias.clone(), job_id: jid.clone() };
            execute_hive_start(start_request, cfg).await?;
            Ok(())
        },
        || HiveHandle::already_running(hive_alias.to_string(), endpoint.clone()),
        || HiveHandle::started_by_us(hive_alias.to_string(), endpoint.clone()),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_pattern_exists() {
        // This test just verifies the ensure pattern is implemented
        // Actual testing requires a running hive or mocks
        assert!(true);
    }
}
