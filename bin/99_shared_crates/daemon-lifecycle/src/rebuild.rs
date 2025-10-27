//! Daemon rebuild utilities
//!
//! TEAM-316: Extracted common rebuild patterns from queen-lifecycle and hive-lifecycle
//! TEAM-328: Added conditional hot reload behavior
//!
//! Provides reusable functions for rebuilding daemons from source:
//! - Conditional hot reload (running → stop → rebuild → start → running)
//! - Cold rebuild (stopped → rebuild → stopped)
//! - Local cargo build execution
//! - Build output handling

use anyhow::Result;
use observability_narration_core::n;
use observability_narration_macros::with_job_id;

use crate::types::rebuild::RebuildConfig; // TEAM-329: Moved to types/rebuild.rs
use crate::types::start::HttpDaemonConfig; // TEAM-329: types/start.rs (renamed from lifecycle.rs)

/// Rebuild daemon with conditional hot reload
///
/// TEAM-328: Implements conditional hot reload behavior:
/// - running → stop → rebuild → start → running (hot reload)
/// - stopped → rebuild → stopped (cold rebuild)
///
/// # Arguments
/// * `rebuild_config` - Rebuild configuration (binary name, features, job_id)
/// * `daemon_config` - HTTP daemon configuration (for start/stop operations)
///
/// # Returns
/// * `Ok(bool)` - true if daemon was restarted (hot reload), false if left stopped
/// * `Err` - Build or lifecycle operation failed
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::rebuild::{RebuildConfig, rebuild_with_hot_reload};
/// use daemon_contract::HttpDaemonConfig;
///
/// # async fn example() -> anyhow::Result<()> {
/// let rebuild_config = RebuildConfig::new("queen-rbee")
///     .with_features(vec!["local-hive".to_string()])
///     .with_job_id("job-123");
///
/// let daemon_config = HttpDaemonConfig {
///     daemon_name: "queen-rbee".to_string(),
///     binary_path: None, // Auto-resolve
///     health_url: "http://localhost:7833".to_string(),
///     args: vec![],
///     env: vec![],
///     job_id: Some("job-123".to_string()),
/// };
///
/// let was_restarted = rebuild_with_hot_reload(rebuild_config, daemon_config).await?;
/// if was_restarted {
///     println!("Hot reload complete - daemon restarted");
/// } else {
///     println!("Cold rebuild complete - daemon left stopped");
/// }
/// # Ok(())
/// # }
/// ```
#[with_job_id(config_param = "rebuild_config")] // TEAM-328: Eliminates job_id context boilerplate
pub async fn rebuild_daemon(
    rebuild_config: RebuildConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<bool> {
    // Step 1: Check if daemon is currently running
    let was_running = crate::status::check_daemon_health(
        &daemon_config.health_url,
        None, // Use default /health endpoint
        Some(std::time::Duration::from_secs(2)),
    )
    .await;

    if was_running {
        n!(
            "hot_reload_start",
            "🔄 Hot reload detected - {} is running, will restart after rebuild",
            rebuild_config.binary_name
        );

        // Step 2: Stop the running daemon
        n!("hot_reload_stop", "⏸️  Stopping {}...", rebuild_config.binary_name);
        crate::stop::stop_daemon(daemon_config.clone()).await?;
        n!("hot_reload_stopped", "✅ {} stopped", rebuild_config.binary_name);
    } else {
        n!(
            "cold_rebuild_start",
            "🔨 Cold rebuild - {} is not running, will rebuild only",
            rebuild_config.binary_name
        );
    }

    // Step 3: Build and install the daemon
    // TEAM-328: install_daemon now builds if needed, no separate build function
    let binary_path = crate::install::install_daemon(
        &rebuild_config.binary_name,
        None, // Will auto-build if not found
        None, // Install to ~/.local/bin
    )
    .await?;

    // Step 4: If it was running, start it again (hot reload)
    if was_running {
        n!("hot_reload_restart", "▶️  Restarting {}...", rebuild_config.binary_name);

        // Update daemon config with built binary path
        let mut start_config = daemon_config;
        start_config.binary_path = Some(binary_path.into());

        crate::start::start_daemon(start_config).await?;
        n!(
            "hot_reload_complete",
            "✅ Hot reload complete - {} is running with new binary",
            rebuild_config.binary_name
        );
        Ok(true)
    } else {
        n!(
            "cold_rebuild_complete",
            "✅ Cold rebuild complete - {} binary updated, daemon left stopped",
            rebuild_config.binary_name
        );
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rebuild_config_builder() {
        let config = RebuildConfig::new("test-daemon")
            .with_features(vec!["feature1".to_string(), "feature2".to_string()])
            .with_job_id("job-123");

        assert_eq!(config.binary_name, "test-daemon");
        assert_eq!(config.features, Some(vec!["feature1".to_string(), "feature2".to_string()]));
        assert_eq!(config.job_id, Some("job-123".to_string()));
    }

    #[test]
    fn test_rebuild_config_no_features() {
        let config = RebuildConfig::new("test-daemon");

        assert_eq!(config.binary_name, "test-daemon");
        assert_eq!(config.features, None);
        assert_eq!(config.job_id, None);
    }
}
