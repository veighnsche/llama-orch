//! Package sync command handler
//!
//! TEAM-282: Added for declarative lifecycle management

use anyhow::Result;
use rbee_operations::Operation;

use crate::job_client::submit_and_stream_job;

/// Sync all hives to match declarative config
///
/// # Arguments
///
/// * `queen_url` - URL of queen-rbee API
/// * `config_path` - Optional path to config file (TEAM-260: for testing different scenarios)
/// * `dry_run` - Show what would be done without making changes
/// * `remove_extra` - Remove components not in config
/// * `force` - Force reinstall even if already installed
/// * `hive_alias` - Optional: sync only this hive
pub async fn handle_sync(
    queen_url: &str,
    config_path: Option<String>,
    dry_run: bool,
    remove_extra: bool,
    force: bool,
    _hive_alias: Option<String>, // TODO: TEAM-283 can add single-hive sync support
) -> Result<()> {
    let operation = Operation::PackageSync {
        config_path, // TEAM-260: Pass through custom config path
        dry_run,
        remove_extra,
        force,
    };

    submit_and_stream_job(queen_url, operation).await
}
