//! Package status command handler
//!
//! TEAM-282: Added for declarative lifecycle management

use anyhow::Result;
use rbee_operations::Operation;

use crate::job_client::submit_and_stream_job;

/// Check package status and detect drift from config
///
/// # Arguments
///
/// * `queen_url` - URL of queen-rbee API
/// * `config_path` - Optional path to config file (TEAM-260: for testing different scenarios)
/// * `verbose` - Show detailed status information
pub async fn handle_package_status(queen_url: &str, config_path: Option<String>, verbose: bool) -> Result<()> {
    let operation = Operation::PackageStatus {
        config_path, // TEAM-260: Pass through custom config path
        verbose,
    };

    submit_and_stream_job(queen_url, operation).await
}
