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
/// * `verbose` - Show detailed status information
pub async fn handle_package_status(queen_url: &str, verbose: bool) -> Result<()> {
    let operation = Operation::PackageStatus {
        config_path: None, // Use default config path
        verbose,
    };

    submit_and_stream_job(queen_url, operation).await
}
