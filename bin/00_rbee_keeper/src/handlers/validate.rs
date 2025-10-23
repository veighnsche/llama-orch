//! Config validation command handler
//!
//! TEAM-282: Added for declarative lifecycle management

use anyhow::Result;
use rbee_operations::Operation;

use crate::job_client::submit_and_stream_job;

/// Validate declarative config file
///
/// # Arguments
///
/// * `queen_url` - URL of queen-rbee API
/// * `config_path` - Optional: path to config file (default: ~/.config/rbee/hives.conf)
pub async fn handle_validate(queen_url: &str, config_path: Option<String>) -> Result<()> {
    let operation = Operation::PackageValidate { config_path };

    submit_and_stream_job(queen_url, operation).await
}
