//! Config migration command handler
//!
//! TEAM-282: Added for declarative lifecycle management

use anyhow::Result;
use rbee_operations::Operation;

use crate::job_client::submit_and_stream_job;

/// Generate declarative config from current state
///
/// # Arguments
///
/// * `queen_url` - URL of queen-rbee API
/// * `output_path` - Path where config should be written
pub async fn handle_migrate(queen_url: &str, output_path: String) -> Result<()> {
    let operation = Operation::PackageMigrate { output_path };

    submit_and_stream_job(queen_url, operation).await
}
