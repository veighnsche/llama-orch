//! Status command handler
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-190: Show live status of all hives and workers from registry

use anyhow::Result;
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations

use crate::job_client::submit_and_stream_job;

pub async fn handle_status(queen_url: &str) -> Result<()> {
    let operation = Operation::Status;
    submit_and_stream_job(queen_url, operation).await
}
