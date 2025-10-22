//! Generic HTTP forwarding for hive-managed operations
//!
//! TEAM-258: Consolidate hive-forwarding operations
//! TEAM-259: Refactored to use rbee-job-client shared crate
//!
//! This module handles forwarding of Worker and Model operations
//! to the appropriate hive via HTTP. This allows new operations to be added
//! to rbee-hive without requiring changes to queen-rbee's job_router.
//!
//! # Architecture
//!
//! ```text
//! queen-rbee (client)
//!     ↓
//! hive_forwarder::forward_to_hive()
//!     ↓
//! rbee_job_client::JobClient
//!     ↓
//! POST http://{hive_host}:{hive_port}/v1/jobs
//!     ↓
//! GET http://{hive_host}:{hive_port}/v1/jobs/{job_id}/stream
//!     ↓
//! Stream responses back to client
//! ```

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use rbee_job_client::JobClient;
use rbee_operations::Operation;
use std::sync::Arc;

const NARRATE: NarrationFactory = NarrationFactory::new("qn-fwd");

/// Forward an operation to the appropriate hive
///
/// TEAM-258: Generic forwarding for all hive-managed operations
/// TEAM-259: Refactored to use rbee-job-client shared crate
///
/// Extracts hive_id from operation, looks up hive config, and forwards via HTTP.
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    config: Arc<RbeeConfig>,
) -> Result<()> {
    // Extract metadata before moving operation
    let operation_name = operation.name();
    let hive_id = operation
        .hive_id()
        .ok_or_else(|| anyhow::anyhow!("Operation does not target a hive"))?
        .to_string();

    NARRATE
        .action("forward_start")
        .job_id(job_id)
        .context(operation_name)
        .context(&hive_id)
        .human("Forwarding {} operation to hive '{}'")
        .emit();

    // Look up hive in config
    let hive_config = config
        .hives
        .get(&hive_id)
        .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found in configuration", hive_id))?;

    // Determine hive host and port
    let hive_host = &hive_config.hostname;
    let hive_port = hive_config.hive_port;

    let hive_url = format!("http://{}:{}", hive_host, hive_port);

    NARRATE
        .action("forward_connect")
        .job_id(job_id)
        .context(&hive_url)
        .human("Connecting to hive at {}")
        .emit();

    // Forward to hive and stream responses
    stream_from_hive(job_id, &hive_url, operation).await?;

    NARRATE
        .action("forward_complete")
        .job_id(job_id)
        .context(&hive_id)
        .human("Operation completed on hive '{}'")
        .emit();

    Ok(())
}

/// Stream responses from hive back to client
///
/// TEAM-259: Extracted to separate function for clarity (mirrors job_client.rs)
async fn stream_from_hive(
    job_id: &str,
    hive_url: &str,
    operation: Operation,
) -> Result<()> {
    // TEAM-259: Use shared JobClient for submission and streaming
    let client = JobClient::new(hive_url);

    client
        .submit_and_stream(operation, |line| {
            // Forward each line to client via narration
            NARRATE
                .action("forward_data")
                .job_id(job_id)
                .context(line)
                .human("{}")
                .emit();
            Ok(())
        })
        .await?;

    Ok(())
}
