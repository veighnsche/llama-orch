//! Generic HTTP forwarding for hive-managed operations
//!
//! TEAM-258: Consolidate hive-forwarding operations
//! TEAM-259: Refactored to use job-client shared crate
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
//! job_client::JobClient
//!     ↓
//! POST http://{hive_host}:{hive_port}/v1/jobs
//!     ↓
//! GET http://{hive_host}:{hive_port}/v1/jobs/{job_id}/stream
//!     ↓
//! Stream responses back to client
//! ```

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use queen_rbee_hive_lifecycle::{execute_hive_start, HiveStartRequest};
use rbee_config::RbeeConfig;
use job_client::JobClient;
use rbee_operations::Operation;
use std::sync::Arc;
use std::time::Duration;

const NARRATE: NarrationFactory = NarrationFactory::new("qn-fwd");

/// Forward an operation to the appropriate hive
///
/// TEAM-258: Generic forwarding for all hive-managed operations
/// TEAM-259: Refactored to use job-client shared crate
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

    // TEAM-259: Ensure hive is running before forwarding (mirrors queen_lifecycle pattern)
    ensure_hive_running(job_id, &hive_id, &hive_url, config.clone()).await?;

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

/// Ensure hive is running before forwarding operations
///
/// TEAM-259: Mirrors rbee-keeper's ensure_queen_running pattern
/// 
/// 1. Check if hive is healthy
/// 2. If not running, start hive daemon
/// 3. Wait for health check to pass
async fn ensure_hive_running(
    job_id: &str,
    hive_id: &str,
    hive_url: &str,
    config: Arc<RbeeConfig>,
) -> Result<()> {
    // Check if hive is already healthy
    if is_hive_healthy(hive_url).await {
        NARRATE
            .action("hive_check")
            .job_id(job_id)
            .context(hive_id)
            .human("Hive '{}' is already running")
            .emit();
        return Ok(());
    }

    // Hive is not running, start it
    NARRATE
        .action("hive_start")
        .job_id(job_id)
        .context(hive_id)
        .human("⚠️  Hive '{}' is not running, starting...")
        .emit();

    // Use hive-lifecycle to start the hive
    let request = HiveStartRequest {
        alias: hive_id.to_string(),
        job_id: job_id.to_string(),
    };
    execute_hive_start(request, config).await?;

    // Wait for hive to become healthy (with timeout)
    let start_time = std::time::Instant::now();
    let timeout = Duration::from_secs(30);

    loop {
        if is_hive_healthy(hive_url).await {
            NARRATE
                .action("hive_start")
                .job_id(job_id)
                .context(hive_id)
                .human("✅ Hive '{}' is now running and healthy")
                .emit();
            return Ok(());
        }

        if start_time.elapsed() > timeout {
            return Err(anyhow::anyhow!(
                "Timeout waiting for hive '{}' to become healthy",
                hive_id
            ));
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Check if hive is healthy via HTTP health check
///
/// TEAM-259: Mirrors rbee-keeper's is_queen_healthy pattern
async fn is_hive_healthy(hive_url: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .ok();

    if let Some(client) = client {
        if let Ok(response) = client.get(format!("{}/health", hive_url)).send().await {
            return response.status().is_success();
        }
    }

    false
}
