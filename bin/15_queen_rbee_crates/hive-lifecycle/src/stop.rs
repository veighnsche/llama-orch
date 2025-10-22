// TEAM-212: Stop hive daemon

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use crate::types::{HiveStopRequest, HiveStopResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Stop hive daemon
///
/// COPIED FROM: job_router.rs lines 718-820
///
/// Steps:
/// 1. Check if running
/// 2. Send SIGTERM (graceful shutdown)
/// 3. Wait 5 seconds for graceful shutdown
/// 4. If still running, send SIGKILL (force kill)
///
/// # Arguments
/// * `request` - Stop request with alias and job_id
/// * `config` - RbeeConfig with hive configuration
///
/// # Returns
/// * `Ok(HiveStopResponse)` - Success message
/// * `Err` - Failed to stop
pub async fn execute_hive_stop(
    request: HiveStopRequest,
    config: Arc<RbeeConfig>,
) -> Result<HiveStopResponse> {
    let job_id = &request.job_id;
    let alias = &request.alias;
    
    let hive_config = validate_hive_exists(&config, alias)?;

    NARRATE
        .action("hive_stop")
        .job_id(job_id)
        .context(alias)
        .human("🛑 Stopping hive '{}'")
        .emit();

    // Check if it's running
    NARRATE
        .action("hive_check")
        .job_id(job_id)
        .human("📋 Checking if hive is running...")
        .emit();

    let health_url = format!(
        "http://{}:{}/health",
        hive_config.hostname, hive_config.hive_port
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    if let Ok(response) = client.get(&health_url).send().await {
        if !response.status().is_success() {
            NARRATE
                .action("hive_not_run")
                .job_id(job_id)
                .context(alias)
                .human("⚠️  Hive '{}' is not running")
                .emit();
            return Ok(HiveStopResponse {
                success: true,
                message: format!("Hive '{}' is not running", alias),
            });
        }
    } else {
        NARRATE
            .action("hive_not_run")
            .job_id(job_id)
            .context(alias)
            .human("⚠️  Hive '{}' is not running")
            .emit();
        return Ok(HiveStopResponse {
            success: true,
            message: format!("Hive '{}' is not running", alias),
        });
    }

    // Stop the hive process
    NARRATE
        .action("hive_sigterm")
        .job_id(job_id)
        .human("📤 Sending SIGTERM (graceful shutdown)...")
        .emit();

    // Use pkill to stop the hive by binary name
    let binary_path = hive_config
        .binary_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Hive '{}' has no binary_path configured", alias))?;

    let binary_name = std::path::Path::new(&binary_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("rbee-hive");

    // Send SIGTERM
    let output = tokio::process::Command::new("pkill")
        .args(&["-TERM", binary_name])
        .output()
        .await?;

    if !output.status.success() {
        NARRATE
            .action("hive_not_found")
            .job_id(job_id)
            .context(binary_name)
            .human("⚠️  No running process found for '{}'")
            .emit();
        return Ok(HiveStopResponse {
            success: true,
            message: format!("No running process found for '{}'", binary_name),
        });
    }

    // Wait for graceful shutdown
    NARRATE
        .action("hive_wait")
        .job_id(job_id)
        .human("⏳ Waiting for graceful shutdown (5s)...")
        .emit();

    for attempt in 1..=5 {
        sleep(Duration::from_secs(1)).await;

        if let Err(_) = client.get(&health_url).send().await {
            // Health check failed - hive stopped
            NARRATE
                .action("hive_success")
                .job_id(job_id)
                .context(alias)
                .human("✅ Hive '{}' stopped successfully")
                .emit();
            return Ok(HiveStopResponse {
                success: true,
                message: format!("Hive '{}' stopped successfully", alias),
            });
        }

        if attempt == 5 {
            // Timeout - force kill
            NARRATE
                .action("hive_sigkill")
                .job_id(job_id)
                .human("⚠️  Graceful shutdown timed out, sending SIGKILL...")
                .emit();

            tokio::process::Command::new("pkill")
                .args(&["-KILL", binary_name])
                .output()
                .await?;

            sleep(Duration::from_millis(500)).await;

            NARRATE
                .action("hive_forced")
                .job_id(job_id)
                .context(alias)
                .human("✅ Hive '{}' force-stopped")
                .emit();
        }
    }

    Ok(HiveStopResponse {
        success: true,
        message: format!("Hive '{}' stopped", alias),
    })
}
