//! Generic forwarding for hive-managed operations
//!
//! TEAM-258: Consolidate hive-forwarding operations
//! TEAM-259: Refactored to use job-client shared crate
//! TEAM-265: Documented three communication modes
//!
//! This module handles forwarding of Worker and Model operations
//! to the appropriate hive. This allows new operations to be added
//! to rbee-hive without requiring changes to queen-rbee's job_router.
//!
//! # Three Communication Modes
//!
//! Queen can communicate with hives in three ways:
//!
//! ## 1. Remote HTTP (Default - Always Available)
//! ```text
//! queen-rbee → HTTP → remote-hive (different machine)
//! - Use case: Production multi-machine setup
//! - Overhead: ~5-10ms per operation (network + serialization)
//! - Example: queen on server1, hive on server2
//! ```
//!
//! ## 2. Localhost HTTP (Default - When hive_id="localhost")
//! ```text
//! queen-rbee → HTTP → localhost-hive (same machine, different process)
//! - Use case: Development, testing, or distributed queen build
//! - Overhead: ~1-2ms per operation (loopback + serialization)
//! - Example: queen on port 8500, hive on port 8600
//! ```
//!
//! ## 3. Integrated (local-hive feature - When hive_id="localhost")
//! ```text
//! queen-rbee → Direct Rust calls → integrated hive (same process)
//! - Use case: Single-machine production (optimal performance)
//! - Overhead: ~0.01ms per operation (direct function calls)
//! - Example: queen with --features local-hive
//! - NOTE: 50-100x faster than localhost HTTP!
//! ```
//!
//! # Mode Selection Logic
//!
//! ```rust,ignore
//! if hive_id == "localhost" && cfg!(feature = "local-hive") {
//!     // Mode 3: Integrated (direct calls)
//!     execute_locally_integrated(operation).await
//! } else if hive_id == "localhost" {
//!     // Mode 2: Localhost HTTP
//!     forward_via_http("http://localhost:8600", operation).await
//! } else {
//!     // Mode 1: Remote HTTP
//!     forward_via_http("http://{remote_host}:{port}", operation).await
//! }
//! ```
//!
//! # Current Implementation Status
//!
//! - ✅ Mode 1: Remote HTTP (implemented)
//! - ✅ Mode 2: Localhost HTTP (implemented)
//! - ⚠️  Mode 3: Integrated (TODO - requires local-hive feature)
//!
//! # Architecture Diagram
//!
//! ```text
//! queen-rbee (client)
//!     ↓
//! hive_forwarder::forward_to_hive()
//!     ↓
//! Mode detection (hive_id + feature check)
//!     ↓
//! ┌─────────────────┬──────────────────┬────────────────────┐
//! │ Mode 1: Remote  │ Mode 2: Localhost│ Mode 3: Integrated │
//! │ HTTP            │ HTTP             │ Direct Calls       │
//! ├─────────────────┼──────────────────┼────────────────────┤
//! │ job_client      │ job_client       │ rbee-hive crates   │
//! │ → HTTP POST     │ → HTTP POST      │ → Direct function  │
//! │ → SSE stream    │ → SSE stream     │ → In-memory result │
//! └─────────────────┴──────────────────┴────────────────────┘
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
/// TEAM-265: Added mode detection and documentation
///
/// # Communication Modes
///
/// This function automatically selects the appropriate communication mode:
///
/// 1. **Integrated Mode** (local-hive feature + localhost):
///    - Direct Rust function calls (~0.01ms overhead)
///    - 50-100x faster than HTTP
///    - Requires queen built with --features local-hive
///
/// 2. **Localhost HTTP** (localhost without local-hive feature):
///    - HTTP over loopback (~1-2ms overhead)
///    - Separate rbee-hive process on same machine
///
/// 3. **Remote HTTP** (non-localhost):
///    - HTTP over network (~5-10ms overhead)
///    - rbee-hive on different machine
///
/// # Arguments
///
/// * `job_id` - Job ID for SSE routing
/// * `operation` - Operation to forward (must have hive_id)
/// * `config` - Configuration containing hive endpoints
///
/// # Returns
///
/// `Ok(())` if operation completed successfully
///
/// # Errors
///
/// - Operation doesn't have hive_id
/// - Hive not found in configuration
/// - HTTP communication failure
/// - Hive failed to start
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

    // TEAM-265: Detect communication mode
    let is_localhost = hive_id == "localhost";
    let has_integrated = cfg!(feature = "local-hive");
    
    let mode = if is_localhost && has_integrated {
        "integrated"
    } else if is_localhost {
        "localhost-http"
    } else {
        "remote-http"
    };

    NARRATE
        .action("forward_start")
        .job_id(job_id)
        .context(operation_name)
        .context(&hive_id)
        .context(mode)
        .human("Forwarding {} operation to hive '{}' (mode: {})")
        .emit();

    // TEAM-265: TODO - Implement integrated mode
    // TEAM-266: Investigation complete - See bin/.plan/TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md
    //
    // CRITICAL: Mode 3 implementation is BLOCKED by missing rbee-hive crate implementations.
    // All worker-lifecycle, model-catalog, and model-provisioner crates are empty stubs.
    // 
    // Prerequisites before Mode 3 implementation:
    // 1. Implement worker-lifecycle crate functions (spawn, list, get, delete)
    // 2. Implement model-catalog crate functions (list, get, delete)
    // 3. Implement model-provisioner crate functions (download)
    // 4. Test HTTP mode (Mode 2) thoroughly
    // 5. Document public APIs
    // 
    // Expected effort: 180+ hours for prerequisites, 30-58 hours for Mode 3
    // Expected speedup: 110x for list/get operations, minimal for spawn/download
    //
    // When local-hive feature is enabled and hive_id == "localhost",
    // we should call rbee-hive crates directly instead of HTTP.
    // This requires:
    // 1. Add rbee-hive crates as optional dependencies
    // 2. Implement execute_integrated() function with direct calls
    // 3. Convert results to narration events (no HTTP/SSE needed)
    // 
    // For now, we always use HTTP (modes 1 & 2).
    // This is correct but not optimal for localhost with local-hive feature.
    
    if is_localhost && has_integrated {
        NARRATE
            .action("forward_mode")
            .job_id(job_id)
            .human("⚠️  Integrated mode detected but not yet implemented - falling back to HTTP")
            .emit();
    }

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
