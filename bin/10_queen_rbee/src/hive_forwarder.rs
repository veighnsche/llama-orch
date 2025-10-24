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
use job_client::JobClient;
use observability_narration_core::NarrationFactory;
// TEAM-285: DELETED execute_hive_start, HiveStartRequest (localhost-only, no lifecycle management)
// TEAM-290: DELETED rbee_config import (file-based config deprecated)
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations
// TEAM-290: DELETED Duration import (not needed anymore)

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
/// - Hive is not localhost
/// - HTTP communication failure
///
/// TEAM-290: Localhost-only mode (rbee-config removed)
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
) -> Result<()> {
    // Extract metadata before moving operation
    let operation_name = operation.name();
    let hive_id = operation
        .hive_id()
        .ok_or_else(|| anyhow::anyhow!("Operation does not target a hive"))?
        .to_string();

    // TEAM-290: Validate localhost only
    if hive_id != "localhost" {
        anyhow::bail!(
            "Only localhost hive is supported. Remote hives are deprecated.\n\
             Requested: '{}'\n\
             Supported: 'localhost'",
            hive_id
        );
    }

    NARRATE
        .action("forward_start")
        .job_id(job_id)
        .context(operation_name)
        .context(&hive_id)
        .human("Forwarding {} operation to localhost hive")
        .emit();

    // TEAM-290: Hardcoded localhost URL (no config needed)
    let hive_url = "http://localhost:9000";

    // TEAM-290: Skip ensure_hive_running (no lifecycle management in localhost-only mode)

    NARRATE
        .action("forward_connect")
        .job_id(job_id)
        .context(hive_url)
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
async fn stream_from_hive(job_id: &str, hive_url: &str, operation: Operation) -> Result<()> {
    // TEAM-259: Use shared JobClient for submission and streaming
    let client = JobClient::new(hive_url);

    client
        .submit_and_stream(operation, |line| {
            // Forward each line to client via narration
            NARRATE.action("forward_data").job_id(job_id).context(line).human("{}").emit();
            Ok(())
        })
        .await?;

    Ok(())
}

// TEAM-290: DELETED ensure_hive_running and is_hive_healthy (no lifecycle management in localhost-only mode)
