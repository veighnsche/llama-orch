// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-164: Implemented hive lifecycle management
// Purpose: Lifecycle management for rbee-hive instances

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-rbee-hive-lifecycle
//!
//! **Category:** Orchestration
//! **Pattern:** Command Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Lifecycle management for rbee-hive instances.
//! Queen orchestrates hive spawning - rbee-keeper does NOT.
//!
//! # Interface
//!
//! ## Request Type
//! ```rust
//! pub struct HiveStartRequest {
//!     pub queen_url: String,
//! }
//! ```
//!
//! ## Response Type
//! ```rust
//! pub struct HiveStartResponse {
//!     pub hive_url: String,
//!     pub hive_id: String,
//!     pub port: u16,
//! }
//! ```
//!
//! ## Entrypoint
//! ```rust
//! pub async fn execute_hive_start(
//!     catalog: Arc<HiveCatalog>,
//!     request: HiveStartRequest,
//! ) -> Result<HiveStartResponse>
//! ```

use anyhow::{Context, Result};
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::{HiveCatalog, HiveRecord, HiveStatus};
use std::process::{Command, Stdio};
use std::sync::Arc;

// Actor and action constants
const ACTOR_HIVE_LIFECYCLE: &str = "🐝 hive-lifecycle";
const ACTION_START: &str = "hive_start";
const ACTION_SPAWN: &str = "spawn_hive";
const ACTION_ORCHESTRATE: &str = "orchestrate";

// ============================================================================
// REQUEST / RESPONSE TYPES (Command Pattern)
// ============================================================================

/// Request to start a hive
#[derive(Debug, Clone)]
pub struct HiveStartRequest {
    /// URL of the queen-rbee instance
    pub queen_url: String,
}

/// Response from starting a hive
#[derive(Debug, Clone)]
pub struct HiveStartResponse {
    /// Full URL to access the hive
    pub hive_url: String,
    /// Hive identifier
    pub hive_id: String,
    /// Port the hive is listening on
    pub port: u16,
}

/// Execute hive start command
///
/// TEAM-186: Removed catalog manipulation - lifecycle does NOT touch configuration!
///
/// **IMPORTANT: Async Fire-and-Forget Pattern**
///
/// This function:
/// 1. Verifies hive exists in catalog (FAIL FAST if not registered)
/// 2. Spawns the hive process
/// 3. Returns immediately (does NOT wait for hive to be ready)
/// 4. Hive will send heartbeat when ready
///
/// **CRITICAL: Hive MUST be registered by user FIRST!**
/// - Catalog is configuration managed by the user
/// - Lifecycle only starts/stops already-registered hives
/// - If hive not in catalog → FAIL FAST
///
/// Flow:
/// 1. Queen receives "start hive" command
/// 2. Queen checks catalog (hive must exist!)
/// 3. Queen spawns hive process → Returns "spawn initiated"
/// 4. Hive starts up (takes time)
/// 5. Hive sends heartbeat → Registry updated (RAM only!)
///
/// # Arguments
/// * `catalog` - Hive catalog (READ-ONLY - for verification)
/// * `request` - Hive start request
///
/// # Returns
/// * `Ok(HiveStartResponse)` - Hive spawn initiated (NOT necessarily running yet)
/// * `Err` - Failed to spawn hive (or hive not registered)
pub async fn execute_hive_start(
    catalog: Arc<HiveCatalog>,
    request: HiveStartRequest,
) -> Result<HiveStartResponse> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_START, "start")
        .human("🐝 Executing hive start")
        .emit();

    // TEAM-186: FAIL FAST - Hive MUST be registered in catalog first!
    let hive = catalog
        .get_hive(&request.hive_id)
        .await
        .context("Failed to check catalog")?
        .ok_or_else(|| {
            anyhow!(
                "Hive '{}' not found in catalog! Register the hive first.",
                request.hive_id
            )
        })?;

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_START, &hive.id)
        .human(format!("✅ Hive '{}' found in catalog", hive.id))
        .emit();

    // TEAM-186: Spawn the hive process (no catalog changes!)
    spawn_hive(hive.port, &request.queen_url).await?;

    let hive_url = format!("http://{}:{}", hive.host, hive.port);

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_START, &hive_url)
        .human(format!("✅ Hive spawn initiated: {} (will send heartbeat when ready)", hive_url))
        .emit();

    // Return structured response (Command Pattern)
    // NOTE: Hive is NOT necessarily running yet - it will send heartbeat when ready
    Ok(HiveStartResponse {
        hive_url,
        hive_id: hive.id,
        port: hive.port,
    })
}

/// Spawn a hive process (internal helper)
///
/// # Arguments
/// * `port` - Port for the hive to listen on
/// * `queen_url` - URL of the queen to report to
///
/// # Returns
/// * `Ok(())` - Hive spawned successfully
/// * `Err` - Failed to spawn hive
async fn spawn_hive(port: u16, queen_url: &str) -> Result<()> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_SPAWN, &format!("port:{}", port))
        .human(format!("🐝 Spawning rbee-hive on port {}", port))
        .emit();

    // TEAM-164: Use Stdio::null() for daemon processes
    // This prevents hanging when parent uses Command::output()
    let _child = Command::new("target/debug/rbee-hive")
        .arg("--port")
        .arg(port.to_string())
        .arg("--queen-url")
        .arg(queen_url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn rbee-hive")?;

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_SPAWN, "success")
        .human("🐝 rbee-hive spawned successfully")
        .emit();

    Ok(())
}
