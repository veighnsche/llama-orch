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

/// Execute hive start orchestration
///
/// **Pattern:** Command Pattern (see CRATE_INTERFACE_STANDARD.md)
///
/// **IMPORTANT:** This function does NOT wait for the hive to be ready!
/// The hive will send a heartbeat when it's online, which acts as the callback.
///
/// Flow:
/// 1. Queen decides where to spawn (localhost for now)
/// 2. Queen decides port (8600)
/// 3. Queen adds hive to catalog (status: Unknown)
/// 4. Queen spawns hive process (fire and forget)
/// 5. Hive starts up asynchronously
/// 6. Hive sends heartbeat → Queen receives → Updates catalog → Triggers device detection
///
/// The heartbeat is the callback mechanism - queen doesn't wait!
///
/// # Arguments
/// * `catalog` - Hive catalog for persistence
/// * `request` - Hive start request
///
/// # Returns
/// * `Ok(HiveStartResponse)` - Hive spawn initiated (NOT necessarily running yet)
/// * `Err` - Failed to spawn hive
pub async fn execute_hive_start(
    catalog: Arc<HiveCatalog>,
    request: HiveStartRequest,
) -> Result<HiveStartResponse> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_START, "start")
        .human("🐝 Executing hive start")
        .emit();

    // ============================================================
    // ORCHESTRATION LOGIC - ALL DECISIONS HAPPEN HERE
    // ============================================================
    // For now: Always spawn on localhost:8600
    // Future: Check available nodes, pick best one, use SSH if remote
    // ============================================================

    let host = "localhost".to_string();
    let port = 8600u16;

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_ORCHESTRATE, &format!("{}:{}", host, port))
        .human(format!("🐝 Orchestrating: spawn hive on {}:{}", host, port))
        .emit();

    // Step 1: Add to hive catalog
    let now_ms = chrono::Utc::now().timestamp_millis();
    let hive = HiveRecord {
        id: host.clone(),
        host: host.clone(),
        port,
        ssh_host: None,
        ssh_port: None,
        ssh_user: None,
        devices: None,
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
    };

    catalog.add_hive(hive).await.context("Failed to add hive to catalog")?;

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_START, &host)
        .human(format!("🐝 Hive {} added to catalog", host))
        .emit();

    // Step 2: Spawn the hive process
    spawn_hive(port, &request.queen_url).await?;

    let hive_url = format!("http://{}:{}", host, port);

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_START, &hive_url)
        .human(format!("✅ Hive spawn initiated: {} (will send heartbeat when ready)", hive_url))
        .emit();

    // Return structured response (Command Pattern)
    // NOTE: Hive is NOT necessarily running yet - it will send heartbeat when ready
    Ok(HiveStartResponse { hive_url, hive_id: host, port })
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
