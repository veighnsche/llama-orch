// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-164: Implemented hive lifecycle management
// Purpose: Lifecycle management for rbee-hive instances

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-rbee-hive-lifecycle
//!
//! Lifecycle management for rbee-hive instances
//!
//! Similar to queen-lifecycle, but for hive processes.
//! Queen orchestrates hive spawning - rbee-keeper does NOT.

use anyhow::{Context, Result};
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::{HiveCatalog, HiveRecord, HiveStatus};
use std::process::{Command, Stdio};
use std::sync::Arc;


// Actor and action constants
const ACTOR_HIVE_LIFECYCLE: &str = "🐝 hive-lifecycle";
const ACTION_ENSURE: &str = "ensure_hive_running";
const ACTION_SPAWN: &str = "spawn_hive";
const ACTION_ORCHESTRATE: &str = "orchestrate";

/// Ensure a hive is running
///
/// This is THE orchestration function. It:
/// 1. Decides where to spawn the hive (localhost for now)
/// 2. Decides what port to use (8600)
/// 3. Adds hive to catalog
/// 4. Spawns the hive process
///
/// # Arguments
/// * `catalog` - Hive catalog
/// * `queen_url` - URL of the queen
///
/// # Returns
/// * `Ok(hive_url)` - Hive is running, returns URL
/// * `Err` - Failed to start hive
pub async fn ensure_hive_running(
    catalog: Arc<HiveCatalog>,
    queen_url: &str,
) -> Result<String> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_ENSURE, "start")
        .human("🐝 Ensuring hive is running")
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
        status: HiveStatus::Unknown,
        last_heartbeat_ms: None,
        devices: None,
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
    };
    
    catalog.add_hive(hive).await
        .context("Failed to add hive to catalog")?;
    
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_ENSURE, &host)
        .human(format!("🐝 Hive {} added to catalog", host))
        .emit();

    // Step 2: Spawn the hive process
    spawn_hive(port, queen_url).await?;

    let hive_url = format!("http://{}:{}", host, port);
    
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_ENSURE, &hive_url)
        .human(format!("✅ Hive running: {}", hive_url))
        .emit();

    Ok(hive_url)
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
