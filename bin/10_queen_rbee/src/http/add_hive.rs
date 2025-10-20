//! Add hive endpoint - simplified for E2E testing
//!
//! Created by: TEAM-160
//!
//! This endpoint adds a hive to the catalog and spawns it if it's localhost.
//! This is the entry point for the "add-hive localhost" happy flow.

use axum::{extract::State, http::StatusCode, Json};
use queen_rbee_hive_catalog::HiveCatalog;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct AddHiveRequest {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Serialize)]
pub struct AddHiveResponse {
    pub hive_id: String,
    pub status: String,
}

pub type AddHiveState = Arc<HiveCatalog>;

/// Handle POST /add-hive
///
/// Adds a hive to the catalog. If it's localhost, spawns the hive process.
///
/// # Happy Flow (from a_human_wrote_this.md lines 29-36):
/// 1. Queen adds localhost to hive catalog
/// 2. Queen spawns rbee-hive on localhost:8600
/// 3. Queen waits for first heartbeat
/// 4. On first heartbeat, queen triggers device detection
///
/// # Arguments
/// * `catalog` - Hive catalog
/// * `payload` - Add hive request
///
/// # Returns
/// * `200 OK` - Hive added successfully
/// * `500 Internal Server Error` - Failed to add hive
pub async fn handle_add_hive(
    State(catalog): State<AddHiveState>,
    Json(payload): Json<AddHiveRequest>,
) -> Result<(StatusCode, Json<AddHiveResponse>), (StatusCode, String)> {
    println!("👑 Adding hive {} to catalog", payload.host);

    // Step 1: Add to hive catalog
    use queen_rbee_hive_catalog::{HiveRecord, HiveStatus};
    
    let now_ms = chrono::Utc::now().timestamp_millis();
    let hive = HiveRecord {
        id: payload.host.clone(),
        host: payload.host.clone(),
        port: payload.port,
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
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    println!("👑 Hive {} added to catalog", payload.host);

    // TODO: If localhost, spawn rbee-hive process
    // TODO: Wait for first heartbeat
    // TODO: Trigger device detection

    let response = AddHiveResponse {
        hive_id: payload.host.clone(),
        status: "added".to_string(),
    };

    Ok((StatusCode::OK, Json(response)))
}
