//! Hive catalog types
//!
//! Created by: TEAM-156
//! Refactored by: TEAM-158
//! TEAM-186: Removed HiveStatus enum and runtime fields (status, last_heartbeat_ms)

use crate::device_types::DeviceCapabilities;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Hive record stored in catalog
///
/// CONFIGURATION ONLY - No runtime/heartbeat data!
/// Runtime data (status, heartbeat, workers) lives in hive-registry (RAM)
/// TEAM-186: Removed status and last_heartbeat_ms fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveRecord {
    /// Hive identifier (e.g., "localhost", "hive-prod-01")
    pub id: String,
    
    /// Host address
    pub host: String,
    
    /// Port
    pub port: u16,
    
    /// SSH host (for remote hives)
    pub ssh_host: Option<String>,
    
    /// SSH port (for remote hives)
    pub ssh_port: Option<u16>,
    
    /// SSH username (for remote hives)
    pub ssh_user: Option<String>,
    
    /// Device capabilities (CPU, GPUs)
    /// None = not yet detected
    pub devices: Option<DeviceCapabilities>,
    
    /// When this hive was added to catalog
    pub created_at_ms: i64,
    
    /// Last time configuration was updated
    pub updated_at_ms: i64,
}
