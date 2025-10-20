//! Hive catalog types
//!
//! Created by: TEAM-156
//! Refactored by: TEAM-158

use crate::device_types::DeviceCapabilities;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Hive status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HiveStatus {
    Unknown,
    Online,
    Offline,
}

impl std::fmt::Display for HiveStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HiveStatus::Unknown => write!(f, "unknown"),
            HiveStatus::Online => write!(f, "online"),
            HiveStatus::Offline => write!(f, "offline"),
        }
    }
}

impl FromStr for HiveStatus {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "unknown" => Ok(HiveStatus::Unknown),
            "online" => Ok(HiveStatus::Online),
            "offline" => Ok(HiveStatus::Offline),
            _ => Err(anyhow::anyhow!("Invalid hive status: {}", s)),
        }
    }
}

/// Hive record stored in catalog
///
/// TEAM-158: Added devices field for device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveRecord {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub ssh_host: Option<String>,
    pub ssh_port: Option<u16>,
    pub ssh_user: Option<String>,
    pub status: HiveStatus,
    pub last_heartbeat_ms: Option<i64>,
    /// Device capabilities (CPU, CUDA, Metal)
    /// TEAM-158: None means devices not yet detected
    pub devices: Option<DeviceCapabilities>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}
