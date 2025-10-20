//! SQLite row mapping utilities
//!
//! Created by: TEAM-158
//! Purpose: DRY - Don't repeat row mapping logic

use crate::device_types::DeviceCapabilities;
use crate::types::{HiveRecord, HiveStatus};
use anyhow::Result;
use sqlx::Row;
use std::str::FromStr;

/// Map a SQLite row to a HiveRecord
///
/// TEAM-158: Centralized row mapping to avoid duplication
pub fn map_row_to_hive(row: &sqlx::sqlite::SqliteRow) -> Result<HiveRecord> {
    let status_str: String = row.try_get("status")?;
    let status = HiveStatus::from_str(&status_str)?;

    // TEAM-158: Parse devices JSON if present
    let devices = row
        .try_get::<Option<String>, _>("devices_json")?
        .and_then(|json| DeviceCapabilities::from_json(&json).ok());

    Ok(HiveRecord {
        id: row.try_get("id")?,
        host: row.try_get("host")?,
        port: row.try_get::<i64, _>("port")? as u16,
        ssh_host: row.try_get("ssh_host")?,
        ssh_port: row.try_get::<Option<i64>, _>("ssh_port")?.map(|p| p as u16),
        ssh_user: row.try_get("ssh_user")?,
        status,
        last_heartbeat_ms: row.try_get("last_heartbeat_ms")?,
        devices,
        created_at_ms: row.try_get("created_at_ms")?,
        updated_at_ms: row.try_get("updated_at_ms")?,
    })
}
