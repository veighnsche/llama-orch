//! Hive catalog implementation
//!
//! Created by: TEAM-156
//! Refactored by: TEAM-158 - CRUD pattern
//!
//! # CRUD Operations
//!
//! This module follows standard CRUD (Create, Read, Update, Delete) pattern:
//!
//! - **Create:** `add_hive()`
//! - **Read:** `get_hive()`, `list_hives()`
//! - **Update:** `update_hive()`, `update_hive_status()`, `update_heartbeat()`
//! - **Delete:** `remove_hive()`

use crate::device_types::DeviceCapabilities;
use crate::row_mapper::map_row_to_hive;
use crate::schema::initialize_schema;
use crate::types::{HiveRecord, HiveStatus};
use anyhow::{Context, Result};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::Path;
use std::str::FromStr;

/// Hive catalog - SQLite-based persistent storage
///
/// TEAM-156: Provides persistent storage for registered hives
/// TEAM-158: Follows CRUD pattern for maintainability
pub struct HiveCatalog {
    pool: SqlitePool,
}

impl HiveCatalog {
    // ========================================================================
    // Initialization
    // ========================================================================

    /// Create a new hive catalog
    ///
    /// TEAM-156: Initializes SQLite database and creates schema if needed
    pub async fn new(db_path: &Path) -> Result<Self> {
        let db_url = format!("sqlite:{}", db_path.display());

        let options = SqliteConnectOptions::from_str(&db_url)?.create_if_missing(true);

        let pool = SqlitePool::connect_with(options)
            .await
            .context("Failed to connect to SQLite database")?;

        // TEAM-158: Schema initialization moved to separate module
        initialize_schema(&pool).await?;

        Ok(Self { pool })
    }

    // ========================================================================
    // CREATE
    // ========================================================================

    /// Add a new hive to the catalog
    ///
    /// TEAM-156: Inserts hive record into database
    /// TEAM-158: CRUD - Create operation (with devices)
    pub async fn add_hive(&self, hive: HiveRecord) -> Result<()> {
        // TEAM-158: Serialize devices to JSON
        let devices_json = hive.devices.as_ref().and_then(|d| d.to_json().ok());

        sqlx::query(
            r#"
            INSERT INTO hives (
                id, host, port, ssh_host, ssh_port, ssh_user,
                status, last_heartbeat_ms, devices_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&hive.id)
        .bind(&hive.host)
        .bind(hive.port as i64)
        .bind(&hive.ssh_host)
        .bind(hive.ssh_port.map(|p| p as i64))
        .bind(&hive.ssh_user)
        .bind(hive.status.to_string())
        .bind(hive.last_heartbeat_ms)
        .bind(devices_json)
        .bind(hive.created_at_ms)
        .bind(hive.updated_at_ms)
        .execute(&self.pool)
        .await
        .context("Failed to add hive")?;

        Ok(())
    }

    // ========================================================================
    // READ
    // ========================================================================

    /// Get a specific hive by ID
    ///
    /// TEAM-156: Returns hive if found, None otherwise
    /// TEAM-158: CRUD - Read operation (single record)
    pub async fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>> {
        let row = sqlx::query("SELECT * FROM hives WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .context("Failed to get hive")?;

        // TEAM-158: Use centralized row mapper
        row.as_ref().map(map_row_to_hive).transpose()
    }

    /// List all hives in the catalog
    ///
    /// TEAM-156: Returns all registered hives
    /// TEAM-158: CRUD - Read operation (all records)
    pub async fn list_hives(&self) -> Result<Vec<HiveRecord>> {
        let rows = sqlx::query("SELECT * FROM hives")
            .fetch_all(&self.pool)
            .await
            .context("Failed to list hives")?;

        // TEAM-158: Use centralized row mapper
        rows.iter().map(map_row_to_hive).collect()
    }

    // ========================================================================
    // UPDATE
    // ========================================================================

    /// Update an entire hive record
    ///
    /// TEAM-158: CRUD - Update operation (full record with devices)
    pub async fn update_hive(&self, hive: HiveRecord) -> Result<()> {
        let now_ms = chrono::Utc::now().timestamp_millis();

        // TEAM-158: Serialize devices to JSON
        let devices_json = hive.devices.as_ref().and_then(|d| d.to_json().ok());

        sqlx::query(
            r#"
            UPDATE hives
            SET host = ?, port = ?, ssh_host = ?, ssh_port = ?, ssh_user = ?,
                status = ?, last_heartbeat_ms = ?, devices_json = ?, updated_at_ms = ?
            WHERE id = ?
            "#,
        )
        .bind(&hive.host)
        .bind(hive.port as i64)
        .bind(&hive.ssh_host)
        .bind(hive.ssh_port.map(|p| p as i64))
        .bind(&hive.ssh_user)
        .bind(hive.status.to_string())
        .bind(hive.last_heartbeat_ms)
        .bind(devices_json)
        .bind(now_ms)
        .bind(&hive.id)
        .execute(&self.pool)
        .await
        .context("Failed to update hive")?;

        Ok(())
    }

    /// Update hive status
    ///
    /// TEAM-156: Updates status and updated_at timestamp
    /// TEAM-158: CRUD - Update operation (partial - status only)
    pub async fn update_hive_status(&self, id: &str, status: HiveStatus) -> Result<()> {
        let now_ms = chrono::Utc::now().timestamp_millis();

        sqlx::query(
            r#"
            UPDATE hives
            SET status = ?, updated_at_ms = ?
            WHERE id = ?
            "#,
        )
        .bind(status.to_string())
        .bind(now_ms)
        .bind(id)
        .execute(&self.pool)
        .await
        .context("Failed to update hive status")?;

        Ok(())
    }

    /// Update hive heartbeat timestamp
    ///
    /// TEAM-156: Records last heartbeat time
    /// TEAM-158: CRUD - Update operation (partial - heartbeat only)
    pub async fn update_heartbeat(&self, id: &str, timestamp_ms: i64) -> Result<()> {
        let now_ms = chrono::Utc::now().timestamp_millis();

        sqlx::query(
            r#"
            UPDATE hives
            SET last_heartbeat_ms = ?, updated_at_ms = ?
            WHERE id = ?
            "#,
        )
        .bind(timestamp_ms)
        .bind(now_ms)
        .bind(id)
        .execute(&self.pool)
        .await
        .context("Failed to update heartbeat")?;

        Ok(())
    }

    /// Update hive device capabilities
    ///
    /// TEAM-158: CRUD - Update operation (partial - devices only)
    /// This is called after device detection completes
    pub async fn update_devices(&self, id: &str, devices: DeviceCapabilities) -> Result<()> {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let devices_json = devices.to_json()?;

        sqlx::query(
            r#"
            UPDATE hives
            SET devices_json = ?, updated_at_ms = ?
            WHERE id = ?
            "#,
        )
        .bind(devices_json)
        .bind(now_ms)
        .bind(id)
        .execute(&self.pool)
        .await
        .context("Failed to update devices")?;

        Ok(())
    }

    // ========================================================================
    // DELETE
    // ========================================================================

    /// Remove a hive from the catalog
    ///
    /// TEAM-158: CRUD - Delete operation
    pub async fn remove_hive(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM hives WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await
            .context("Failed to remove hive")?;

        Ok(())
    }
}
