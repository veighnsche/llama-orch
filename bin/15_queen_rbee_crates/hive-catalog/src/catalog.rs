//! Hive catalog implementation
//!
//! Created by: TEAM-156
//! Refactored by: TEAM-158 - CRUD pattern
//! TEAM-186: Removed all runtime/heartbeat functions (status, heartbeat updates, status queries)
//!
//! # CRUD Operations
//!
//! CONFIGURATION ONLY - No runtime/heartbeat data!
//! Runtime data (status, heartbeat, workers) lives in hive-registry (RAM)
//!
//! - **Create:** `add_hive()`
//! - **Read:** `get_hive()`, `list_hives()`
//! - **Update:** `update_hive()`, `update_devices()`
//! - **Delete:** `remove_hive()`
//!
//! TEAM-186: Removed functions:
//! - update_hive_status(), update_heartbeat()
//! - find_hives_by_status(), find_online_hives(), find_offline_hives()
//! - find_stale_hives(), count_hives(), count_by_status()

use crate::device_types::DeviceCapabilities;
use crate::row_mapper::map_row_to_hive;
use crate::schema::initialize_schema;
use crate::types::HiveRecord;
use anyhow::{Context, Result};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::Path;
use std::str::FromStr;

/// Hive catalog - SQLite-based persistent storage
///
/// CONFIGURATION ONLY - No runtime/heartbeat data!
/// Stores: host, port, SSH credentials, device capabilities
/// Does NOT store: status, heartbeat, workers (those are in hive-registry)
/// TEAM-186: Enforced configuration-only architecture
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
    /// Stores CONFIGURATION only (host, port, SSH, devices)
    /// Does NOT store runtime data (status, heartbeat)
    pub async fn add_hive(&self, hive: HiveRecord) -> Result<()> {
        // TEAM-158: Serialize devices to JSON
        let devices_json = hive.devices.as_ref().and_then(|d| d.to_json().ok());

        sqlx::query(
            r#"
            INSERT INTO hives (
                id, host, port, ssh_host, ssh_port, ssh_user,
                devices_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&hive.id)
        .bind(&hive.host)
        .bind(hive.port as i64)
        .bind(&hive.ssh_host)
        .bind(hive.ssh_port.map(|p| p as i64))
        .bind(&hive.ssh_user)
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

    /// Update hive configuration
    ///
    /// TEAM-158: CRUD - Update operation (full record with devices)
    /// Updates CONFIGURATION only (host, port, SSH)
    /// Does NOT update runtime data (use hive-registry for that)
    pub async fn update_hive(&self, hive: HiveRecord) -> Result<()> {
        let now_ms = chrono::Utc::now().timestamp_millis();
        // TEAM-158: Serialize devices to JSON
        let devices_json = hive.devices.as_ref().and_then(|d| d.to_json().ok());

        sqlx::query(
            r#"
            UPDATE hives
            SET host = ?, port = ?, ssh_host = ?, ssh_port = ?, ssh_user = ?,
                devices_json = ?, updated_at_ms = ?
            WHERE id = ?
            "#,
        )
        .bind(&hive.host)
        .bind(hive.port as i64)
        .bind(&hive.ssh_host)
        .bind(hive.ssh_port.map(|p| p as i64))
        .bind(&hive.ssh_user)
        .bind(devices_json)
        .bind(now_ms)
        .bind(&hive.id)
        .execute(&self.pool)
        .await
        .context("Failed to update hive")?;

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

    // ========================================================================
    // QUERY OPERATIONS (Configuration only)
    // ========================================================================
    // TEAM-186: Removed all status/heartbeat query functions
    // TEAM-186: Kept only device-related configuration queries

    /// Find hives with devices detected
    ///
    /// Returns hives that have device capabilities information.
    /// Useful for knowing which hives have been fully configured.
    pub async fn find_hives_with_devices(&self) -> Result<Vec<HiveRecord>> {
        let rows = sqlx::query("SELECT * FROM hives WHERE devices_json IS NOT NULL")
            .fetch_all(&self.pool)
            .await
            .context("Failed to find hives with devices")?;

        rows.iter().map(map_row_to_hive).collect()
    }

    /// Find hives without devices detected
    ///
    /// Returns hives that need device detection.
    /// Useful for triggering device detection on new hives.
    pub async fn find_hives_without_devices(&self) -> Result<Vec<HiveRecord>> {
        let rows = sqlx::query("SELECT * FROM hives WHERE devices_json IS NULL")
            .fetch_all(&self.pool)
            .await
            .context("Failed to find hives without devices")?;

        rows.iter().map(map_row_to_hive).collect()
    }

    /// Check if hive exists in catalog
    ///
    /// Useful for validation before operations.
    pub async fn hive_exists(&self, id: &str) -> Result<bool> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM hives WHERE id = ?")
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .context("Failed to check if hive exists")?;

        use sqlx::Row;
        let count: i64 = row.try_get("count")?;
        Ok(count > 0)
    }
}
