//! Hive catalog for queen-rbee
//!
//! **Category:** Data Management
//! **Pattern:** CRUD Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! TEAM-156: Created by TEAM-156
//! TEAM-158: Refactored to CRUD pattern
//!
//! This crate provides persistent storage for hive metadata using SQLite.
//! It follows a standard CRUD (Create, Read, Update, Delete) pattern for
//! maintainability and consistency.
//!
//! # Interface
//!
//! ## CRUD Operations
//! ```rust
//! // CREATE
//! pub async fn add_hive(&self, hive: HiveRecord) -> Result<()>
//!
//! // READ
//! pub async fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>>
//! pub async fn list_hives(&self) -> Result<Vec<HiveRecord>>
//!
//! // UPDATE
//! pub async fn update_hive(&self, hive: HiveRecord) -> Result<()>
//! pub async fn update_hive_status(&self, id: &str, status: HiveStatus) -> Result<>()
//! pub async fn update_heartbeat(&self, id: &str, timestamp_ms: i64) -> Result<>()
//! pub async fn update_devices(&self, id: &str, devices: DeviceCapabilities) -> Result<>()
//!
//! // DELETE
//! pub async fn remove_hive(&self, id: &str) -> Result<()>
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use queen_rbee_hive_catalog::{HiveCatalog, HiveRecord, HiveStatus};
//! use std::path::Path;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let catalog = HiveCatalog::new(Path::new("hives.db")).await.unwrap();
//! let hives = catalog.list_hives().await.unwrap();
//! # }
//! ```
//!
//! # Module Structure
//!
//! - `types` - Data types (HiveStatus, HiveRecord)
//! - `schema` - Database schema management
//! - `row_mapper` - SQLite row to struct mapping
//! - `catalog` - Main HiveCatalog implementation

// TEAM-158: Modular structure for maintainability
mod catalog;
mod device_types;
mod heartbeat_traits; // TEAM-159: HiveCatalog trait impl for rbee-heartbeat
mod row_mapper;
mod schema;
mod types;

// Re-export public API
pub use catalog::HiveCatalog;
pub use device_types::{CpuDevice, DeviceBackend, DeviceCapabilities, GpuDevice};
pub use types::{HiveRecord, HiveStatus};

/// Actor constant for narration
pub const ACTOR_HIVE_CATALOG: &str = "👑 queen-rbee / ⚙️ hive-catalog";

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_create_catalog() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = HiveCatalog::new(&db_path).await.unwrap();
        let hives = catalog.list_hives().await.unwrap();

        assert_eq!(hives.len(), 0);
    }

    #[tokio::test]
    async fn test_add_and_list_hives() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = HiveCatalog::new(&db_path).await.unwrap();

        let now_ms = chrono::Utc::now().timestamp_millis();
        let hive = HiveRecord {
            id: "localhost".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8600,
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            status: HiveStatus::Unknown,
            last_heartbeat_ms: None,
            devices: None, // TEAM-158: Devices not yet detected
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        catalog.add_hive(hive).await.unwrap();

        let hives = catalog.list_hives().await.unwrap();
        assert_eq!(hives.len(), 1);
        assert_eq!(hives[0].id, "localhost");
        assert_eq!(hives[0].host, "127.0.0.1");
        assert_eq!(hives[0].port, 8600);
    }

    #[tokio::test]
    async fn test_get_hive() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = HiveCatalog::new(&db_path).await.unwrap();

        let now_ms = chrono::Utc::now().timestamp_millis();
        let hive = HiveRecord {
            id: "test-hive".to_string(),
            host: "192.168.1.100".to_string(),
            port: 8600,
            ssh_host: Some("192.168.1.100".to_string()),
            ssh_port: Some(22),
            ssh_user: Some("user".to_string()),
            status: HiveStatus::Online,
            last_heartbeat_ms: Some(now_ms),
            devices: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        catalog.add_hive(hive).await.unwrap();

        let retrieved = catalog.get_hive("test-hive").await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, "test-hive");
        assert_eq!(retrieved.status, HiveStatus::Online);

        let not_found = catalog.get_hive("nonexistent").await.unwrap();
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_update_status() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = HiveCatalog::new(&db_path).await.unwrap();

        let now_ms = chrono::Utc::now().timestamp_millis();
        let hive = HiveRecord {
            id: "status-test".to_string(),
            host: "localhost".to_string(),
            port: 8600,
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            status: HiveStatus::Unknown,
            last_heartbeat_ms: None,
            devices: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        catalog.add_hive(hive).await.unwrap();
        catalog.update_hive_status("status-test", HiveStatus::Online).await.unwrap();

        let updated = catalog.get_hive("status-test").await.unwrap().unwrap();
        assert_eq!(updated.status, HiveStatus::Online);
    }

    #[tokio::test]
    async fn test_update_heartbeat() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = HiveCatalog::new(&db_path).await.unwrap();

        let now_ms = chrono::Utc::now().timestamp_millis();
        let hive = HiveRecord {
            id: "heartbeat-test".to_string(),
            host: "localhost".to_string(),
            port: 8600,
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            status: HiveStatus::Unknown,
            last_heartbeat_ms: None,
            devices: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        catalog.add_hive(hive).await.unwrap();

        let heartbeat_time = chrono::Utc::now().timestamp_millis();
        catalog.update_heartbeat("heartbeat-test", heartbeat_time).await.unwrap();

        let updated = catalog.get_hive("heartbeat-test").await.unwrap().unwrap();
        assert_eq!(updated.last_heartbeat_ms, Some(heartbeat_time));
    }

    #[tokio::test]
    async fn test_update_hive() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = HiveCatalog::new(&db_path).await.unwrap();

        let now_ms = chrono::Utc::now().timestamp_millis();
        let mut hive = HiveRecord {
            id: "update-test".to_string(),
            host: "localhost".to_string(),
            port: 8600,
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            status: HiveStatus::Unknown,
            last_heartbeat_ms: None,
            devices: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        catalog.add_hive(hive.clone()).await.unwrap();

        // Update the record
        hive.host = "192.168.1.100".to_string();
        hive.port = 9000;
        hive.status = HiveStatus::Online;
        catalog.update_hive(hive).await.unwrap();

        let updated = catalog.get_hive("update-test").await.unwrap().unwrap();
        assert_eq!(updated.host, "192.168.1.100");
        assert_eq!(updated.port, 9000);
        assert_eq!(updated.status, HiveStatus::Online);
    }

    #[tokio::test]
    async fn test_remove_hive() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = HiveCatalog::new(&db_path).await.unwrap();

        let now_ms = chrono::Utc::now().timestamp_millis();
        let hive = HiveRecord {
            id: "delete-test".to_string(),
            host: "localhost".to_string(),
            port: 8600,
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            status: HiveStatus::Unknown,
            last_heartbeat_ms: None,
            devices: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        catalog.add_hive(hive).await.unwrap();

        // Verify it exists
        let exists = catalog.get_hive("delete-test").await.unwrap();
        assert!(exists.is_some());

        // Remove it
        catalog.remove_hive("delete-test").await.unwrap();

        // Verify it's gone
        let not_found = catalog.get_hive("delete-test").await.unwrap();
        assert!(not_found.is_none());
    }
}
