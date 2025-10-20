//! Hive catalog for queen-rbee
//!
//! Created by: TEAM-156
//!
//! This crate provides SQLite-based persistent storage for hive information,
//! including capabilities, devices, and status.
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

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use sqlx::Row;
use std::path::Path;
use std::str::FromStr;

/// Actor constant for narration
pub const ACTOR_HIVE_CATALOG: &str = "👑 queen-rbee / ⚙️ hive-catalog";

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
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// Hive catalog - SQLite-based persistent storage
///
/// TEAM-156: Provides persistent storage for registered hives
pub struct HiveCatalog {
    pool: SqlitePool,
}

impl HiveCatalog {
    /// Create a new hive catalog
    ///
    /// TEAM-156: Initializes SQLite database and creates schema if needed
    pub async fn new(db_path: &Path) -> Result<Self> {
        let db_url = format!("sqlite:{}", db_path.display());
        
        let options = SqliteConnectOptions::from_str(&db_url)?
            .create_if_missing(true);
        
        let pool = SqlitePool::connect_with(options)
            .await
            .context("Failed to connect to SQLite database")?;
        
        // TEAM-156: Create schema if not exists
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS hives (
                id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                ssh_host TEXT,
                ssh_port INTEGER,
                ssh_user TEXT,
                status TEXT NOT NULL,
                last_heartbeat_ms INTEGER,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            )
            "#,
        )
        .execute(&pool)
        .await
        .context("Failed to create hives table")?;
        
        Ok(Self { pool })
    }

    /// List all hives in the catalog
    ///
    /// TEAM-156: Returns all registered hives
    pub async fn list_hives(&self) -> Result<Vec<HiveRecord>> {
        let rows = sqlx::query("SELECT * FROM hives")
            .fetch_all(&self.pool)
            .await
            .context("Failed to list hives")?;
        
        let mut hives = Vec::new();
        for row in rows {
            let status_str: String = row.try_get("status")?;
            let status = HiveStatus::from_str(&status_str)?;
            
            hives.push(HiveRecord {
                id: row.try_get("id")?,
                host: row.try_get("host")?,
                port: row.try_get::<i64, _>("port")? as u16,
                ssh_host: row.try_get("ssh_host")?,
                ssh_port: row.try_get::<Option<i64>, _>("ssh_port")?.map(|p| p as u16),
                ssh_user: row.try_get("ssh_user")?,
                status,
                last_heartbeat_ms: row.try_get("last_heartbeat_ms")?,
                created_at_ms: row.try_get("created_at_ms")?,
                updated_at_ms: row.try_get("updated_at_ms")?,
            });
        }
        
        Ok(hives)
    }

    /// Get a specific hive by ID
    ///
    /// TEAM-156: Returns hive if found, None otherwise
    pub async fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>> {
        let row = sqlx::query("SELECT * FROM hives WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .context("Failed to get hive")?;
        
        if let Some(row) = row {
            let status_str: String = row.try_get("status")?;
            let status = HiveStatus::from_str(&status_str)?;
            
            Ok(Some(HiveRecord {
                id: row.try_get("id")?,
                host: row.try_get("host")?,
                port: row.try_get::<i64, _>("port")? as u16,
                ssh_host: row.try_get("ssh_host")?,
                ssh_port: row.try_get::<Option<i64>, _>("ssh_port")?.map(|p| p as u16),
                ssh_user: row.try_get("ssh_user")?,
                status,
                last_heartbeat_ms: row.try_get("last_heartbeat_ms")?,
                created_at_ms: row.try_get("created_at_ms")?,
                updated_at_ms: row.try_get("updated_at_ms")?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Add a new hive to the catalog
    ///
    /// TEAM-156: Inserts hive record into database
    pub async fn add_hive(&self, hive: HiveRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO hives (
                id, host, port, ssh_host, ssh_port, ssh_user,
                status, last_heartbeat_ms, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        .bind(hive.created_at_ms)
        .bind(hive.updated_at_ms)
        .execute(&self.pool)
        .await
        .context("Failed to add hive")?;
        
        Ok(())
    }

    /// Update hive status
    ///
    /// TEAM-156: Updates status and updated_at timestamp
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
}

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
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };
        
        catalog.add_hive(hive).await.unwrap();
        
        let heartbeat_time = chrono::Utc::now().timestamp_millis();
        catalog.update_heartbeat("heartbeat-test", heartbeat_time).await.unwrap();
        
        let updated = catalog.get_hive("heartbeat-test").await.unwrap().unwrap();
        assert_eq!(updated.last_heartbeat_ms, Some(heartbeat_time));
    }
}
