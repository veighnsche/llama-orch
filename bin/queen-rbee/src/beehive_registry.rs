// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - SQLite registry implementation

//! rbee-hive Registry Module
//!
//! Created by: TEAM-043
//!
//! Manages persistent registry of rbee-hive nodes with SSH connection details.
//! Stores in SQLite at ~/.rbee/beehives.db

use anyhow::{Context, Result};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
    pub last_connected_unix: Option<i64>,
    pub status: String,
    // TEAM-052: Backend capabilities
    pub backends: Option<String>, // JSON array: ["cuda", "metal", "cpu"]
    pub devices: Option<String>,  // JSON object: {"cuda": 2, "metal": 1, "cpu": 1}
}

pub struct BeehiveRegistry {
    db_path: PathBuf,
    conn: tokio::sync::Mutex<rusqlite::Connection>,
}

impl BeehiveRegistry {
    /// Create or open the beehive registry database
    pub async fn new(db_path: Option<PathBuf>) -> Result<Self> {
        let db_path = db_path.unwrap_or_else(|| {
            let home = dirs::home_dir().expect("Could not determine home directory");
            home.join(".rbee").join("beehives.db")
        });

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Open database (creates if not exists)
        let conn = rusqlite::Connection::open(&db_path).context("Failed to open beehives.db")?;

        // Create table if not exists
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS beehives (
                node_name TEXT PRIMARY KEY,
                ssh_host TEXT NOT NULL,
                ssh_port INTEGER NOT NULL DEFAULT 22,
                ssh_user TEXT NOT NULL,
                ssh_key_path TEXT,
                git_repo_url TEXT NOT NULL,
                git_branch TEXT NOT NULL,
                install_path TEXT NOT NULL,
                last_connected_unix INTEGER,
                status TEXT NOT NULL DEFAULT 'unknown',
                backends TEXT,
                devices TEXT
            )
            "#,
            [],
        )?;

        Ok(Self { db_path, conn: tokio::sync::Mutex::new(conn) })
    }

    /// Add or update a node in the registry
    pub async fn add_node(&self, node: BeehiveNode) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            r#"
            INSERT OR REPLACE INTO beehives (
                node_name, ssh_host, ssh_port, ssh_user, ssh_key_path,
                git_repo_url, git_branch, install_path, last_connected_unix, status,
                backends, devices
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            rusqlite::params![
                node.node_name,
                node.ssh_host,
                node.ssh_port,
                node.ssh_user,
                node.ssh_key_path,
                node.git_repo_url,
                node.git_branch,
                node.install_path,
                node.last_connected_unix,
                node.status,
                node.backends,
                node.devices,
            ],
        )?;
        Ok(())
    }

    /// Get a node by name
    pub async fn get_node(&self, node_name: &str) -> Result<Option<BeehiveNode>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT node_name, ssh_host, ssh_port, ssh_user, ssh_key_path, \
             git_repo_url, git_branch, install_path, last_connected_unix, status, \
             backends, devices \
             FROM beehives WHERE node_name = ?1",
        )?;

        let node = stmt
            .query_row([node_name], |row| {
                Ok(BeehiveNode {
                    node_name: row.get(0)?,
                    ssh_host: row.get(1)?,
                    ssh_port: row.get(2)?,
                    ssh_user: row.get(3)?,
                    ssh_key_path: row.get(4)?,
                    git_repo_url: row.get(5)?,
                    git_branch: row.get(6)?,
                    install_path: row.get(7)?,
                    last_connected_unix: row.get(8)?,
                    status: row.get(9)?,
                    backends: row.get(10)?,
                    devices: row.get(11)?,
                })
            })
            .optional()?;

        Ok(node)
    }

    /// List all nodes
    pub async fn list_nodes(&self) -> Result<Vec<BeehiveNode>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT node_name, ssh_host, ssh_port, ssh_user, ssh_key_path, \
             git_repo_url, git_branch, install_path, last_connected_unix, status, \
             backends, devices \
             FROM beehives ORDER BY node_name",
        )?;

        let nodes = stmt
            .query_map([], |row| {
                Ok(BeehiveNode {
                    node_name: row.get(0)?,
                    ssh_host: row.get(1)?,
                    ssh_port: row.get(2)?,
                    ssh_user: row.get(3)?,
                    ssh_key_path: row.get(4)?,
                    git_repo_url: row.get(5)?,
                    git_branch: row.get(6)?,
                    install_path: row.get(7)?,
                    last_connected_unix: row.get(8)?,
                    status: row.get(9)?,
                    backends: row.get(10)?,
                    devices: row.get(11)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(nodes)
    }

    /// Remove a node from the registry
    pub async fn remove_node(&self, node_name: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        let rows_affected =
            conn.execute("DELETE FROM beehives WHERE node_name = ?1", [node_name])?;
        Ok(rows_affected > 0)
    }

    /// Update node status
    pub async fn update_status(
        &self,
        node_name: &str,
        status: &str,
        last_connected: Option<i64>,
    ) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE beehives SET status = ?1, last_connected_unix = ?2 WHERE node_name = ?3",
            rusqlite::params![status, last_connected, node_name],
        )?;
        Ok(())
    }

    /// Clear all nodes (for testing)
    pub async fn clear(&self) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM beehives", [])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_crud() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_beehives.db");

        let registry = BeehiveRegistry::new(Some(db_path)).await.unwrap();

        // Add node
        let node = BeehiveNode {
            node_name: "workstation".to_string(),
            ssh_host: "workstation.home.arpa".to_string(),
            ssh_port: 22,
            ssh_user: "vince".to_string(),
            ssh_key_path: Some("/home/vince/.ssh/id_ed25519".to_string()),
            git_repo_url: "https://github.com/user/llama-orch.git".to_string(),
            git_branch: "main".to_string(),
            install_path: "/home/vince/rbee".to_string(),
            last_connected_unix: Some(1728508603),
            status: "reachable".to_string(),
            backends: Some(r#"["cuda","cpu"]"#.to_string()),
            devices: Some(r#"{"cuda":2,"cpu":1}"#.to_string()),
        };

        registry.add_node(node.clone()).await.unwrap();

        // Get node
        let retrieved = registry.get_node("workstation").await.unwrap().unwrap();
        assert_eq!(retrieved.node_name, "workstation");
        assert_eq!(retrieved.ssh_host, "workstation.home.arpa");

        // List nodes
        let nodes = registry.list_nodes().await.unwrap();
        assert_eq!(nodes.len(), 1);

        // Update status
        registry.update_status("workstation", "offline", None).await.unwrap();
        let updated = registry.get_node("workstation").await.unwrap().unwrap();
        assert_eq!(updated.status, "offline");

        // Remove node
        let removed = registry.remove_node("workstation").await.unwrap();
        assert!(removed);
        let not_found = registry.get_node("workstation").await.unwrap();
        assert!(not_found.is_none());
    }
}
