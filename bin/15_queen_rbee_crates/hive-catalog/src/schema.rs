//! Database schema management
//!
//! Created by: TEAM-158
//! Purpose: Separate schema creation from business logic
//! TEAM-186: Removed status and last_heartbeat_ms columns

use anyhow::{Context, Result};
use sqlx::sqlite::SqlitePool;

/// SQL schema for hives table
///
/// CONFIGURATION ONLY - No runtime/heartbeat data!
/// Runtime data (status, heartbeat, workers) lives in hive-registry (RAM)
/// TEAM-186: Removed status and last_heartbeat_ms columns from schema
const HIVES_TABLE_SCHEMA: &str = r#"
    CREATE TABLE IF NOT EXISTS hives (
        id TEXT PRIMARY KEY,
        host TEXT NOT NULL,
        port INTEGER NOT NULL,
        ssh_host TEXT,
        ssh_port INTEGER,
        ssh_user TEXT,
        devices_json TEXT,
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL
    )
"#;

/// Initialize database schema
///
/// TEAM-158: Creates all required tables if they don't exist
pub async fn initialize_schema(pool: &SqlitePool) -> Result<()> {
    sqlx::query(HIVES_TABLE_SCHEMA).execute(pool).await.context("Failed to create hives table")?;

    Ok(())
}
