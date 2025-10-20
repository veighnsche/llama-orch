//! Database schema management
//!
//! Created by: TEAM-158
//! Purpose: Separate schema creation from business logic

use anyhow::{Context, Result};
use sqlx::sqlite::SqlitePool;

/// SQL schema for hives table
///
/// TEAM-158: Added devices_json column for device capabilities
const HIVES_TABLE_SCHEMA: &str = r#"
    CREATE TABLE IF NOT EXISTS hives (
        id TEXT PRIMARY KEY,
        host TEXT NOT NULL,
        port INTEGER NOT NULL,
        ssh_host TEXT,
        ssh_port INTEGER,
        ssh_user TEXT,
        status TEXT NOT NULL,
        last_heartbeat_ms INTEGER,
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
