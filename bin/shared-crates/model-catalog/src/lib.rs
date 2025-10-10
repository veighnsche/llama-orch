//! Model catalog - SQLite-backed model tracking
//!
//! Per test-001-mvp.md Phase 3: Model Provisioning
//! Tracks downloaded models with their local paths
//!
//! Created by: TEAM-029

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::{Connection, SqliteConnection};

/// Model catalog - tracks downloaded models
pub struct ModelCatalog {
    db_path: String,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model reference (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    pub reference: String,
    /// Provider (e.g., "hf" for HuggingFace)
    pub provider: String,
    /// Local file path
    pub local_path: String,
    /// File size in bytes
    pub size_bytes: i64,
    /// Unix timestamp when downloaded
    pub downloaded_at: i64,
}

impl ModelCatalog {
    /// Create new model catalog
    ///
    /// # Arguments
    /// * `db_path` - Path to SQLite database file
    pub fn new(db_path: String) -> Self {
        Self { db_path }
    }

    /// Get connection string with sqlite:// prefix
    /// TEAM-029: Same pattern as worker-registry
    fn connection_string(&self) -> String {
        if self.db_path.starts_with("sqlite://") 
            || self.db_path.starts_with(":memory:") 
            || self.db_path.starts_with("file:") 
        {
            self.db_path.clone()
        } else {
            format!("sqlite://{}?mode=rwc", self.db_path)
        }
    }

    /// Initialize database schema
    ///
    /// Creates models table if it doesn't exist
    pub async fn init(&self) -> Result<()> {
        // TEAM-029: Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&self.db_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut conn = SqliteConnection::connect(&self.connection_string()).await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS models (
                reference TEXT NOT NULL,
                provider TEXT NOT NULL,
                local_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                downloaded_at INTEGER NOT NULL,
                PRIMARY KEY (reference, provider)
            )
            "#,
        )
        .execute(&mut conn)
        .await?;

        Ok(())
    }

    /// Find model by reference and provider
    ///
    /// Per test-001-mvp.md lines 69-73
    ///
    /// # Arguments
    /// * `reference` - Model reference (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    /// * `provider` - Provider (e.g., "hf")
    ///
    /// # Returns
    /// Model info if found
    pub async fn find_model(&self, reference: &str, provider: &str) -> Result<Option<ModelInfo>> {
        let mut conn = SqliteConnection::connect(&self.connection_string()).await?;

        let row = sqlx::query_as::<_, (String, String, String, i64, i64)>(
            r#"
            SELECT reference, provider, local_path, size_bytes, downloaded_at
            FROM models 
            WHERE reference = ? AND provider = ?
            "#,
        )
        .bind(reference)
        .bind(provider)
        .fetch_optional(&mut conn)
        .await?;

        Ok(row.map(|(reference, provider, local_path, size_bytes, downloaded_at)| ModelInfo {
            reference,
            provider,
            local_path,
            size_bytes,
            downloaded_at,
        }))
    }

    /// Register model
    ///
    /// Per test-001-mvp.md Phase 3: After download completes
    ///
    /// # Arguments
    /// * `model` - Model info to register
    pub async fn register_model(&self, model: &ModelInfo) -> Result<()> {
        let mut conn = SqliteConnection::connect(&self.connection_string()).await?;

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO models 
            (reference, provider, local_path, size_bytes, downloaded_at)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(&model.reference)
        .bind(&model.provider)
        .bind(&model.local_path)
        .bind(model.size_bytes)
        .bind(model.downloaded_at)
        .execute(&mut conn)
        .await?;

        Ok(())
    }

    /// Remove model from catalog
    ///
    /// # Arguments
    /// * `reference` - Model reference
    /// * `provider` - Provider
    pub async fn remove_model(&self, reference: &str, provider: &str) -> Result<()> {
        let mut conn = SqliteConnection::connect(&self.connection_string()).await?;

        sqlx::query(
            r#"
            DELETE FROM models WHERE reference = ? AND provider = ?
            "#,
        )
        .bind(reference)
        .bind(provider)
        .execute(&mut conn)
        .await?;

        Ok(())
    }

    /// List all models
    ///
    /// # Returns
    /// All models in the catalog
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let mut conn = SqliteConnection::connect(&self.connection_string()).await?;

        let rows = sqlx::query_as::<_, (String, String, String, i64, i64)>(
            r#"
            SELECT reference, provider, local_path, size_bytes, downloaded_at
            FROM models
            ORDER BY downloaded_at DESC
            "#,
        )
        .fetch_all(&mut conn)
        .await?;

        Ok(rows
            .into_iter()
            .map(|(reference, provider, local_path, size_bytes, downloaded_at)| ModelInfo {
                reference,
                provider,
                local_path,
                size_bytes,
                downloaded_at,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_catalog_init() {
        let catalog = ModelCatalog::new(":memory:".to_string());
        assert!(catalog.init().await.is_ok());
    }

    #[tokio::test]
    async fn test_catalog_register_and_find() {
        // TEAM-033: Use temp file instead of :memory: to avoid SQLite connection isolation
        let temp_file = format!("/tmp/test_catalog_{}.db", Uuid::new_v4());
        let catalog = ModelCatalog::new(temp_file.clone());
        catalog.init().await.unwrap();

        let model = ModelInfo {
            reference: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/tinyllama.gguf".to_string(),
            size_bytes: 669000000,
            downloaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        };

        catalog.register_model(&model).await.unwrap();

        let found = catalog
            .find_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "hf")
            .await
            .unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().local_path, "/models/tinyllama.gguf");
        
        // TEAM-033: Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);
    }

    // TEAM-031: Additional comprehensive tests
    #[tokio::test]
    async fn test_catalog_find_nonexistent() {
        // TEAM-033: Use temp file instead of :memory: to avoid SQLite connection isolation
        let temp_file = format!("/tmp/test_catalog_{}.db", Uuid::new_v4());
        let catalog = ModelCatalog::new(temp_file.clone());
        catalog.init().await.unwrap();

        let found = catalog.find_model("nonexistent/model", "hf").await.unwrap();
        assert!(found.is_none());
        
        // TEAM-033: Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_catalog_remove_model() {
        // TEAM-033: Use temp file instead of :memory: to avoid SQLite connection isolation
        let temp_file = format!("/tmp/test_catalog_{}.db", Uuid::new_v4());
        let catalog = ModelCatalog::new(temp_file.clone());
        catalog.init().await.unwrap();

        let model = ModelInfo {
            reference: "test/model".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/test.gguf".to_string(),
            size_bytes: 1000,
            downloaded_at: 1234567890,
        };

        catalog.register_model(&model).await.unwrap();
        assert!(catalog.find_model("test/model", "hf").await.unwrap().is_some());

        catalog.remove_model("test/model", "hf").await.unwrap();
        assert!(catalog.find_model("test/model", "hf").await.unwrap().is_none());
        
        // TEAM-033: Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_catalog_list_models() {
        // TEAM-033: Use temp file instead of :memory: to avoid SQLite connection isolation
        let temp_file = format!("/tmp/test_catalog_{}.db", Uuid::new_v4());
        let catalog = ModelCatalog::new(temp_file.clone());
        catalog.init().await.unwrap();

        let model1 = ModelInfo {
            reference: "model1".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/model1.gguf".to_string(),
            size_bytes: 1000,
            downloaded_at: 1234567890,
        };

        let model2 = ModelInfo {
            reference: "model2".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/model2.gguf".to_string(),
            size_bytes: 2000,
            downloaded_at: 1234567891,
        };

        catalog.register_model(&model1).await.unwrap();
        catalog.register_model(&model2).await.unwrap();

        let models = catalog.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        
        // TEAM-033: Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_catalog_list_models_empty() {
        // TEAM-033: Use temp file instead of :memory: to avoid SQLite connection isolation
        let temp_file = format!("/tmp/test_catalog_{}.db", Uuid::new_v4());
        let catalog = ModelCatalog::new(temp_file.clone());
        catalog.init().await.unwrap();

        let models = catalog.list_models().await.unwrap();
        assert_eq!(models.len(), 0);
        
        // TEAM-033: Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_catalog_replace_model() {
        // TEAM-033: Use temp file instead of :memory: to avoid SQLite connection isolation
        let temp_file = format!("/tmp/test_catalog_{}.db", Uuid::new_v4());
        let catalog = ModelCatalog::new(temp_file.clone());
        catalog.init().await.unwrap();

        let model1 = ModelInfo {
            reference: "test/model".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/old.gguf".to_string(),
            size_bytes: 1000,
            downloaded_at: 1234567890,
        };

        let model2 = ModelInfo {
            reference: "test/model".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/new.gguf".to_string(),
            size_bytes: 2000,
            downloaded_at: 1234567900,
        };

        catalog.register_model(&model1).await.unwrap();
        catalog.register_model(&model2).await.unwrap();

        let found = catalog.find_model("test/model", "hf").await.unwrap().unwrap();
        assert_eq!(found.local_path, "/models/new.gguf");
        assert_eq!(found.size_bytes, 2000);

        let models = catalog.list_models().await.unwrap();
        assert_eq!(models.len(), 1); // Should replace, not duplicate
        
        // TEAM-033: Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_catalog_different_providers() {
        // TEAM-033: Use temp file instead of :memory: to avoid SQLite connection isolation
        let temp_file = format!("/tmp/test_catalog_{}.db", Uuid::new_v4());
        let catalog = ModelCatalog::new(temp_file.clone());
        catalog.init().await.unwrap();

        let model_hf = ModelInfo {
            reference: "test/model".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/hf.gguf".to_string(),
            size_bytes: 1000,
            downloaded_at: 1234567890,
        };

        let model_other = ModelInfo {
            reference: "test/model".to_string(),
            provider: "other".to_string(),
            local_path: "/models/other.gguf".to_string(),
            size_bytes: 2000,
            downloaded_at: 1234567890,
        };

        catalog.register_model(&model_hf).await.unwrap();
        catalog.register_model(&model_other).await.unwrap();

        let found_hf = catalog.find_model("test/model", "hf").await.unwrap().unwrap();
        assert_eq!(found_hf.local_path, "/models/hf.gguf");

        let found_other = catalog.find_model("test/model", "other").await.unwrap().unwrap();
        assert_eq!(found_other.local_path, "/models/other.gguf");

        let models = catalog.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        
        // TEAM-033: Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_model_info_serialization() {
        let model = ModelInfo {
            reference: "test/model".to_string(),
            provider: "hf".to_string(),
            local_path: "/models/test.gguf".to_string(),
            size_bytes: 1000,
            downloaded_at: 1234567890,
        };

        let json = serde_json::to_value(&model).unwrap();
        assert_eq!(json["reference"], "test/model");
        assert_eq!(json["provider"], "hf");
        assert_eq!(json["local_path"], "/models/test.gguf");
        assert_eq!(json["size_bytes"], 1000);
        assert_eq!(json["downloaded_at"], 1234567890);
    }

    #[test]
    fn test_connection_string_memory() {
        let catalog = ModelCatalog::new(":memory:".to_string());
        assert_eq!(catalog.connection_string(), ":memory:");
    }

    #[test]
    fn test_connection_string_file() {
        let catalog = ModelCatalog::new("/tmp/test.db".to_string());
        assert_eq!(catalog.connection_string(), "sqlite:///tmp/test.db?mode=rwc");
    }

    #[test]
    fn test_connection_string_already_prefixed() {
        let catalog = ModelCatalog::new("sqlite:///tmp/test.db".to_string());
        assert_eq!(catalog.connection_string(), "sqlite:///tmp/test.db");
    }
}
