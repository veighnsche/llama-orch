// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-267: Implemented model catalog types and filesystem-based storage

#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-model-catalog
//!
//! Model catalog management for tracking available models.
//!
//! This crate provides a filesystem-based catalog for managing LLM models.
//! Models are stored in platform-specific cache directories with YAML metadata.
//!
//! # Architecture
//!
//! - **Storage:** `~/.cache/rbee/models/` (Linux/Mac) or `%LOCALAPPDATA%\rbee\models\` (Windows)
//! - **Structure:** Each model has a directory with `metadata.yaml`
//! - **Catalog:** Scans filesystem on demand (no in-memory cache)
//!
//! # Example
//!
//! ```no_run
//! use rbee_hive_model_catalog::{ModelCatalog, ModelEntry};
//! use std::path::PathBuf;
//!
//! # fn main() -> anyhow::Result<()> {
//! let catalog = ModelCatalog::new()?;
//!
//! // Add a model
//! let model = ModelEntry::new(
//!     "meta-llama/Llama-2-7b-chat-hf".to_string(),
//!     "Llama 2 7B Chat".to_string(),
//!     PathBuf::from("/path/to/model"),
//!     7_000_000_000,
//! );
//! catalog.add(model)?;
//!
//! // List all models
//! let models = catalog.list();
//! println!("Found {} models", models.len());
//!
//! // Get a specific model
//! let model = catalog.get("meta-llama/Llama-2-7b-chat-hf")?;
//! println!("Model: {}", model.name);
//! # Ok(())
//! # }
//! ```

/// Model catalog implementation
pub mod catalog;
/// Model catalog types
pub mod types;

// Re-export main types for convenience
pub use catalog::ModelCatalog;
pub use types::{ModelEntry, ModelMetadata, ModelStatus};
