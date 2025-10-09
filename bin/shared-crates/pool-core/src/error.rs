//! Error types for pool-core
//!
//! Created by: TEAM-022

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PoolError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already exists: {0}")]
    ModelAlreadyExists(String),

    #[error("Worker not found: {0}")]
    WorkerNotFound(String),

    #[error("Invalid backend: {0}")]
    InvalidBackend(String),

    #[error("Catalog not found at: {0}")]
    CatalogNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, PoolError>;
