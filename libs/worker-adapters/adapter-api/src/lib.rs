//! Adapter API — shared trait and types for engine adapters.

use contracts_api_types as api;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealth {
    pub live: bool,
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerProps {
    pub slots_total: Option<u32>,
    pub slots_free: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    pub kind: String, // "started" | "token" | "metrics" | "end" | "error"
    pub data: serde_json::Value,
}

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("deadline unmet")]
    DeadlineUnmet,
    #[error("pool unavailable")]
    PoolUnavailable,
    #[error("decode timeout")]
    DecodeTimeout,
    #[error("worker reset")]
    WorkerReset,
    #[error("internal: {0}")]
    Internal(String),
    #[error("adapter error: {0}")]
    Adapter(String),
}

pub type TokenStream = BoxStream<'static, core::result::Result<TokenEvent, WorkerError>>;

pub trait WorkerAdapter: Send + Sync {
    /// Health includes Ready state; dispatch must target Ready replicas only.
    fn health(&self) -> core::result::Result<WorkerHealth, WorkerError>;
    fn props(&self) -> core::result::Result<WorkerProps, WorkerError>;
    /// Admission enforces bounded queues and full policies upstream.
    fn submit(&self, req: api::TaskRequest) -> core::result::Result<TokenStream, WorkerError>;
    fn cancel(&self, task_id: &str) -> core::result::Result<(), WorkerError>;
    /// Engine version captured for replica set determinism pinning.
    fn engine_version(&self) -> core::result::Result<String, WorkerError>;
}
