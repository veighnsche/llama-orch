//! Core orchestrator library (pre-code).
//! Defines the WorkerAdapter trait used by engine adapters. No I/O or logic here.

use contracts_api_types as api;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("adapter error: {0}")]
    Adapter(String),
}

pub type TokenStream = BoxStream<'static, core::result::Result<TokenEvent, WorkerError>>;

pub trait WorkerAdapter: Send + Sync {
    fn health(&self) -> core::result::Result<WorkerHealth, WorkerError>;
    fn props(&self) -> core::result::Result<WorkerProps, WorkerError>;
    fn submit(&self, req: api::TaskRequest) -> core::result::Result<TokenStream, WorkerError>;
    fn cancel(&self, task_id: &str) -> core::result::Result<(), WorkerError>;
    fn engine_version(&self) -> core::result::Result<String, WorkerError>;
}
