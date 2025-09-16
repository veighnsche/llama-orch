//! Core orchestrator library (pre-code).
//! Defines the WorkerAdapter trait used by engine adapters. No I/O or logic here.
//!
//! Traceability (SPEC):
//! - OC-CORE-1001, OC-CORE-1002, OC-CORE-1004 (queue & admission invariants)
//! - OC-CORE-1010, OC-CORE-1011, OC-CORE-1012 (placement & readiness)
//! - OC-CORE-1030 (determinism invariants)
//! - OC-CORE-1040, OC-CORE-1041 (observability fields)

pub mod queue;

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
    /// OC-CORE-1010: health includes Ready state; dispatch must target Ready replicas only.
    fn health(&self) -> core::result::Result<WorkerHealth, WorkerError>;
    fn props(&self) -> core::result::Result<WorkerProps, WorkerError>;
    /// OC-CORE-1001/1002: admission enforces bounded queues and full policies upstream.
    fn submit(&self, req: api::TaskRequest) -> core::result::Result<TokenStream, WorkerError>;
    fn cancel(&self, task_id: &str) -> core::result::Result<(), WorkerError>;
    /// OC-CORE-1031: engine_version captured for replica set determinism pinning.
    fn engine_version(&self) -> core::result::Result<String, WorkerError>;
}
