// TEAM-270: Worker contract types and API specification
//
// This crate defines the contract that ALL worker implementations must follow.
// Workers can be:
// - Bespoke (llm-worker-rbee using Candle)
// - Adapters (llama.cpp, vLLM, ComfyUI, Ollama, etc.)
//
// All workers communicate with queen-rbee using this contract.

#![warn(missing_docs)]
#![warn(clippy::all)]

//! worker-contract
//!
//! Contract definition for LLM worker implementations in the rbee system.
//!
//! # Overview
//!
//! This crate defines the types and protocols that ALL workers must implement,
//! regardless of their underlying implementation (Candle, llama.cpp, vLLM, etc.).
//!
//! # Worker Lifecycle
//!
//! ```text
//! Hive spawns worker → Worker loads model → Worker reports ready → Worker accepts requests
//!                                    ↓
//!                            Heartbeat every 30s to queen
//! ```
//!
//! # Key Concepts
//!
//! - **WorkerInfo**: Complete worker state (model, device, status, etc.)
//! - **WorkerHeartbeat**: Periodic status update sent to queen
//! - **WorkerStatus**: Current worker state (Starting, Ready, Busy, Stopped)
//! - **Worker HTTP API**: Endpoints all workers must implement
//!
//! # Example
//!
//! ```no_run
//! use worker_contract::{WorkerInfo, WorkerStatus, WorkerHeartbeat};
//! use chrono::Utc;
//!
//! // Worker creates its info
//! let worker = WorkerInfo {
//!     id: "worker-abc123".to_string(),
//!     model_id: "meta-llama/Llama-2-7b".to_string(),
//!     device: "GPU-0".to_string(),
//!     port: 9301,
//!     status: WorkerStatus::Ready,
//!     implementation: "llm-worker-rbee".to_string(),
//!     version: "0.1.0".to_string(),
//! };
//!
//! // Worker sends heartbeat to queen
//! let heartbeat = WorkerHeartbeat {
//!     worker: worker.clone(),
//!     timestamp: Utc::now(),
//! };
//!
//! // Send to queen: POST http://queen:8500/v1/worker-heartbeat
//! ```

/// Worker HTTP API specification
pub mod api;
/// Heartbeat protocol
pub mod heartbeat;
/// Worker contract types
pub mod types;

// Re-export main types for convenience
pub use api::{InferRequest, InferResponse, WorkerApiSpec};
pub use heartbeat::{WorkerHeartbeat, HEARTBEAT_INTERVAL_SECS, HEARTBEAT_TIMEOUT_SECS};
pub use types::{WorkerInfo, WorkerStatus};

// TEAM-284: Re-export shared-contract types for convenience
pub use shared_contract::{HealthStatus, HeartbeatTimestamp, OperationalStatus};
