// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Module structure

//! llm-worker-rbee - Candle-based Llama-2 inference library
//!
//! TEAM-009 rewrite: Uses `candle-transformers::models::llama::Llama` directly
//! instead of custom layer implementations.
//!
//! Architecture:
//! - `SafeTensors` model loading via `VarBuilder`
//! - `HuggingFace` tokenizers integration
//! - Multi-backend support (CPU, CUDA, Accelerate)
//! - Worker integration via `InferenceBackend` trait
//!
//! Created by: TEAM-000 (Foundation)
//! Modified by: TEAM-010 (Removed all deprecated modules)
//! Modified by: TEAM-014 (Added `token_output_stream` module)
//! Modified by: TEAM-015 (Integrated worker-common and worker-http)
//! Modified by: TEAM-088 (added comprehensive error narration)
//! Modified by: TEAM-115 (added heartbeat mechanism)
//! Modified by: TEAM-154 (migrated `job_registry` to shared crate)

pub mod backend;
pub mod common;
pub mod device;
pub mod error;
pub mod http;
pub mod narration;
pub mod token_output_stream;

// Re-export commonly used types
pub use backend::CandleInferenceBackend;
pub use common::{InferenceResult, SamplingConfig, StopReason, WorkerError};
pub use error::LlorchError;
pub use http::{create_router, HttpServer, InferenceBackend};

// TEAM-135: Re-export heartbeat from shared crate (migrated from local module)
pub use rbee_heartbeat as heartbeat;
