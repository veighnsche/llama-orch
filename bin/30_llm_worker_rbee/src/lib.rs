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
//! Modified by: TEAM-154 (migrated `job_server` to shared crate)

pub mod backend;
pub mod common;
pub mod device;
pub mod error;
pub mod heartbeat;
pub mod http;
pub mod job_router; // TEAM-353: Job-based architecture
pub mod narration;
pub mod token_output_stream;

// Re-export commonly used types
pub use backend::CandleInferenceBackend;
pub use common::{InferenceResult, SamplingConfig, StopReason, WorkerError};
pub use error::LlorchError;
pub use http::{create_router, HttpServer, InferenceBackend};

// TEAM-NARRATION-FIX: Helper to setup worker with job-based architecture
// This hides the RequestQueue + JobRegistry + GenerationEngine boilerplate
// from backend-specific binaries (cpu.rs, cuda.rs, metal.rs)
pub fn setup_worker_with_backend(
    backend: CandleInferenceBackend,
    expected_token: String,
) -> axum::Router {
    use backend::{generation_engine::GenerationEngine, request_queue::RequestQueue};
    use job_server::JobRegistry;
    use std::sync::{Arc, Mutex};

    // Wrap backend in Arc<Mutex> for sharing between engine and routes
    let backend = Arc::new(Mutex::new(backend));

    // Create request queue
    let (request_queue, request_rx) = RequestQueue::new();
    let request_queue = Arc::new(request_queue);

    // Create job registry for dual-call pattern
    let job_server: Arc<JobRegistry<backend::request_queue::TokenResponse>> =
        Arc::new(JobRegistry::new());

    // Start generation engine in background
    let generation_engine = GenerationEngine::new(Arc::clone(&backend), request_rx);
    generation_engine.start();

    // Create and return router
    create_router(request_queue, job_server, expected_token)
}
