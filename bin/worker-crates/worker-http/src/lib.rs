//! Shared HTTP server and SSE streaming for llama-orch workers
//!
//! This crate provides common HTTP server infrastructure used by all worker
//! implementations (worker-orcd, worker-aarmd, etc.).
//!
//! # Components
//!
//! - `HttpServer`: Axum-based HTTP server with graceful shutdown
//! - SSE streaming helpers for token-by-token inference
//! - Request validation and error handling
//! - Route definitions

// TODO: Extract from worker-orcd/src/http/
// - server.rs
// - sse.rs
// - routes.rs
// - validation.rs

pub mod placeholder {
    //! Placeholder module until extraction is complete
    
    pub fn version() -> &'static str {
        "0.1.0"
    }
}
