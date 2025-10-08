//! HTTP API for llama-orch workers
//!
//! This module provides platform-agnostic HTTP server infrastructure for workers,
//! including:
//! - Server lifecycle management (`server`)
//! - Route configuration (`routes`)
//! - Health endpoint (`health`)
//! - Execute endpoint (`execute`)
//! - Platform abstraction (`backend`)
//!
//! # Spec References
//! - M0-W-1110: Server initialization
//! - M0-W-1320: Health endpoint
//! - M0-W-1330: Execute endpoint
//!
//! Integrated by: TEAM-015 (from worker-http crate)

pub mod backend;
pub mod execute;
pub mod health;
pub mod routes;
pub mod server;
pub mod sse;
pub mod validation;

// Re-export commonly used types
pub use backend::{AppState, InferenceBackend};
pub use routes::create_router;
pub use server::HttpServer;

// ---
// Built by Foundation-Alpha 🏗️
