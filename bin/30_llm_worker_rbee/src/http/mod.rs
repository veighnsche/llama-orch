//! HTTP API for llama-orch workers
//!
//! This module provides platform-agnostic HTTP server infrastructure for workers,
//! including:
//! - Server lifecycle management (`server`)
//! - Route configuration (`routes`)
//! - Health endpoint (`health`)
//! - Execute endpoint (`execute`)
//! - Loading progress endpoint (`loading`) - TEAM-035
//! - Ready endpoint (`ready`) - TEAM-045
//! - Platform abstraction (`backend`)
//!
//! # Spec References
//! - M0-W-1110: Server initialization
//! - M0-W-1320: Health endpoint
//! - M0-W-1330: Execute endpoint
//! - `SSE_IMPLEMENTATION_PLAN.md` Phase 2: Loading progress
//!
//! Integrated by: TEAM-015 (from worker-http crate)
//! Modified by: TEAM-035 (added loading progress)
//! Modified by: TEAM-154 (added stream endpoint for dual-call pattern)

pub mod backend;
pub mod execute;
pub mod health;
pub mod loading;
pub mod middleware; // TEAM-102: Authentication middleware
pub mod narration_channel; // TEAM-039: Narration SSE channel
pub mod ready; // TEAM-045: Worker readiness endpoint
pub mod routes;
pub mod server;
pub mod sse;
pub mod stream; // TEAM-154: Stream endpoint for dual-call pattern
pub mod validation;

// Re-export commonly used types
pub use backend::{AppState, InferenceBackend};
pub use middleware::AuthState; // TEAM-102
pub use routes::create_router;
pub use server::HttpServer;

// ---
// Built by Foundation-Alpha 🏗️
