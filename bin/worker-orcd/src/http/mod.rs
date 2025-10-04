//! HTTP API for worker-orcd
//!
//! This module provides the complete HTTP server infrastructure for worker-orcd,
//! including:
//! - Server lifecycle management (`server`)
//! - Route configuration (`routes`)
//! - Health endpoint (`health`)
//! - Execute endpoint (`execute`)
//!
//! # Spec References
//! - M0-W-1110: Server initialization
//! - M0-W-1320: Health endpoint
//! - M0-W-1330: Execute endpoint

pub mod execute;
pub mod health;
pub mod routes;
pub mod server;
pub mod validation;

// Re-export commonly used types
pub use routes::create_router;
pub use server::HttpServer;

// ---
// Built by Foundation-Alpha 🏗️
