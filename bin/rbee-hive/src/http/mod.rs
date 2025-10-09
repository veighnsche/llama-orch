//! HTTP API for rbee-hive pool manager daemon
//!
//! This module provides HTTP server infrastructure for the pool manager,
//! including:
//! - Server lifecycle management (`server`)
//! - Route configuration (`routes`)
//! - Health endpoint (`health`)
//! - Worker management endpoints (`workers`)
//! - Model management endpoints (`models`)
//!
//! Created by: TEAM-026

pub mod health;
pub mod models;
pub mod routes;
pub mod server;
pub mod workers;

// Re-export commonly used types
pub use routes::create_router;
pub use server::HttpServer;
