//! HTTP API for rbee-hive pool manager daemon
//!
//! This module provides HTTP server infrastructure for the pool manager,
//! including:
//! - Server lifecycle management (`server`)
//! - Route configuration (`routes`)
//! - Health endpoint (`health`)
//! - Worker management endpoints (`workers`)
//! - Model management endpoints (`models`)
//! - Metrics endpoint (`metrics`) - TEAM-104
//! - Heartbeat endpoint (`heartbeat`) - TEAM-115
//!
//! Created by: TEAM-026
//! Modified by: TEAM-104 (added metrics), TEAM-115 (added heartbeat endpoint)

pub mod health;
pub mod heartbeat; // TEAM-115: Heartbeat endpoint
pub mod metrics; // TEAM-104: Prometheus metrics endpoint
pub mod middleware; // TEAM-102: Authentication middleware
pub mod models;
pub mod routes;
pub mod server;
pub mod workers;

// Re-export commonly used types
pub use routes::create_router;
pub use server::HttpServer;
