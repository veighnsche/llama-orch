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
//! - Device detection endpoint (`devices`) - TEAM-151
//! - VRAM capacity check endpoint (`capacity`) - TEAM-151
//! - Graceful shutdown endpoint (`shutdown`) - TEAM-151
//!
//! Created by: TEAM-026
//! Modified by: TEAM-104 (added metrics), TEAM-115 (added heartbeat endpoint)
//! Modified by: TEAM-151 (added devices, capacity, shutdown endpoints + heartbeat relay)

pub mod capacity; // TEAM-151: VRAM capacity check endpoint
pub mod devices; // TEAM-151: Device detection endpoint
pub mod health;
pub mod heartbeat; // TEAM-115: Heartbeat endpoint (TEAM-151: added relay to queen)
pub mod metrics; // TEAM-104: Prometheus metrics endpoint
pub mod middleware; // TEAM-102: Authentication middleware
pub mod models;
pub mod routes;
pub mod server;
pub mod shutdown; // TEAM-151: Graceful shutdown endpoint
pub mod workers;

// Re-export commonly used types
pub use routes::create_router;
pub use server::HttpServer;
