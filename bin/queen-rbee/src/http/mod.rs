//! HTTP API for queen-rbee orchestrator daemon
//!
//! This module provides HTTP server infrastructure for the orchestrator,
//! including:
//! - Server lifecycle management (in main.rs)
//! - Route configuration (`routes`)
//! - Health endpoint (`health`)
//! - Beehive registry endpoints (`beehives`)
//! - Worker management endpoints (`workers`)
//! - Inference task orchestration (`inference`)
//! - Request/Response types (`types`)
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052

pub mod beehives;
pub mod health;
pub mod inference;
pub mod routes;
pub mod types;
pub mod workers;

// Re-export commonly used types
pub use routes::{create_router, AppState};
pub use types::*;
