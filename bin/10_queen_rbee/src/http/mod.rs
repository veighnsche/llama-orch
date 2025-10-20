//! HTTP API for queen-rbee orchestrator daemon
//!
//! TEAM-151: Temporarily only health endpoint is active
//! Other endpoints require registries to be migrated first
//!
//! This module provides HTTP server infrastructure for the orchestrator,
//! including:
//! - Server lifecycle management (in main.rs)
//! - Health endpoint (`health`) ✅ ACTIVE
//! - Route configuration (`routes`) ⏳ TODO: needs registries
//! - Beehive registry endpoints (`beehives`) ⏳ TODO: needs beehive_registry
//! - Worker management endpoints (`workers`) ⏳ TODO: needs worker_registry
//! - Inference task orchestration (`inference`) ⏳ TODO: needs registries
//! - Request/Response types (`types`) ⏳ TODO: needs registries
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052

// TEAM-151: Only health endpoint active for now
pub mod health;
pub mod types; // Only for HealthResponse

// TODO: Uncomment when registries are migrated
// pub mod beehives;
// pub mod inference;
// pub mod middleware;
// pub mod routes;
// pub mod workers;

// Re-export commonly used types
// TODO: Uncomment when routes are active
// pub use routes::{create_router, AppState};
// pub use types::HealthResponse; // Not needed, used directly in health.rs
