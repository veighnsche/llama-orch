// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
// Purpose: queen-rbee library code
// Status: STUB - Awaiting implementation

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-rbee library
//!
//! TEAM-164: Extracted shared logic from main.rs into library modules
//! TEAM-186: Added job_router module for operation routing
//! TEAM-196: Add hive_client module to lib.rs
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete

pub mod hive_forwarder;
pub mod hive_subscriber; // TEAM-373: Subscribe to hive SSE streams
pub mod http; // TEAM-186: Reorganized into http/ folder with mod.rs (includes health, heartbeat)
              // TEAM-275: Removed inference_scheduler module - moved to queen-rbee-inference-scheduler crate
pub mod job_router; // TEAM-186: Job routing and operation dispatch
pub mod narration;
pub mod rhai;

// TODO: Implement library functionality
