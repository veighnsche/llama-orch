// TEAM-135: Created by TEAM-135 (scaffolding)
// Purpose: queen-rbee library code
// Status: STUB - Awaiting implementation

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-rbee library
//!
//! TEAM-164: Extracted shared logic from main.rs into library modules
//! TEAM-186: Added job_router module for operation routing

pub mod health;
pub mod heartbeat;
pub mod http;  // TEAM-186: Reorganized into http/ folder with mod.rs
pub mod job_router;  // TEAM-186: Job routing and operation dispatch
pub mod operations;

// TODO: Implement library functionality
