// TEAM-135: Created by TEAM-135 (scaffolding)
// Purpose: Generic heartbeat protocol for health monitoring
// Status: STUB - Awaiting implementation
// NOTE: Moved to shared-crates because BOTH workers AND hives send heartbeats

#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-heartbeat
//!
//! Generic heartbeat mechanism for health monitoring in llama-orch.
//!
//! This crate provides a reusable heartbeat sender used by:
//! - Workers: Send heartbeats to rbee-hive (30s interval)
//! - Hives: Send aggregated heartbeats to queen-rbee (15s interval)
//!
//! See README.md for detailed documentation and usage examples.

// TODO: Implement generic heartbeat functionality
// TODO: Support configurable payload types (worker vs pool heartbeats)
