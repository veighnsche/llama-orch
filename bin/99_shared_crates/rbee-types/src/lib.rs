// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-022: Original implementation in hive-core (merged 2025-10-19)
// Purpose: Shared type definitions for all rbee binaries
// Status: ACTIVE - Merged from hive-core

#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-types
//!
//! Shared type definitions and data structures used across all rbee binaries.
//!
//! This crate was previously named `hive-core` but was renamed to `rbee-types`
//! to better reflect its purpose as a system-wide shared types crate.

pub mod catalog;
pub mod error;
pub mod worker;

pub use catalog::{ModelCatalog, ModelEntry};
pub use error::{PoolError, Result};
pub use worker::{Backend, WorkerInfo};
