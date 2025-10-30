//! Shared utilities for lifecycle operations
//!
//! TEAM-367: Extracted from lifecycle-local and lifecycle-ssh

pub mod serde;

// Re-export serde utilities for convenience
pub use serde::{deserialize_systemtime, serialize_systemtime};
