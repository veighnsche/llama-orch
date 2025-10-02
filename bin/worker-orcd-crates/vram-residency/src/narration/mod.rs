//! Narration module for vram-residency
//!
//! Provides structured, human-readable narration for all VRAM operations.
//!
//! # Purpose
//!
//! Narration provides observability for:
//! - Debugging (what happened and why)
//! - Performance analysis (timing, throughput)
//! - Capacity planning (VRAM usage patterns)
//! - Incident investigation (correlation with audit logs)

pub mod events;

pub use events::*;
