// TEAM-300: Modular reorganization - API module
//! Public API for the narration system

pub mod builder;
pub mod emit;
pub mod macro_impl;

// TEAM-310: short_job_id moved to format.rs module
pub use builder::{Narration, NarrationFactory};
pub use emit::*;
pub use macro_impl::{macro_emit, macro_emit_auto, macro_emit_auto_with_level, macro_emit_with_actor}; // TEAM-309: Export all variants, TEAM-311: Added with_level
