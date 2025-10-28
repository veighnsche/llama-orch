// TEAM-300: Modular reorganization - API module
//! Public API for the narration system

pub mod builder;
pub mod emit;
pub mod macro_impl;

// TEAM-310: short_job_id moved to format.rs module
pub use builder::{Narration, NarrationFactory};
pub use emit::*;
pub use macro_impl::macro_emit; // TEAM-312: Deleted 5 backwards compat wrappers (RULE ZERO)
