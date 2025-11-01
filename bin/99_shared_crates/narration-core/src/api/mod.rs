// TEAM-300: Modular reorganization - API module
// TEAM-380: DELETED builder.rs (RULE ZERO - deprecated code removed)
//! Public API for the narration system

pub mod emit;
pub mod macro_impl;

pub use emit::narrate; // TEAM-380: Only export the modern API
pub use macro_impl::macro_emit; // TEAM-312: Deleted 5 backwards compat wrappers (RULE ZERO)
