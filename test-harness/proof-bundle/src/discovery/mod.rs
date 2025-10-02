//! Test discovery using cargo_metadata

pub mod cargo_meta;
pub mod targets;

pub use cargo_meta::discover_tests;
pub use targets::TestTarget;
