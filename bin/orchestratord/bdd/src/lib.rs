//! Local BDD harness for orchestrator crate (orchestratord)

pub mod steps;

// Re-export cucumber World as crate::world::World for attribute macros
pub use steps::world;

pub fn version() -> &'static str {
    "0.0.0"
}
