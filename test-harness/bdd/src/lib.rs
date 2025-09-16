//! Minimal BDD harness scaffolding (no runtime execution).

pub mod steps;

// Expose cucumber World at crate::world::World for attribute macros
pub use steps::world;

// Note: cucumber 0.20 attribute macros auto-register steps via inventory when
// compiled. Ensuring all step modules are referenced from steps/mod.rs is
// sufficient. No explicit registration macro is required here.

pub fn version() -> &'static str {
    "0.0.0"
}
