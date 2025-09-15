//! Minimal BDD harness scaffolding (no runtime execution).

#[cfg(feature = "bdd-cucumber")]
pub mod steps;

// Expose cucumber World at crate::world::World for attribute macros
#[cfg(feature = "bdd-cucumber")]
pub use steps::world as world;

// Register all cucumber steps for the World type so attribute macros compile.
#[cfg(feature = "bdd-cucumber")]
cucumber::steps!(
    crate::world::World =>
        crate::steps::data_plane,
        crate::steps::control_plane,
        crate::steps::error_taxonomy,
        crate::steps::security,
        crate::steps::catalog,
        crate::steps::lifecycle,
        crate::steps::scheduling,
        crate::steps::deadlines_preemption,
        crate::steps::pool_manager,
        crate::steps::config,
        crate::steps::determinism,
        crate::steps::observability,
        crate::steps::adapters,
        crate::steps::policy_host,
        crate::steps::policy_sdk,
        crate::steps::preflight_steps,
        crate::steps::apply_steps
);

pub fn version() -> &'static str {
    "0.0.0"
}
