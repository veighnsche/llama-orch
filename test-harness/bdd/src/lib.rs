//! Minimal BDD harness scaffolding (no runtime execution).

pub mod steps;

#[cfg(feature = "bdd-cucumber")]
pub mod steps_2;

// Expose cucumber World at crate::world::World for attribute macros
#[cfg(feature = "bdd-cucumber")]
pub use steps_2::world as world;

// Register all cucumber steps for the World type so attribute macros compile.
#[cfg(feature = "bdd-cucumber")]
cucumber::steps!(
    crate::world::World =>
        crate::steps_2::data_plane,
        crate::steps_2::control_plane,
        crate::steps_2::error_taxonomy,
        crate::steps_2::security,
        crate::steps_2::catalog,
        crate::steps_2::lifecycle,
        crate::steps_2::scheduling,
        crate::steps_2::deadlines_preemption,
        crate::steps_2::pool_manager,
        crate::steps_2::config,
        crate::steps_2::determinism,
        crate::steps_2::observability,
        crate::steps_2::adapters,
        crate::steps_2::policy_host,
        crate::steps_2::policy_sdk,
        crate::steps_2::preflight_steps,
        crate::steps_2::apply_steps
);

pub fn version() -> &'static str {
    "0.0.0"
}
