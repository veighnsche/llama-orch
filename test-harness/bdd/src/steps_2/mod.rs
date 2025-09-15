pub mod world;
pub mod data_plane;
pub mod control_plane;
pub mod error_taxonomy;
pub mod security;
pub mod catalog;
pub mod lifecycle;
pub mod scheduling;
pub mod deadlines_preemption;
pub mod pool_manager;
pub mod config;
pub mod determinism;
pub mod observability;
pub mod adapters;
pub mod policy_host;
pub mod policy_sdk;
pub mod preflight_steps;
pub mod apply_steps;

// WorldInventory registration is declared at crate root (lib.rs) to provide
// crate::world::World path that attribute macros expect.
