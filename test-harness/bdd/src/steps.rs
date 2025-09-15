//! Step registry and domain modules for BDD tests (no runtime execution yet).

use regex::Regex;

pub mod placeholders;
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
pub mod core_guardrails;

/// Shared test world (reserved for future runner integration)
#[allow(dead_code)]
pub struct World;

#[allow(dead_code)]
impl World {
    pub fn new() -> Self { Self }
}

/// Aggregate registry of all step regexes across domains.
pub fn registry() -> Vec<Regex> {
    let mut v = Vec::new();
    v.extend(placeholders::registry());
    v.extend(data_plane::registry());
    v.extend(control_plane::registry());
    v.extend(error_taxonomy::registry());
    v.extend(security::registry());
    v.extend(catalog::registry());
    v.extend(lifecycle::registry());
    v.extend(scheduling::registry());
    v.extend(deadlines_preemption::registry());
    v.extend(pool_manager::registry());
    v.extend(config::registry());
    v.extend(determinism::registry());
    v.extend(observability::registry());
    v.extend(adapters::registry());
    v.extend(policy_host::registry());
    v.extend(policy_sdk::registry());
    v.extend(core_guardrails::registry());
    v
}
