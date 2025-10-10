// Step definitions for BDD tests
// Created by: TEAM-040
// Modified by: TEAM-051 (added global_queen module)
// Modified by: TEAM-053 (added missing modules: gguf, inference_execution, lifecycle)
// Modified by: TEAM-061 (added error_handling module)

pub mod background;
pub mod beehive_registry;
pub mod cli_commands;
pub mod edge_cases;
pub mod error_handling;
pub mod error_helpers; // TEAM-062: Error verification helpers
pub mod error_responses;
pub mod gguf;
pub mod global_queen;
pub mod happy_path;
pub mod inference_execution;
pub mod lifecycle;
pub mod model_provisioning;
pub mod pool_preflight;
pub mod registry;
pub mod worker_health;
pub mod worker_preflight;
pub mod worker_registration;
pub mod worker_startup;
pub mod world;
