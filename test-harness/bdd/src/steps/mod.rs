// Step definitions for BDD tests
// Created by: TEAM-040
// Modified by: TEAM-051 (added global_queen module)
// Modified by: TEAM-053 (added missing modules: gguf, inference_execution, lifecycle)
// Modified by: TEAM-061 (added error_handling module)
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)
// Modified by: TEAM-076 (made modules public for library access)

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
