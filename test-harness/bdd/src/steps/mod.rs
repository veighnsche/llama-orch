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
// Modified by: TEAM-078 (added new step modules for M1 feature reorganization)

pub mod background;
pub mod beehive_registry;
pub mod cli_commands;
pub mod concurrency; // TEAM-079: Concurrency and race condition testing
pub mod edge_cases;
pub mod error_handling;
pub mod error_helpers; // TEAM-062: Error verification helpers
pub mod error_responses;
pub mod failure_recovery; // TEAM-079: Failover and recovery scenarios
pub mod gguf;
pub mod global_hive;   // TEAM-085: Global rbee-hive lifecycle for localhost tests
pub mod global_queen;
pub mod happy_path;
pub mod inference_execution;
pub mod integration; // TEAM-083: Integration and E2E tests
pub mod lifecycle;
pub mod model_catalog; // TEAM-078: SQLite model catalog queries
pub mod model_provisioning;
pub mod narration_verification; // TEAM-085: Verify product code emits narration
pub mod pool_preflight;
pub mod queen_rbee_registry; // TEAM-078: Global worker registry (in-memory)
pub mod rbee_hive_preflight; // TEAM-078: rbee-hive preflight validation
pub mod registry;
pub mod ssh_preflight; // TEAM-078: SSH preflight validation
pub mod worker_health;
pub mod worker_preflight;
pub mod worker_provisioning; // TEAM-078: Worker binary provisioning (cargo build)
pub mod worker_registration;
pub mod worker_startup;
pub mod world;
