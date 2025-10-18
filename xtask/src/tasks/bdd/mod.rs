// TEAM-111: BDD Test Runner - Rust port from bash script
//
// This module provides comprehensive BDD test execution with:
// - Live output streaming (default)
// - Quiet mode with progress spinner
// - Tag and feature filtering
// - Failure-focused reporting
// - Auto-generated rerun commands
// - Timestamped logs

mod runner;
mod parser;
mod reporter;
mod files;
mod types;
mod live_filters;

// Test modules - Testing behavior, not coverage
#[cfg(test)]
mod parser_tests;
#[cfg(test)]
mod types_tests;
#[cfg(test)]
mod files_tests;
#[cfg(test)]
mod reporter_tests;
#[cfg(test)]
mod runner_tests;

pub use runner::run_bdd_tests;
pub use types::BddConfig;
pub use live_filters::{bdd_tail, bdd_head, bdd_grep};
