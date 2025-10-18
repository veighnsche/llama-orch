// TEAM-111: BDD Test Runner - Rust port from bash script
//
// This module provides comprehensive BDD test execution with:
// - Live output streaming (default)
// - Quiet mode with progress spinner
// - Tag and feature filtering
// - Failure-focused reporting
// - Auto-generated rerun commands
// - Timestamped logs

mod files;
mod live_filters;
mod parser;
mod reporter;
mod runner;
mod types;

// Test modules - Testing behavior, not coverage
#[cfg(test)]
mod files_tests;
#[cfg(test)]
mod parser_tests;
#[cfg(test)]
mod reporter_tests;
#[cfg(test)]
mod runner_tests;
#[cfg(test)]
mod types_tests;

pub use live_filters::{bdd_grep, bdd_head, bdd_tail};
pub use runner::run_bdd_tests;
pub use types::BddConfig;
