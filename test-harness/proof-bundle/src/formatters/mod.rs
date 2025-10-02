//! Human-readable report formatters
//!
//! This module provides formatters that convert test results into beautiful,
//! scannable markdown reports for different audiences:
//!
//! - **Executive summaries** — For management (non-technical, risk-focused)
//! - **Developer reports** — For technical reviewers (detailed, with links)
//! - **Failure reports** — For debugging (stack traces, context)
//! - **Metadata reports** — For compliance and requirements tracking
//!
//! # Philosophy
//!
//! Management requirement: Developers hate writing formatters. We provide them
//! so no crate ever duplicates this code. These formatters are the *standard*
//! across the entire repository.
//!
//! # Example
//!
//! ```rust
//! use proof_bundle::{TestSummary, formatters};
//!
//! let summary = TestSummary {
//!     total: 100,
//!     passed: 98,
//!     failed: 2,
//!     ignored: 0,
//!     duration_secs: 5.2,
//!     pass_rate: 98.0,
//!     tests: vec![/* ... */],
//! };
//!
//! let executive = formatters::generate_executive_summary(&summary);
//! let developer = formatters::generate_test_report(&summary);
//! let failures = formatters::generate_failure_report(&summary);
//! let metadata = formatters::generate_metadata_report(&summary);
//! ```

mod executive;
mod developer;
mod failure;
mod metadata_report;
mod helpers;

#[cfg(test)]
mod tests;

// Re-export public API
pub use executive::generate_executive_summary;
pub use developer::generate_test_report;
pub use failure::generate_failure_report;
pub use metadata_report::generate_metadata_report;
