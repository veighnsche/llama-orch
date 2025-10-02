//! Test result capture from cargo test
//!
//! Provides helpers to run cargo test and capture results automatically.

mod test_capture;
mod types;

pub use test_capture::TestCaptureBuilder;
pub use types::{TestResult, TestStatus, TestSummary};
