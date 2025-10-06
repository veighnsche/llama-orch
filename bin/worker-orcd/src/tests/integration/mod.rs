//! Integration test framework
//!
//! Provides test harness for end-to-end testing of HTTP → FFI → CUDA → HTTP flow.
//!
//! # Spec References
//! - M0-W-1820: Integration test framework

pub mod fixtures;
pub mod framework;
pub mod helpers;

pub use fixtures::{TestConfig, TestModel, TestPrompts};
pub use framework::WorkerTestHarness;
pub use helpers::{assert_event_order, collect_sse_events, extract_tokens, make_test_request};

// ---
// Built by Foundation-Alpha 🏗️
