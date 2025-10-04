//! Integration test framework
//!
//! Provides test harness for end-to-end testing of HTTP → FFI → CUDA → HTTP flow.
//!
//! # Spec References
//! - M0-W-1820: Integration test framework

pub mod framework;
pub mod helpers;
pub mod fixtures;

pub use framework::WorkerTestHarness;
pub use helpers::{collect_sse_events, assert_event_order, extract_tokens, make_test_request};
pub use fixtures::{TestModel, TestConfig, TestPrompts};

// ---
// Built by Foundation-Alpha 🏗️
