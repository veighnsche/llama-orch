//! Common test utilities
//!
//! Provides shared functionality for integration tests.

use std::env;

/// Initialize test environment
pub fn init_test_env() {
    // Set test environment variables or perform other initialization
    env::set_var("RUST_LOG", "debug");
}

/// Macro to announce stub mode for tests
#[macro_export]
macro_rules! announce_stub_mode {
    ($test_name:expr) => {
        eprintln!("🧪 STUB MODE: Running {} (stub implementation)", $test_name);
    };
}

// ---
