//! Test output parsers
//!
//! This module provides parsers for different cargo test output formats:
//!
//! - **JSON parser** — Parses `cargo test --format json` output
//! - **Stable parser** — Parses standard cargo test output (fallback)
//!
//! # Philosophy
//!
//! Management requirement: Support both JSON and stable output formats.
//! JSON is preferred for structured data, but stable format is the fallback
//! for compatibility with older Rust versions.
//!
//! # Example
//!
//! ```rust
//! use proof_bundle::parsers;
//!
//! let json_output = r#"{"type":"test","event":"ok","name":"test_foo"}..."#;
//! let summary = parsers::parse_json_output(json_output).unwrap();
//! assert_eq!(summary.total, 1);
//! ```

mod json;
mod stable;

// Re-export public API
pub use json::parse_json_output;
pub use stable::parse_stable_output;
