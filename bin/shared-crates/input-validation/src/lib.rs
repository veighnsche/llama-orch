//! Input Validation — Collection of Security Validation Applets
//!
//! This crate provides a collection of small, focused validation functions (applets)
//! that prevent injection attacks, path traversal, and resource exhaustion.
//!
//! Each applet is:
//! - **Independent**: No shared state, pure functions
//! - **Composable**: Use only what you need
//! - **Security-critical**: Never panics, always returns Result
//! - **Fast**: O(n) or better, early termination
//!
//! # Architecture
//!
//! This crate follows an **applet-based architecture**:
//! - Each validation function is a self-contained applet
//! - No framework, no validation engine, just utilities
//! - Mix and match applets as needed
//! - Similar to Unix philosophy: small tools that do one thing well
//!
//! # Available Applets
//!
//! - [`validate_identifier`] — Validate IDs (shard_id, task_id, pool_id)
//! - [`validate_model_ref`] — Validate model references (prevents injection)
//! - [`validate_hex_string`] — Validate hex strings (digests, hashes)
//! - [`validate_path`] — Validate filesystem paths (prevents traversal)
//! - [`validate_prompt`] — Validate user prompts (prevents exhaustion)
//! - [`validate_range`] — Validate integer ranges (prevents overflow)
//! - [`sanitize_string`] — Sanitize strings for logging (prevents injection)
//!
//! # Example
//!
//! ```rust
//! use input_validation::{validate_identifier, validate_model_ref, validate_hex_string};
//!
//! // Validate shard ID
//! validate_identifier("shard-abc123", 256)?;
//!
//! // Validate model reference
//! validate_model_ref("meta-llama/Llama-3.1-8B")?;
//!
//! // Validate SHA-256 digest
//! let digest = "a".repeat(64);
//! validate_hex_string(&digest, 64)?;
//! ```
//!
//! # Security Properties
//!
//! - **No panics**: All functions return `Result`, never panic
//! - **No information leakage**: Errors contain only metadata, not input content
//! - **Minimal dependencies**: Only `thiserror` (no regex, no async)
//! - **TIER 2 security**: Strict Clippy enforcement

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::missing_errors_doc)]

// Applet modules
mod error;
mod identifier;
mod model_ref;
mod hex_string;
mod path;
mod prompt;
mod range;
mod sanitize;

// Re-export public API
pub use error::{ValidationError, Result};
pub use identifier::validate_identifier;
pub use model_ref::validate_model_ref;
pub use hex_string::validate_hex_string;
pub use path::validate_path;
pub use prompt::validate_prompt;
pub use range::validate_range;
pub use sanitize::sanitize_string;

// Note: All validation implementations are in their respective applet modules.
// This file only re-exports the public API for easy access.
