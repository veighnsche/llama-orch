//! auth-min — Minimal authentication utilities for llama-orch
//!
//! This crate provides security-hardened primitives for authentication:
//! - **Timing-safe token comparison** - Prevents timing attacks (CWE-208)
//! - **Token fingerprinting** - SHA-256 based 6-char identifiers for safe logging
//! - **Bearer token parsing** - Robust HTTP Authorization header parsing
//! - **Bind policy enforcement** - Loopback detection for startup validation
//! - **Proxy auth trust gate** - Optional proxy header trust (use with caution)
//!
//! # Security Properties
//!
//! All token comparisons use constant-time algorithms to prevent timing side-channel attacks.
//! Token fingerprints are non-reversible SHA-256 hashes suitable for audit logs.
//!
//! # Example
//!
//! ```rust
//! use auth_min::{timing_safe_eq, token_fp6, parse_bearer};
//!
//! // Parse Bearer token from header
//! let auth_header = Some("Bearer secret-token-abc123");
//! let token = parse_bearer(auth_header).expect("valid bearer token");
//!
//! // Compare with expected token (timing-safe)
//! let expected = "secret-token-abc123";
//! if timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
//!     // Log with fingerprint (safe for logs)
//!     let fp6 = token_fp6(&token);
//!     println!("Authenticated: token:{}", fp6);
//! }
//! ```
//!
//! # Specifications
//!
//! Implements requirements from:
//! - `.specs/11_min_auth_hooks.md` (AUTH-1001..AUTH-1008)
//! - `.specs/12_auth-min-hardening.md` (SEC-AUTH-*)

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]

mod compare;
mod error;
mod fingerprint;
mod parse;
mod policy;

// Re-export public API
pub use compare::timing_safe_eq;
pub use error::{AuthError, Result};
pub use fingerprint::token_fp6;
pub use parse::parse_bearer;
pub use policy::{enforce_startup_bind_policy, is_loopback_addr, trust_proxy_auth};

#[cfg(test)]
mod tests;
