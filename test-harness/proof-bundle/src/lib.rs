//! Proof Bundle V3 - Clean Architecture
//!
//! This is a complete rewrite with the following principles:
//! 1. Use cargo's native JSON output (no fragile text parsing)
//! 2. Extract metadata from source code (not lost in comments)
//! 3. Fail fast with clear errors (no silent failures)
//! 4. Zero boilerplate (proc macros optional)
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use proof_bundle; // crate name
//!
//! // One line to generate complete proof bundle
//! proof_bundle::generate_for_crate("my-crate", proof_bundle::Mode::UnitFast)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │  cargo test     │ (stable text output on STDERR)
//! │                 │
//! └────────┬────────┘
//!          │
//!          ├──> Structured test events
//!          ├──> Timing per test
//!          └──> Output per test
//!               
//! ┌─────────────────┐
//! │ Source Parser   │ (syn crate)
//! │ Extract @meta   │
//! └────────┬────────┘
//!          │
//!          └──> Metadata index
//!               
//! ┌─────────────────┐
//! │ Merge & Format  │
//! └────────┬────────┘
//!          │
//!          ├──> Executive summary
//!          ├──> Developer report
//!          ├──> Failure report
//!          └──> Metadata report
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod core;
pub mod discovery;
pub mod extraction;
pub mod runners;
pub mod formatters;
pub mod bundle;
pub mod api;

// Re-export core types
pub use core::{
    TestResult,
    TestSummary,
    TestStatus,
    TestMetadata,
    Mode,
};

// Re-export main API
pub use api::generate_for_crate;
pub use api::Builder;

// Re-export errors
pub use core::error::ProofBundleError;

/// Result type for proof bundle operations
pub type Result<T> = std::result::Result<T, ProofBundleError>;
