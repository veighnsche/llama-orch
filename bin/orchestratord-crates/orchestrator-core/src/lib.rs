//! Core orchestrator library (pre-code).
//! Queueing, scheduling hooks, and core invariants. No HTTP or adapter traits here.
//!
//! Traceability (SPEC):
//! - OC-CORE-1001, OC-CORE-1002, OC-CORE-1004 (queue & admission invariants)
//! - OC-CORE-1010, OC-CORE-1011, OC-CORE-1012 (placement & readiness)
//! - OC-CORE-1030 (determinism invariants)
//! - OC-CORE-1040, OC-CORE-1041 (observability fields)

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

pub mod queue;
