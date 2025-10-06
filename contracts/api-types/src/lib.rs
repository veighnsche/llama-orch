//! Generated API types crate (pre-code). Types are generated via `cargo xtask regen-openapi`.

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::arithmetic_side_effects)]
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

#[allow(dead_code)]
pub mod generated;

#[allow(dead_code)]
pub mod generated_control;

pub use generated::*;
pub mod control {
    pub use crate::generated_control::*;
}
