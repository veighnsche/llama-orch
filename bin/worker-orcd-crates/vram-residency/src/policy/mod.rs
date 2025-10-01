//! VRAM-only policy enforcement
//!
//! Ensures models reside entirely in VRAM during inference.

pub mod enforcement;
pub mod validation;

pub use enforcement::enforce_vram_only_policy;
pub use validation::{validate_device_properties, check_unified_memory};
