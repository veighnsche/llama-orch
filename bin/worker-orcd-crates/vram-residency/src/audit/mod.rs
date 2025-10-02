//! Audit logging integration
//!
//! Emits audit events for all VRAM operations.

pub mod events;

pub use events::{emit_vram_sealed, emit_seal_verified, emit_seal_verification_failed, emit_vram_deallocated, emit_policy_violation};
