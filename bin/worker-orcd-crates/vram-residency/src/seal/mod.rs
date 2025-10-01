//! Cryptographic sealing operations
//!
//! This module handles HMAC-SHA256 seal signature computation and verification.

pub mod signature;
pub mod digest;
pub mod key_derivation;

pub use signature::{compute_signature, verify_signature};
pub use digest::compute_digest;
pub use key_derivation::derive_seal_key;
