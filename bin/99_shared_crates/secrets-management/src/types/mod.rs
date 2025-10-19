//! Secret types (Secret, SecretKey)
//!
//! Provides opaque wrappers for sensitive data with automatic zeroization on drop.
//!
//! # Security Properties
//!
//! - No Debug/Display/ToString/Serialize traits (prevents accidental logging)
//! - Automatic zeroization on drop using `zeroize` crate
//! - Timing-safe comparison for verification
//! - Uses `secrecy` crate for battle-tested implementation

mod secret;
mod secret_key;

pub use secret::Secret;
pub use secret_key::SecretKey;
