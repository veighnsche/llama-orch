//! Validation modules for model loading
//!
//! Each validation step is isolated for clarity and testing.

pub mod hash;
pub mod path;
pub mod gguf;

// TODO(M0): Add more validation modules per 20_security.md
// pub mod signature;  // Signature verification (optional feature)
// pub mod limits;     // Resource limit enforcement
