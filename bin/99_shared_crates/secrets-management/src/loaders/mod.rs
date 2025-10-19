//! Secret loading methods
//!
//! Provides functions for loading secrets from various sources:
//! - File-based loading (primary method)
//! - Systemd credentials (production deployment)
//! - Key derivation (HKDF-SHA256)
//!
//! # Security
//!
//! All loaders validate inputs and enforce security properties:
//! - File permission validation (Unix)
//! - Path canonicalization (prevents traversal)
//! - Secure key derivation (HKDF-SHA256)

pub mod derivation;
pub mod environment;
pub mod file;
pub mod systemd;
