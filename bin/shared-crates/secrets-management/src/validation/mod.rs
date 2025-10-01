//! File permission and path validation
//!
//! Provides security validation for file operations:
//! - File permission validation (Unix)
//! - Path canonicalization (prevents traversal)
//!
//! # Security
//!
//! All validation occurs BEFORE reading file contents to prevent TOCTOU issues.

mod permissions;
mod paths;

pub use permissions::validate_file_permissions;
pub use paths::{canonicalize_path, validate_path_within_root};
