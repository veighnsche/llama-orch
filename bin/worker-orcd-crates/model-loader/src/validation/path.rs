//! Filesystem path validation
//!
//! Implements PATH-001 to PATH-008 from 20_security.md
//! Prevents CWE-22 (Path Traversal)

use crate::error::{LoadError, Result};
use std::path::{Path, PathBuf};

/// Validate filesystem path for model loading
///
/// # Security Requirements
/// - PATH-001: All paths MUST be validated via input-validation crate
/// - PATH-002: Paths MUST be canonicalized (resolve .. and symlinks)
/// - PATH-003: Canonicalized paths MUST be checked against allowed root
/// - PATH-004: Paths MUST NOT contain null bytes
/// - PATH-005: Paths MUST NOT contain path traversal sequences
/// - PATH-006: Symlinks MUST be resolved and validated
/// - PATH-007: Absolute paths outside allowed root MUST be rejected
/// - PATH-008: Path validation failures MUST NOT expose sensitive paths
pub fn validate_path(path: &Path, allowed_root: &Path) -> Result<PathBuf> {
    // Convert relative paths to absolute by joining with allowed_root
    // input-validation will canonicalize and validate
    let path_to_validate = if path.is_relative() {
        allowed_root.join(path)
    } else {
        path.to_path_buf()
    };
    
    // Use input-validation crate for secure path validation
    // This will:
    // 1. Check for path traversal sequences (.. and .)
    // 2. Check for null bytes
    // 3. Reject absolute paths (unless we made it absolute by joining)
    // 4. Canonicalize and verify it's within allowed_root
    input_validation::validate_path(&path_to_validate, allowed_root)
        .map_err(|e| LoadError::PathValidationFailed(e.to_string()))?;
    
    // Canonicalize path (resolves symlinks and ..)
    let canonical = path_to_validate.canonicalize().map_err(|e| LoadError::Io(e))?;
    
    // Double-check canonical path is within allowed root
    let canonical_root = allowed_root.canonicalize().map_err(|e| LoadError::Io(e))?;
    if !canonical.starts_with(&canonical_root) {
        return Err(LoadError::PathValidationFailed(
            "Path outside allowed directory after canonicalization".to_string()
        ));
    }
    
    tracing::debug!(
        path = ?canonical,
        allowed_root = ?allowed_root,
        "Path validated"
    );
    
    Ok(canonical)
}

// TODO(Post-M0): Add path security tests per 20_security.md §6.3
// - Test path traversal rejection
// - Test symlink escape rejection
// - Test null byte rejection
// - Test absolute path rejection

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rejects_null_byte() {
        let path = Path::new("model\0.gguf");
        let allowed = Path::new("/models");
        
        let result = validate_path(path, allowed);
        assert!(matches!(result, Err(LoadError::PathValidationFailed(_))));
    }
    
    #[test]
    fn test_rejects_path_traversal() {
        let path = Path::new("../../../etc/passwd");
        let allowed = Path::new("/models");
        
        let result = validate_path(path, allowed);
        assert!(matches!(result, Err(LoadError::PathValidationFailed(_))));
    }
}
