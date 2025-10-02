//! Path canonicalization and validation
//!
//! Canonicalizes file paths to prevent directory traversal attacks.

use crate::{Result, SecretError};
use std::path::{Path, PathBuf};

/// Canonicalize a file path
///
/// # Security
///
/// - Resolves `..` sequences
/// - Resolves symlinks
/// - Returns absolute path
/// - Prevents directory traversal attacks
///
/// # Errors
///
/// Returns `SecretError::FileNotFound` if path does not exist or cannot be canonicalized.
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::validation::canonicalize_path;
/// use std::path::Path;
///
/// let path = Path::new("../../../etc/passwd");
/// let canonical = canonicalize_path(path)?;
/// // Returns absolute path, not relative traversal
/// # Ok::<(), secrets_management::SecretError>(())
/// ```
pub fn canonicalize_path(path: &Path) -> Result<PathBuf> {
    path.canonicalize().map_err(|_| SecretError::FileNotFound(path.display().to_string()))
}

/// Validate path is within allowed root directory (optional)
/// # Security
///
/// - Ensures canonicalized path starts with allowed root
/// - Prevents access to files outside allowed directory
///
/// # Errors
///
/// Returns `SecretError::PathValidationFailed` if path is outside allowed root.
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::validation::{canonicalize_path, validate_path_within_root};
/// use std::path::{Path, PathBuf};
///
/// # fn main() -> Result<(), secrets_management::SecretError> {
/// let allowed_root = PathBuf::from("/etc/llorch/secrets");
/// let path = Path::new("/etc/llorch/secrets/api-token");
///
/// let canonical = canonicalize_path(path)?;
/// validate_path_within_root(&canonical, &allowed_root)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)] // Reserved for future use
pub fn validate_path_within_root(canonical: &Path, allowed_root: &Path) -> Result<()> {
    if !canonical.starts_with(allowed_root) {
        return Err(SecretError::PathValidationFailed(format!(
            "path '{}' is outside allowed directory '{}'",
            canonical.display(),
            allowed_root.display()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_canonicalize_valid_path() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "test").unwrap();

        let canonical = canonicalize_path(&file).unwrap();
        assert!(canonical.is_absolute());
    }

    #[test]
    fn test_canonicalize_nonexistent() {
        let result = canonicalize_path(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_within_root() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "test").unwrap();

        let canonical = canonicalize_path(&file).unwrap();
        let result = validate_path_within_root(&canonical, dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_outside_root() {
        let dir = TempDir::new().unwrap();
        let other_dir = TempDir::new().unwrap();
        let file = other_dir.path().join("test.txt");
        fs::write(&file, "test").unwrap();

        let canonical = canonicalize_path(&file).unwrap();
        let result = validate_path_within_root(&canonical, dir.path());
        assert!(result.is_err());
    }
}
