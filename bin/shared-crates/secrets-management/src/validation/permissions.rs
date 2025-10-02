//! File permission validation (Unix)
//!
//! Validates that secret files have correct permissions (0600) to prevent
//! unauthorized access.

use crate::{Result, SecretError};
use std::path::Path;

/// Validate file permissions (Unix only)
///
/// # Security
///
/// - Rejects files with world-readable bit (0o004)
/// - Rejects files with group-readable bit (0o040)
/// - Recommended permissions: 0600 (owner read/write only)
///
/// # Errors
///
/// Returns `SecretError::PermissionsTooOpen` if file has world or group readable bits set.
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::validation::validate_file_permissions;
/// use std::path::Path;
///
/// let path = Path::new("/etc/llorch/secrets/api-token");
/// validate_file_permissions(path)?;
/// # Ok::<(), secrets_management::SecretError>(())
/// ```
#[cfg(unix)]
pub fn validate_file_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::metadata(path)?;
    let mode = metadata.permissions().mode();

    // Check if world or group readable (0o077 = 0o040 | 0o004 | other bits)
    if mode & 0o077 != 0 {
        return Err(SecretError::PermissionsTooOpen { path: path.display().to_string(), mode });
    }

    Ok(())
}

/// Validate file permissions (non-Unix platforms)
///
/// On non-Unix platforms, permission validation is not available.
/// Emits a warning and returns Ok.
#[cfg(not(unix))]
pub fn validate_file_permissions(path: &Path) -> Result<()> {
    tracing::warn!(
        path = %path.display(),
        "File permission validation not available on this platform"
    );
    Ok(())
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::NamedTempFile;

    #[test]
    fn test_accept_owner_only() {
        let file = NamedTempFile::new().unwrap();
        let mut perms = fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        fs::set_permissions(file.path(), perms).unwrap();

        assert!(validate_file_permissions(file.path()).is_ok());
    }

    #[test]
    fn test_reject_world_readable() {
        let file = NamedTempFile::new().unwrap();
        let mut perms = fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o644);
        fs::set_permissions(file.path(), perms).unwrap();

        let result = validate_file_permissions(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
    }

    #[test]
    fn test_reject_group_readable() {
        let file = NamedTempFile::new().unwrap();
        let mut perms = fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o640);
        fs::set_permissions(file.path(), perms).unwrap();

        let result = validate_file_permissions(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
    }
}
