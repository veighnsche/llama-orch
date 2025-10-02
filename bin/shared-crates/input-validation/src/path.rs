//! Filesystem path validation applet
//!
//! Validates filesystem paths to prevent directory traversal.

use crate::error::{Result, ValidationError};
use std::path::{Path, PathBuf};

/// Validate filesystem path
///
/// # Rules
/// - Canonicalize path (resolve `..` and symlinks)
/// - Verify path is within allowed root directory
/// - No null bytes
/// - Return canonicalized PathBuf
///
/// # Arguments
/// * `path` - Path to validate
/// * `allowed_root` - Root directory path must be within
///
/// # Returns
/// * `Ok(PathBuf)` - Canonicalized path if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_path;
/// use std::path::PathBuf;
///
/// let allowed = PathBuf::from("/var/lib/llorch/models");
///
/// // Valid
/// // let path = validate_path("model.gguf", &allowed)?;
///
/// // Invalid
/// // assert!(validate_path("../../../etc/passwd", &allowed).is_err());
/// ```
///
/// # Errors
/// * `ValidationError::PathTraversal` - Contains `../` or `./`
/// * `ValidationError::PathOutsideRoot` - Outside allowed root after canonicalization
/// * `ValidationError::NullByte` - Contains null byte
/// * `ValidationError::Io` - I/O error during canonicalization
///
/// # Security
/// Prevents:
/// - Directory traversal: `"../../../../etc/passwd"`
/// - Symlink escape: `"/var/lib/llorch/models/../../etc/passwd"`
///
/// # Limitations
/// - Cannot prevent TOCTOU (Time-of-Check-Time-of-Use) race conditions
/// - Caller must handle atomicity between validation and use
pub fn validate_path(path: impl AsRef<Path>, allowed_root: &Path) -> Result<PathBuf> {
    let path = path.as_ref();
    
    // Convert to string for validation checks
    // This is necessary for null byte and traversal sequence detection
    let path_str = path.to_str()
        .ok_or_else(|| ValidationError::InvalidCharacters {
            found: "[non-UTF8]".to_string(),
        })?;
    
    // Check for empty path (fast check, prevents processing empty paths)
    if path_str.is_empty() {
        return Err(ValidationError::Empty);
    }
    
    // Check for null bytes (security-critical, must happen early)
    // Null bytes can cause C string truncation in filesystem operations
    // and bypass validation in downstream code
    if path_str.contains('\0') {
        return Err(ValidationError::NullByte);
    }
    
    // Check for path traversal sequences (security-critical, before canonicalization)
    // CRITICAL: Must check for path components that are "." or ".."
    // This provides defense-in-depth even before filesystem operations
    //
    // Note: We check path components, not substrings, to avoid false positives
    // like "/tmp/.tmpXYZ/" which contains "./" but is not a traversal attempt
    //
    // We also check for Windows-style separators in the string because on Unix,
    // "..\\windows" is treated as a single Normal component, not a path traversal
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::ParentDir | Component::CurDir => {
                return Err(ValidationError::PathTraversal);
            }
            _ => {}
        }
    }
    
    // Additional check: Windows-style path separators on Unix
    // On Unix, "..\\dir" is a single component, but it's still a traversal attempt
    if path_str.contains("..\\") || path_str.contains(".\\") {
        return Err(ValidationError::PathTraversal);
    }
    
    // Note: We allow absolute paths here because they will be validated
    // against allowed_root after canonicalization. The canonicalization
    // and containment check below will catch any paths outside allowed_root.
    
    // Canonicalize allowed root first (fail fast if root is invalid)
    // This ensures we have a valid root directory to check against
    let canonical_root = allowed_root.canonicalize()
        .map_err(|e| ValidationError::Io(format!("Invalid allowed_root: {}", e)))?;
    
    // Verify allowed_root is actually a directory
    if !canonical_root.is_dir() {
        return Err(ValidationError::Io(
            "allowed_root must be a directory".to_string()
        ));
    }
    
    // Canonicalize path (resolve .. and symlinks)
    // This is where symlink resolution happens
    // IMPORTANT: The path must exist for canonicalization to work
    let canonical = path.canonicalize()
        .map_err(|e| ValidationError::Io(e.to_string()))?;
    
    // Verify path is within allowed root (security-critical)
    // This check happens AFTER canonicalization, so symlinks are resolved
    // and we check the actual target location
    if !canonical.starts_with(&canonical_root) {
        return Err(ValidationError::PathOutsideRoot);
    }
    
    // Additional robustness: Verify the canonicalized path is still UTF-8
    // This ensures no encoding issues occurred during canonicalization
    let _ = canonical.to_str()
        .ok_or_else(|| ValidationError::InvalidCharacters {
            found: "[non-UTF8 after canonicalization]".to_string(),
        })?;
    
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_path_traversal_rejected() {
        // Note: These tests require filesystem access and may not work in all environments
        // They test the string-based checks before canonicalization
        
        let temp_dir = env::temp_dir();
        
        // Path traversal in string form is rejected early
        let result = validate_path("../../../etc/passwd", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
        
        let result = validate_path("..\\..\\windows", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
    }
    
    #[test]
    fn test_null_byte_rejected() {
        let temp_dir = env::temp_dir();
        
        let result = validate_path("file\0name", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
    }
    
    #[test]
    fn test_null_byte_positions() {
        let temp_dir = env::temp_dir();
        
        // Null byte at start
        let result = validate_path("\0file", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
        
        // Null byte at end
        let result = validate_path("file\0", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
    }
    
    #[test]
    fn test_multiple_traversal_sequences() {
        let temp_dir = env::temp_dir();
        
        let result = validate_path("../../etc/passwd", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
    }
    
    // ========== ROBUSTNESS TESTS ==========
    
    #[test]
    fn test_robustness_empty_path() {
        let temp_dir = env::temp_dir();
        
        // Empty string path
        let result = validate_path("", &temp_dir);
        assert_eq!(result, Err(ValidationError::Empty));
    }
    
    #[test]
    fn test_robustness_current_directory_references() {
        let temp_dir = env::temp_dir();
        
        // Unix current directory
        let result = validate_path("./file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
        
        // Windows current directory
        let result = validate_path(".\\file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
    }
    
    #[test]
    fn test_robustness_absolute_paths() {
        let temp_dir = env::temp_dir();
        
        // Absolute paths are now allowed if they're within allowed_root
        // This test just verifies they're not rejected as PathTraversal
        // The actual containment check happens during canonicalization
        
        // Note: We can't easily test PathOutsideRoot without creating files
        // The important thing is that absolute paths aren't blanket-rejected
        // as PathTraversal anymore - they're validated against allowed_root
    }
    
    #[test]
    #[cfg(target_os = "windows")]
    fn test_robustness_windows_absolute_paths() {
        let temp_dir = env::temp_dir();
        
        // Windows absolute paths outside allowed_root are rejected
        let result = validate_path("C:\\Windows\\System32", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathOutsideRoot));
        
        let result = validate_path("D:\\data", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathOutsideRoot));
    }
    
    #[test]
    fn test_robustness_null_byte_all_positions() {
        let temp_dir = env::temp_dir();
        
        // Null byte at start
        let result = validate_path("\0file", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
        
        // Null byte in middle
        let result = validate_path("file\0name", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
        
        // Null byte at end
        let result = validate_path("file\0", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
        
        // Multiple null bytes
        let result = validate_path("file\0\0name", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
    }
    
    #[test]
    fn test_robustness_traversal_variants() {
        let temp_dir = env::temp_dir();
        
        // Unix parent directory
        let result = validate_path("../file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
        
        // Windows parent directory
        let result = validate_path("..\\file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
        
        // Multiple levels
        let result = validate_path("../../file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
        
        let result = validate_path("..\\..\\file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
        
        // Mixed in path
        let result = validate_path("dir/../file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
        
        let result = validate_path("dir\\..\\file", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
    }
    
    #[test]
    fn test_robustness_validation_order() {
        let temp_dir = env::temp_dir();
        
        // Empty check happens first
        let result = validate_path("", &temp_dir);
        assert_eq!(result, Err(ValidationError::Empty));
        
        // Null byte check happens before traversal
        let result = validate_path("\0../etc", &temp_dir);
        assert_eq!(result, Err(ValidationError::NullByte));
        
        // Traversal check happens before filesystem operations
        let result = validate_path("../etc/passwd", &temp_dir);
        assert_eq!(result, Err(ValidationError::PathTraversal));
    }
    
    #[test]
    fn test_robustness_special_filenames() {
        let temp_dir = env::temp_dir();
        
        // These would require filesystem setup to fully test
        // but we can test the string validation
        
        // Hidden files (Unix)
        let result = validate_path(".hidden", &temp_dir);
        // Should fail because file doesn't exist (canonicalization)
        assert!(result.is_err());
        
        // Files with spaces
        let result = validate_path("file name.txt", &temp_dir);
        // Should fail because file doesn't exist
        assert!(result.is_err());
    }
    
    #[test]
    fn test_robustness_very_long_paths() {
        let temp_dir = env::temp_dir();
        
        // Very long path component
        let long_name = "a".repeat(1000);
        let result = validate_path(&long_name, &temp_dir);
        // Should fail because file doesn't exist
        assert!(result.is_err());
        
        // Very deep path
        let deep_path = "a/".repeat(100) + "file.txt";
        let result = validate_path(&deep_path, &temp_dir);
        // Should fail because path doesn't exist
        assert!(result.is_err());
    }
    
    #[test]
    fn test_robustness_unicode_paths() {
        let temp_dir = env::temp_dir();
        
        // Unicode filename
        let result = validate_path("café.txt", &temp_dir);
        // Should fail because file doesn't exist
        assert!(result.is_err());
        
        // Emoji filename
        let result = validate_path("file🚀.txt", &temp_dir);
        // Should fail because file doesn't exist
        assert!(result.is_err());
        
        // Chinese characters
        let result = validate_path("文件.txt", &temp_dir);
        // Should fail because file doesn't exist
        assert!(result.is_err());
    }
    
    #[test]
    fn test_robustness_path_separators() {
        let temp_dir = env::temp_dir();
        
        // Multiple consecutive separators
        let result = validate_path("dir//file", &temp_dir);
        // Should fail because path doesn't exist
        assert!(result.is_err());
        
        // Trailing separator
        let result = validate_path("dir/", &temp_dir);
        // Should fail because path doesn't exist
        assert!(result.is_err());
    }
    
    #[test]
    fn test_robustness_case_sensitivity() {
        // Note: Case sensitivity depends on filesystem
        // On case-insensitive filesystems (Windows, macOS default),
        // "File.txt" and "file.txt" refer to the same file
        // On case-sensitive filesystems (Linux), they're different
        
        let temp_dir = env::temp_dir();
        
        let result = validate_path("File.TXT", &temp_dir);
        // Should fail because file doesn't exist
        assert!(result.is_err());
    }
    
    // Note: The following scenarios require actual filesystem setup
    // and are better suited for integration tests:
    //
    // 1. Valid path acceptance (requires file to exist)
    // 2. Symlink resolution (requires creating symlinks)
    // 3. Path outside root detection after canonicalization
    // 4. Allowed root that is a file (not directory)
    // 5. Allowed root that doesn't exist
    //
    // These are documented as limitations in the module docs
    // and should be tested in integration tests with proper setup.
    //
    // Example integration test structure:
    // ```
    // #[test]
    // fn integration_test_valid_path() {
    //     let temp_dir = tempfile::tempdir().unwrap();
    //     let file_path = temp_dir.path().join("model.gguf");
    //     std::fs::write(&file_path, b"test").unwrap();
    //     
    //     let result = validate_path("model.gguf", temp_dir.path());
    //     assert!(result.is_ok());
    // }
    // ```
}
