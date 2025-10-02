//! Security tests for model-loader
//!
//! Tests security vulnerabilities from 20_security.md §3

use model_loader::{LoadError, ModelLoader};
use std::fs;
use std::os::unix::fs as unix_fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_buffer_overflow_oversized_string() {
    use model_loader::validation::gguf::parser;
    
    // GGUF-003: String length validation
    // Try to read a string with 1GB length (should be rejected)
    let mut bytes = vec![];
    let huge_len = 1_000_000_000u64;
    bytes.extend_from_slice(&huge_len.to_le_bytes());
    
    let result = parser::read_string(&bytes, 0);
    
    // Must reject oversized string
    assert!(matches!(result, Err(LoadError::StringTooLong { .. })));
}

#[test]
fn test_buffer_overflow_read_past_end() {
    use model_loader::validation::gguf::parser;
    
    let bytes = vec![0x01, 0x02, 0x03];
    
    // Try to read u32 at offset that would go past end
    let result = parser::read_u32(&bytes, 1);
    assert!(matches!(result, Err(LoadError::BufferOverflow { .. })));
    
    // Try to read u64 at offset that would go past end
    let result = parser::read_u64(&bytes, 0);
    assert!(matches!(result, Err(LoadError::BufferOverflow { .. })));
}

#[test]
fn test_integer_overflow_tensor_count() {
    // GGUF-001: Tensor count limit
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    bytes.extend_from_slice(&3u32.to_le_bytes()); // Version
    bytes.extend_from_slice(&100_000u64.to_le_bytes()); // Huge tensor count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&bytes, None);
    
    // Must reject excessive tensor count
    assert!(matches!(result, Err(LoadError::TensorCountExceeded { .. })));
}

#[test]
fn test_path_traversal_dotdot() {
    let temp_dir = TempDir::new().unwrap();
    let loader = ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    
    // Create a file outside allowed root
    let outside_dir = TempDir::new().unwrap();
    let outside_file = outside_dir.path().join("secret.gguf");
    fs::write(&outside_file, b"secret data").unwrap();
    
    // Try to access via path traversal
    let traversal_path = temp_dir.path().join("../../../").join(&outside_file);
    
    let request = model_loader::LoadRequest::new(&traversal_path);
    let result = loader.load_and_validate(request);
    
    // Must reject path traversal
    assert!(result.is_err());
}

#[test]
fn test_symlink_escape() {
    let temp_dir = TempDir::new().unwrap();
    let loader = ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    
    // Create a file outside allowed root
    let outside_dir = TempDir::new().unwrap();
    let outside_file = outside_dir.path().join("secret.gguf");
    fs::write(&outside_file, b"secret data").unwrap();
    
    // Create symlink inside allowed root pointing outside
    let symlink_path = temp_dir.path().join("escape.gguf");
    unix_fs::symlink(&outside_file, &symlink_path).unwrap();
    
    let request = model_loader::LoadRequest::new(&symlink_path);
    let result = loader.load_and_validate(request);
    
    // Must reject symlink escape
    assert!(result.is_err());
}

#[test]
fn test_null_byte_injection() {
    use model_loader::validation::path;
    
    let temp_dir = TempDir::new().unwrap();
    
    // Path with null byte
    let path_str = format!("{}/model\0.gguf", temp_dir.path().display());
    let path = PathBuf::from(path_str);
    
    let result = path::validate_path(&path, temp_dir.path());
    
    // Must reject null byte
    assert!(result.is_err());
}

#[test]
fn test_hash_format_validation() {
    use model_loader::validation::hash;
    
    let bytes = b"test data";
    
    // Invalid hash: too short
    let result = hash::verify_hash(bytes, "abc123");
    assert!(result.is_err());
    
    // Invalid hash: non-hex characters
    let result = hash::verify_hash(bytes, &"z".repeat(64));
    assert!(result.is_err());
    
    // Invalid hash: wrong length
    let result = hash::verify_hash(bytes, &"a".repeat(63));
    assert!(result.is_err());
}

#[test]
fn test_hash_mismatch_rejection() {
    use model_loader::validation::hash;
    
    let bytes = b"test data";
    let wrong_hash = "0".repeat(64);
    
    let result = hash::verify_hash(bytes, &wrong_hash);
    
    // Must reject hash mismatch
    assert!(matches!(result, Err(LoadError::HashMismatch { .. })));
}

#[test]
fn test_resource_exhaustion_metadata_pairs() {
    // GGUF-004: Metadata pair limit
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    bytes.extend_from_slice(&3u32.to_le_bytes()); // Version
    bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
    bytes.extend_from_slice(&10_000u64.to_le_bytes()); // Huge metadata count
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&bytes, None);
    
    // Must reject excessive metadata pairs
    assert!(result.is_err());
}

#[test]
fn test_file_size_limit() {
    // Test file size limit enforcement via LoadRequest
    let temp_dir = TempDir::new().unwrap();
    let loader = ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    
    // Create a valid GGUF file
    let file_path = temp_dir.path().join("model.gguf");
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend(vec![0u8; 1000]); // Add some data (total: 1024 bytes)
    
    fs::write(&file_path, &bytes).unwrap();
    
    // Load with max_size smaller than file size
    let request = model_loader::LoadRequest::new(&file_path)
        .with_max_size(100);
    
    let result = loader.load_and_validate(request);
    
    // Must reject file too large (checked before path validation)
    assert!(result.is_err(), "Expected error for file too large, got: {:?}", result);
}

#[test]
fn test_invalid_magic_number() {
    let bytes = vec![0x00, 0x00, 0x00, 0x00]; // Wrong magic
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&bytes, None);
    
    // Must reject invalid magic
    assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
}

#[test]
fn test_invalid_version() {
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Valid magic
    bytes.extend_from_slice(&99u32.to_le_bytes()); // Invalid version
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&bytes, None);
    
    // Must reject invalid version
    assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
}

#[test]
fn test_file_too_small() {
    let bytes = vec![0x47, 0x47]; // Only 2 bytes
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&bytes, None);
    
    // Must reject file too small
    assert!(result.is_err());
}
