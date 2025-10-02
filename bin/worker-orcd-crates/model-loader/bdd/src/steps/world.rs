//! BDD World for model-loader tests

use cucumber::World;
use model_loader::{LoadError, ModelLoader};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, World)]
#[world(init = Self::new)]
pub struct BddWorld {
    /// ModelLoader instance
    pub loader: ModelLoader,

    /// Temporary directory for test files
    pub temp_dir: Option<tempfile::TempDir>,

    /// Path to test model file
    pub model_path: Option<PathBuf>,

    /// Model bytes (for in-memory validation)
    pub model_bytes: Option<Vec<u8>>,

    /// Expected hash for validation
    pub expected_hash: Option<String>,

    /// Load result (success or error)
    pub load_result: Option<Result<Vec<u8>, LoadError>>,

    /// Validation result (for validate_bytes)
    pub validation_result: Option<Result<(), LoadError>>,

    /// Test metadata
    pub metadata: HashMap<String, String>,
}

impl BddWorld {
    pub fn new() -> Self {
        Self {
            loader: ModelLoader::new(),
            temp_dir: None,
            model_path: None,
            model_bytes: None,
            expected_hash: None,
            load_result: None,
            validation_result: None,
            metadata: HashMap::new(),
        }
    }

    /// Helper: Create valid GGUF bytes
    pub fn create_valid_gguf() -> Vec<u8> {
        let mut gguf = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF" magic
        gguf.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        gguf.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
        gguf.extend_from_slice(&0u64.to_le_bytes()); // Metadata KV count
        gguf
    }

    /// Helper: Create invalid GGUF bytes
    pub fn create_invalid_gguf() -> Vec<u8> {
        vec![0x00, 0x00, 0x00, 0x00] // Invalid magic
    }

    /// Helper: Compute SHA-256 hash
    pub fn compute_hash(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }
}

impl Default for BddWorld {
    fn default() -> Self {
        Self::new()
    }
}
