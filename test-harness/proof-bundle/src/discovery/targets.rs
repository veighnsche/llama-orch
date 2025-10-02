//! Test target representation

use std::path::PathBuf;

/// A discovered test target
#[derive(Debug, Clone)]
pub struct TestTarget {
    /// Target name (e.g., "my_crate")
    pub name: String,
    
    /// Target kind (e.g., ["lib"], ["test"])
    pub kinds: Vec<String>,
    
    /// Path to source file
    pub src_path: PathBuf,
    
    /// Package name
    pub package: String,

    /// Crate manifest directory (directory containing Cargo.toml)
    pub manifest_dir: PathBuf,
}

impl TestTarget {
    /// Check if this is a library target
    pub fn is_lib(&self) -> bool {
        self.kinds.iter().any(|k| k == "lib")
    }
    
    /// Check if this is a test target
    pub fn is_test(&self) -> bool {
        self.kinds.iter().any(|k| k == "test")
    }
    
    /// Check if this is a benchmark target
    pub fn is_bench(&self) -> bool {
        self.kinds.iter().any(|k| k == "bench")
    }
}
