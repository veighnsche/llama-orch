// TEAM-135: Created by TEAM-135 (BDD scaffolding)
// TEAM-156: Added hive catalog test state
//\! BDD World for queen-rbee integration tests

use cucumber::World;
use std::path::PathBuf;
use tempfile::TempDir;

#[derive(Debug, World)]
pub struct BddWorld {
    /// Last validation result
    pub last_result: Option<Result<(), String>>,
    
    // TEAM-156: Hive catalog test state
    pub temp_dir: Option<TempDir>,
    pub catalog_path: Option<PathBuf>,
    pub hive_count: usize,
    
    // TODO: Add integration test state fields here
    // e.g., HTTP client, process handles, temp directories
}

impl Default for BddWorld {
    fn default() -> Self {
        Self {
            last_result: None,
            temp_dir: None,
            catalog_path: None,
            hive_count: 0,
        }
    }
}

impl BddWorld {
    /// Store validation result
    pub fn store_result(&mut self, result: Result<(), String>) {
        self.last_result = Some(result);
    }

    /// Check if last validation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }

    /// Check if last validation failed
    pub fn last_failed(&self) -> bool {
        matches!(self.last_result, Some(Err(_)))
    }
}
