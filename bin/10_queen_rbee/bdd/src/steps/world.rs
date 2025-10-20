// TEAM-135: Created by TEAM-135 (BDD scaffolding)
// TEAM-156: Added hive catalog test state
//\! BDD World for queen-rbee integration tests

use cucumber::World;
use queen_rbee_hive_catalog::HiveCatalog;
use rbee_heartbeat::HiveHeartbeatPayload;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

#[derive(World)]
pub struct BddWorld {
    /// Last validation result
    pub last_result: Option<Result<(), String>>,

    // TEAM-156: Hive catalog test state
    pub temp_dir: Option<TempDir>,
    pub catalog_path: Option<PathBuf>,
    pub hive_count: usize,

    // TEAM-158: Heartbeat test state
    #[world(skip)] // TEAM-158: HiveCatalog doesn't implement Debug
    pub hive_catalog: Option<Arc<HiveCatalog>>,
    pub current_hive_id: Option<String>,
    pub heartbeat_payload: Option<HiveHeartbeatPayload>,
    // TODO: Add integration test state fields here
    // e.g., HTTP client, process handles, temp directories
}

// TEAM-158: Manual Debug impl since HiveCatalog doesn't implement Debug
impl std::fmt::Debug for BddWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BddWorld")
            .field("last_result", &self.last_result)
            .field("temp_dir", &self.temp_dir)
            .field("catalog_path", &self.catalog_path)
            .field("hive_count", &self.hive_count)
            .field("hive_catalog", &self.hive_catalog.as_ref().map(|_| "<HiveCatalog>"))
            .field("current_hive_id", &self.current_hive_id)
            .field("heartbeat_payload", &self.heartbeat_payload)
            .finish()
    }
}

impl Default for BddWorld {
    fn default() -> Self {
        Self {
            last_result: None,
            temp_dir: None,
            catalog_path: None,
            hive_count: 0,
            hive_catalog: None,
            current_hive_id: None,
            heartbeat_payload: None,
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
