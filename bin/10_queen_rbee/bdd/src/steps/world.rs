// TEAM-135: Created by TEAM-135 (BDD scaffolding)
// TEAM-156: Added hive catalog test state
//\! BDD World for queen-rbee integration tests

use cucumber::World;
use queen_rbee_hive_catalog::HiveCatalog;
use rbee_heartbeat::HiveHeartbeatPayload;
use std::path::PathBuf;
use std::process::Child;
use std::sync::Arc;
use tempfile::TempDir;
use wiremock::MockServer;

#[derive(World)]
#[derive(Default)]
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
    
    // TEAM-159: Mock HTTP server for device detection tests
    #[world(skip)] // MockServer doesn't implement Debug
    pub mock_server: Option<MockServer>,
    
    // TEAM-160: Process handles for integration tests
    #[world(skip)] // Child doesn't implement Debug
    pub queen_process: Option<Child>,
    #[world(skip)] // Child doesn't implement Debug
    pub hive_process: Option<Child>,
    pub queen_port: Option<u16>,
    pub hive_port: Option<u16>,
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
            .field("mock_server", &self.mock_server.as_ref().map(|_| "<MockServer>"))
            .field("queen_process", &self.queen_process.as_ref().map(|_| "<Child>"))
            .field("hive_process", &self.hive_process.as_ref().map(|_| "<Child>"))
            .field("queen_port", &self.queen_port)
            .field("hive_port", &self.hive_port)
            .finish()
    }
}

// TEAM-160: Cleanup spawned processes on drop
impl Drop for BddWorld {
    fn drop(&mut self) {
        if let Some(mut process) = self.queen_process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
        if let Some(mut process) = self.hive_process.take() {
            let _ = process.kill();
            let _ = process.wait();
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
