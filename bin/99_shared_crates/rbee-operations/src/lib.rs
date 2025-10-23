//! Shared operation types for rbee-keeper ↔ queen-rbee contract
//!
//! TEAM-186: Created to ensure type safety between client (rbee-keeper) and server (queen-rbee)
//!
//! This crate defines:
//! - Operation enum (all supported operations)
//! - Request/Response types for each operation
//! - Serialization/deserialization for JSON payloads
//!
//! # Architecture
//!
//! ```text
//! rbee-keeper (client) -> POST /v1/jobs -> queen-rbee (server)
//!                         |
//!                    JSON payload with "operation" field
//!                         |
//!                    Parsed into Operation enum
//!                         |
//!                    Routed to appropriate handler
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// OPERATION ENUM
// ============================================================================

/// All supported operations in the rbee system
///
/// TEAM-186: Single source of truth for operation types
/// TEAM-190: Added Status operation for live hive/worker overview
///
/// # Adding a New Operation (3-File Pattern)
///
/// When adding a new operation, you MUST update these 3 files:
///
/// 1. **THIS FILE** (rbee-operations/src/lib.rs):
///    - Add variant to Operation enum (line ~34)
///    - Add case to Operation::name() (line ~148)
///    - Add case to Operation::hive_id() if needed (line ~173)
///    - Add constant to constants module if needed (line ~204)
///
/// 2. **job_router.rs** (queen-rbee/src/job_router.rs):
///    - Add match arm in route_operation() (line ~132)
///    - Import any new request types from lifecycle crates
///
/// 3. **main.rs** (rbee-keeper/src/main.rs):
///    - Add CLI command variant (Commands/HiveAction/WorkerAction/etc.)
///    - Add match arm in handle_command() to construct Operation
///
/// See existing operations below for examples.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    // System-wide operations
    /// TEAM-190: Show live status of all hives and workers from registry
    Status,

    // Hive operations
    // TEAM-186: Renamed create→install, delete→uninstall
    // TEAM-194: Simplified to alias-based lookups (config from hives.conf)
    SshTest {
        /// Alias from hives.conf
        alias: String,
    },
    HiveInstall {
        /// Alias from hives.conf (must exist before install)
        alias: String,
    },
    HiveUninstall {
        /// Alias from hives.conf
        alias: String,
    },
    HiveStart {
        /// Alias from hives.conf (defaults to "localhost")
        #[serde(default = "default_hive_id")]
        alias: String,
    },
    HiveStop {
        /// Alias from hives.conf (defaults to "localhost")
        #[serde(default = "default_hive_id")]
        alias: String,
    },
    HiveList,
    HiveGet {
        /// Alias from hives.conf (defaults to "localhost")
        #[serde(default = "default_hive_id")]
        alias: String,
    },
    /// TEAM-189: Check hive health endpoint status
    HiveStatus {
        /// Alias from hives.conf (defaults to "localhost")
        #[serde(default = "default_hive_id")]
        alias: String,
    },
    /// TEAM-196: Refresh device capabilities for a running hive
    HiveRefreshCapabilities {
        /// Alias from hives.conf
        alias: String,
    },
    /// Import SSH config into hives.conf
    HiveImportSsh {
        /// Path to SSH config file (defaults to ~/.ssh/config)
        #[serde(default = "default_ssh_config_path")]
        ssh_config_path: String,
        /// Default HivePort for all imported hosts
        #[serde(default = "default_hive_port")]
        default_hive_port: u16,
    },

    // Worker binary operations (hive-local)
    // TEAM-272: These manage worker BINARIES on the hive, not running workers
    /// Download worker binary to hive
    WorkerDownload {
        hive_id: String,
        worker_type: String, // e.g., "cuda-llm-worker", "cpu-llm-worker"
    },
    /// Build worker binary on hive
    WorkerBuild {
        hive_id: String,
        worker_type: String,
    },
    /// List worker binaries available on hive
    WorkerBinaryList {
        hive_id: String,
    },
    /// Get details of a worker binary on hive
    WorkerBinaryGet {
        hive_id: String,
        worker_type: String,
    },
    /// Delete worker binary from hive
    WorkerBinaryDelete {
        hive_id: String,
        worker_type: String,
    },

    // Worker process operations (hive-local)
    // TEAM-272: These manage worker PROCESSES on the hive
    /// Spawn a worker process on hive
    WorkerSpawn {
        hive_id: String,
        model: String,
        worker: String,
        device: u32,
    },
    /// List worker processes running on hive (local ps, not registry)
    WorkerProcessList {
        hive_id: String,
    },
    /// Get details of a worker process on hive (local ps, not registry)
    WorkerProcessGet {
        hive_id: String,
        pid: u32,
    },
    /// Delete (kill) a worker process on hive
    WorkerProcessDelete {
        hive_id: String,
        pid: u32,
    },

    // Active worker operations (queen-tracked)
    // TEAM-272: These query queen's registry of workers sending heartbeats
    /// List active workers (from queen's heartbeat registry)
    ActiveWorkerList,
    /// Get details of an active worker (from queen's registry)
    ActiveWorkerGet {
        worker_id: String,
    },
    /// Retire an active worker (stop accepting new requests)
    ActiveWorkerRetire {
        worker_id: String,
    },

    // Model operations
    ModelDownload {
        hive_id: String,
        model: String,
    },
    ModelList {
        hive_id: String,
    },
    ModelGet {
        hive_id: String,
        id: String,
    },
    ModelDelete {
        hive_id: String,
        id: String,
    },

    // Inference operation
    Infer {
        hive_id: String,
        model: String,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        top_p: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        top_k: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        device: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        worker_id: Option<String>,
        #[serde(default = "default_stream")]
        stream: bool,
    },
}

fn default_stream() -> bool {
    true
}

fn default_hive_id() -> String {
    "localhost".to_string()
}

fn default_ssh_config_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    format!("{}/.ssh/config", home)
}

fn default_hive_port() -> u16 {
    8081
}

impl Operation {
    /// Get the operation name as a string (for logging/narration)
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Status => "status", // TEAM-190
            Operation::SshTest { .. } => "ssh_test",
            Operation::HiveInstall { .. } => "hive_install",
            Operation::HiveUninstall { .. } => "hive_uninstall",
            Operation::HiveStart { .. } => "hive_start",
            Operation::HiveStop { .. } => "hive_stop",
            Operation::HiveList => "hive_list",
            Operation::HiveGet { .. } => "hive_get",
            Operation::HiveStatus { .. } => "hive_status",
            Operation::HiveRefreshCapabilities { .. } => "hive_refresh_capabilities", // TEAM-196
            Operation::HiveImportSsh { .. } => "hive_import_ssh",
            // Worker binary operations
            Operation::WorkerDownload { .. } => "worker_download",
            Operation::WorkerBuild { .. } => "worker_build",
            Operation::WorkerBinaryList { .. } => "worker_binary_list",
            Operation::WorkerBinaryGet { .. } => "worker_binary_get",
            Operation::WorkerBinaryDelete { .. } => "worker_binary_delete",
            // Worker process operations
            Operation::WorkerSpawn { .. } => "worker_spawn",
            Operation::WorkerProcessList { .. } => "worker_process_list",
            Operation::WorkerProcessGet { .. } => "worker_process_get",
            Operation::WorkerProcessDelete { .. } => "worker_process_delete",
            // Active worker operations
            Operation::ActiveWorkerList => "active_worker_list",
            Operation::ActiveWorkerGet { .. } => "active_worker_get",
            Operation::ActiveWorkerRetire { .. } => "active_worker_retire",
            Operation::ModelDownload { .. } => "model_download",
            Operation::ModelList { .. } => "model_list",
            Operation::ModelGet { .. } => "model_get",
            Operation::ModelDelete { .. } => "model_delete",
            Operation::Infer { .. } => "infer",
        }
    }

    /// Get the hive_id if the operation targets a specific hive
    pub fn hive_id(&self) -> Option<&str> {
        match self {
            Operation::HiveInstall { alias } => Some(alias),
            Operation::HiveUninstall { alias } => Some(alias),
            Operation::HiveStart { alias } => Some(alias),
            Operation::HiveStop { alias } => Some(alias),
            Operation::HiveGet { alias } => Some(alias),
            Operation::HiveStatus { alias } => Some(alias),
            Operation::HiveRefreshCapabilities { alias } => Some(alias), // TEAM-196
            // Worker binary operations
            Operation::WorkerDownload { hive_id, .. } => Some(hive_id),
            Operation::WorkerBuild { hive_id, .. } => Some(hive_id),
            Operation::WorkerBinaryList { hive_id } => Some(hive_id),
            Operation::WorkerBinaryGet { hive_id, .. } => Some(hive_id),
            Operation::WorkerBinaryDelete { hive_id, .. } => Some(hive_id),
            // Worker process operations
            Operation::WorkerSpawn { hive_id, .. } => Some(hive_id),
            Operation::WorkerProcessList { hive_id } => Some(hive_id),
            Operation::WorkerProcessGet { hive_id, .. } => Some(hive_id),
            Operation::WorkerProcessDelete { hive_id, .. } => Some(hive_id),
            // Model operations
            Operation::ModelDownload { hive_id, .. } => Some(hive_id),
            Operation::ModelList { hive_id } => Some(hive_id),
            Operation::ModelGet { hive_id, .. } => Some(hive_id),
            Operation::ModelDelete { hive_id, .. } => Some(hive_id),
            Operation::Infer { hive_id, .. } => Some(hive_id),
            _ => None,
        }
    }

    /// Check if this operation should be forwarded to a hive
    ///
    /// TEAM-258: Consolidate hive-forwarding operations
    /// TEAM-272: Updated per corrected architecture
    ///
    /// **Forwarded to Hive (hive-local operations):**
    /// - Worker binary operations - Download/build/manage worker binaries on hive
    /// - Worker process operations - Spawn/list/kill worker processes on hive (local ps)
    /// - Model operations - Download/manage models on hive
    ///
    /// **Handled by Queen (orchestration):**
    /// - Active worker operations - Query heartbeat registry (ActiveWorkerList/Get/Retire)
    /// - Infer - Scheduling and routing to active workers
    /// - Hive operations - Managed by queen (install/start/stop/list)
    ///
    /// This allows new operations to be added to rbee-hive without modifying queen-rbee.
    pub fn should_forward_to_hive(&self) -> bool {
        matches!(
            self,
            // Worker binary operations (hive-local)
            Operation::WorkerDownload { .. }
                | Operation::WorkerBuild { .. }
                | Operation::WorkerBinaryList { .. }
                | Operation::WorkerBinaryGet { .. }
                | Operation::WorkerBinaryDelete { .. }
                // Worker process operations (hive-local)
                | Operation::WorkerSpawn { .. }
                | Operation::WorkerProcessList { .. }
                | Operation::WorkerProcessGet { .. }
                | Operation::WorkerProcessDelete { .. }
                // Model operations (hive-local)
                | Operation::ModelDownload { .. }
                | Operation::ModelList { .. }
                | Operation::ModelGet { .. }
                | Operation::ModelDelete { .. }
        )
    }
}

// ============================================================================
// OPERATION CONSTANTS (for backward compatibility)
// ============================================================================

/// Operation name constants
///
/// TEAM-186: Kept for backward compatibility with string-based code
/// TEAM-194: Removed OP_HIVE_UPDATE (operation removed)
pub mod constants {
    // TEAM-186: Renamed create→install, delete→uninstall
    pub const OP_HIVE_INSTALL: &str = "hive_install";
    pub const OP_HIVE_UNINSTALL: &str = "hive_uninstall";
    pub const OP_HIVE_START: &str = "hive_start";
    pub const OP_HIVE_STOP: &str = "hive_stop";
    pub const OP_HIVE_LIST: &str = "hive_list";
    pub const OP_HIVE_GET: &str = "hive_get";
    pub const OP_HIVE_STATUS: &str = "hive_status"; // TEAM-189

    pub const OP_WORKER_SPAWN: &str = "worker_spawn";
    pub const OP_WORKER_LIST: &str = "worker_list";
    pub const OP_WORKER_GET: &str = "worker_get";
    pub const OP_WORKER_DELETE: &str = "worker_delete";

    pub const OP_MODEL_DOWNLOAD: &str = "model_download";
    pub const OP_MODEL_LIST: &str = "model_list";
    pub const OP_MODEL_GET: &str = "model_get";
    pub const OP_MODEL_DELETE: &str = "model_delete";

    pub const OP_INFER: &str = "infer";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_hive_list() {
        let op = Operation::HiveList;
        let json = serde_json::to_string(&op).unwrap();
        assert_eq!(json, r#"{"operation":"hive_list"}"#);
    }

    #[test]
    fn test_serialize_hive_install() {
        // TEAM-194: Test alias-based install
        let op = Operation::HiveInstall { alias: "localhost".to_string() };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"hive_install"#));
        assert!(json.contains(r#""alias":"localhost"#));
    }

    #[test]
    fn test_serialize_hive_install_remote() {
        // TEAM-194: Test remote alias install
        let op = Operation::HiveInstall { alias: "workstation".to_string() };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"hive_install"#));
        assert!(json.contains(r#""alias":"workstation"#));
    }

    #[test]
    fn test_serialize_hive_uninstall() {
        // TEAM-194: Test alias-based uninstall
        let op = Operation::HiveUninstall { alias: "localhost".to_string() };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"hive_uninstall"#));
        assert!(json.contains(r#""alias":"localhost"#));
    }

    #[test]
    fn test_serialize_hive_start() {
        let op = Operation::HiveStart { alias: "localhost".to_string() };
        let json = serde_json::to_string(&op).unwrap();
        assert_eq!(json, r#"{"operation":"hive_start","alias":"localhost"}"#);
    }

    #[test]
    fn test_hive_start_defaults_to_localhost() {
        // TEAM-194: Test that alias defaults to "localhost"
        let json = r#"{"operation":"hive_start"}"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        match op {
            Operation::HiveStart { alias } => {
                assert_eq!(alias, "localhost");
            }
            _ => panic!("Wrong operation type"),
        }
    }

    #[test]
    fn test_serialize_worker_spawn() {
        let op = Operation::WorkerSpawn {
            hive_id: "localhost".to_string(),
            model: "test-model".to_string(),
            worker: "cpu".to_string(),
            device: 0,
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"worker_spawn"#));
        assert!(json.contains(r#""hive_id":"localhost"#));
        assert!(json.contains(r#""model":"test-model"#));
    }

    #[test]
    fn test_serialize_infer() {
        let op = Operation::Infer {
            hive_id: "localhost".to_string(),
            model: "test-model".to_string(),
            prompt: "hello".to_string(),
            max_tokens: 20,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
            device: None,
            worker_id: None,
            stream: true,
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"infer"#));
        assert!(json.contains(r#""prompt":"hello"#));
        assert!(json.contains(r#""top_p":0.9"#));
        assert!(!json.contains(r#""top_k""#)); // Should be omitted
    }

    #[test]
    fn test_deserialize_hive_list() {
        let json = r#"{"operation":"hive_list"}"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        assert_eq!(op, Operation::HiveList);
    }

    #[test]
    fn test_deserialize_worker_spawn() {
        let json = r#"{
            "operation": "worker_spawn",
            "hive_id": "localhost",
            "model": "test-model",
            "worker": "cpu",
            "device": 0
        }"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        match op {
            Operation::WorkerSpawn { hive_id, model, worker, device } => {
                assert_eq!(hive_id, "localhost");
                assert_eq!(model, "test-model");
                assert_eq!(worker, "cpu");
                assert_eq!(device, 0);
            }
            _ => panic!("Wrong operation type"),
        }
    }

    #[test]
    fn test_operation_name() {
        assert_eq!(Operation::HiveList.name(), "hive_list");
        assert_eq!(Operation::HiveStart { alias: "test".to_string() }.name(), "hive_start");
        // TEAM-194: Test alias-based operation names
        assert_eq!(Operation::HiveInstall { alias: "test".to_string() }.name(), "hive_install");
        assert_eq!(Operation::HiveUninstall { alias: "test".to_string() }.name(), "hive_uninstall");
    }

    #[test]
    fn test_operation_hive_id() {
        let op = Operation::HiveStart { alias: "localhost".to_string() };
        assert_eq!(op.hive_id(), Some("localhost"));

        let op = Operation::HiveList;
        assert_eq!(op.hive_id(), None);

        // TEAM-194: Test alias extraction for new operations
        let op = Operation::HiveInstall { alias: "hive-prod".to_string() };
        assert_eq!(op.hive_id(), Some("hive-prod"));
    }
}
