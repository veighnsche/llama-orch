//! Operations contract for queen-rbee ↔ rbee-hive communication
//!
//! TEAM-186: Created to ensure type safety between client (rbee-keeper) and server (queen-rbee)
//! TEAM-284: Renamed from rbee-operations to operations-contract
//! TEAM-284: Added request/response types and API specification
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
// MODULES
// ============================================================================

/// Request types for operations
pub mod requests;

/// Response types for operations
pub mod responses;

/// API specification
pub mod api;

// Re-export commonly used types
pub use api::{HealthResponse, HiveApiSpec, JobResponse};
pub use requests::*;
pub use responses::*;

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
    
    /// TEAM-312: Deep narration test through queen job server (like self-check but via SSE)
    QueenCheck,
    
    /// TEAM-313: Deep narration test through hive job server (tests hive SSE streaming)
    HiveCheck {
        /// Hive alias (only "localhost" supported)
        #[serde(default = "default_hive_id")]
        alias: String,
    },

    // Hive operations
    // TEAM-278: DELETED HiveInstall, HiveUninstall, SshTest
    // TEAM-284: DELETED PackageSync, PackageStatus, PackageInstall, PackageUninstall, PackageValidate, PackageMigrate (SSH/remote operations removed)
    // TEAM-285: DELETED HiveStart, HiveStop (localhost-only, no lifecycle management)
    HiveList,
    HiveGet {
        /// Hive alias (only "localhost" supported)
        #[serde(default = "default_hive_id")]
        alias: String,
    },
    /// TEAM-189: Check hive health endpoint status
    HiveStatus {
        /// Hive alias (only "localhost" supported)
        #[serde(default = "default_hive_id")]
        alias: String,
    },
    /// TEAM-196: Refresh device capabilities for a running hive
    HiveRefreshCapabilities {
        /// Hive alias (only "localhost" supported)
        alias: String,
    },
    // TEAM-323: DELETED HiveInstall, HiveUninstall, HiveRebuild - use daemon-lifecycle directly (same pattern as queen)
    // TEAM-278: DELETED HiveImportSsh - not needed in declarative arch

    // TEAM-278: DELETED all worker binary operations
    // WorkerDownload, WorkerBuild, WorkerBinaryList, WorkerBinaryGet, WorkerBinaryDelete
    // These are replaced by PackageSync (TEAM-279 will add)

    // Worker process operations (hive-local)
    // TEAM-272: These manage worker PROCESSES on the hive
    // TEAM-284: Updated to use typed requests
    /// Spawn a worker process on hive
    WorkerSpawn(WorkerSpawnRequest),
    /// List worker processes running on hive (local ps, not registry)
    WorkerProcessList(WorkerProcessListRequest),
    /// Get details of a worker process on hive (local ps, not registry)
    WorkerProcessGet(WorkerProcessGetRequest),
    /// Delete (kill) a worker process on hive
    WorkerProcessDelete(WorkerProcessDeleteRequest),

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
    // TEAM-284: Updated to use typed requests
    ModelDownload(ModelDownloadRequest),
    ModelList(ModelListRequest),
    ModelGet(ModelGetRequest),
    ModelDelete(ModelDeleteRequest),

    // Inference operation
    // TEAM-284: Updated to use typed request
    Infer(InferRequest),
}

fn default_hive_id() -> String {
    "localhost".to_string()
}

// TEAM-278: DELETED default_ssh_config_path() and default_hive_port() - not needed

impl Operation {
    /// Get the operation name as a string (for logging/narration)
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Status => "status", // TEAM-190
            Operation::QueenCheck => "queen_check", // TEAM-312
            Operation::HiveCheck { .. } => "hive_check", // TEAM-313
            // TEAM-278: DELETED ssh_test, hive_install, hive_uninstall
            // TEAM-284: DELETED package_sync, package_status, package_install, package_uninstall, package_validate, package_migrate
            // TEAM-285: DELETED hive_start, hive_stop (localhost-only, no lifecycle management)
            Operation::HiveList => "hive_list",
            Operation::HiveGet { .. } => "hive_get",
            Operation::HiveStatus { .. } => "hive_status",
            Operation::HiveRefreshCapabilities { .. } => "hive_refresh_capabilities", // TEAM-196
            // TEAM-323: DELETED hive_install, hive_uninstall, hive_rebuild - use daemon-lifecycle directly
            // TEAM-278: DELETED hive_import_ssh, worker_download, worker_build, worker_binary_*
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
    /// TEAM-284: Updated to work with typed requests
    pub fn hive_id(&self) -> Option<&str> {
        match self {
            // TEAM-278: DELETED HiveInstall, HiveUninstall
            // TEAM-285: DELETED HiveStart, HiveStop (localhost-only, no lifecycle management)
            Operation::HiveCheck { alias } => Some(alias), // TEAM-313
            Operation::HiveGet { alias } => Some(alias),
            Operation::HiveStatus { alias } => Some(alias),
            Operation::HiveRefreshCapabilities { alias } => Some(alias), // TEAM-196
            // TEAM-323: DELETED HiveInstall, HiveUninstall, HiveRebuild - use daemon-lifecycle directly
            // TEAM-278: DELETED worker binary operations
            // Worker process operations (TEAM-284: typed requests)
            Operation::WorkerSpawn(req) => Some(&req.hive_id),
            Operation::WorkerProcessList(req) => Some(&req.hive_id),
            Operation::WorkerProcessGet(req) => Some(&req.hive_id),
            Operation::WorkerProcessDelete(req) => Some(&req.hive_id),
            // Model operations (TEAM-284: typed requests)
            Operation::ModelDownload(req) => Some(&req.hive_id),
            Operation::ModelList(req) => Some(&req.hive_id),
            Operation::ModelGet(req) => Some(&req.hive_id),
            Operation::ModelDelete(req) => Some(&req.hive_id),
            Operation::Infer(req) => Some(&req.hive_id),
            _ => None,
        }
    }

    /// Check if this operation should be forwarded to a hive
    ///
    /// TEAM-258: Consolidate hive-forwarding operations
    /// TEAM-272: Updated per corrected architecture
    /// TEAM-279: Updated to clarify package operations handled by queen
    ///
    /// **Forwarded to Hive (hive-local operations):**
    /// - Worker process operations - Spawn/list/kill worker processes on hive (local ps)
    /// - Model operations - Download/manage models on hive
    ///
    /// **Handled by Queen (orchestration):**
    /// - Package operations - Config-driven lifecycle (PackageSync/Status/Install/etc.)
    /// - Active worker operations - Query heartbeat registry (ActiveWorkerList/Get/Retire)
    /// - Infer - Scheduling and routing to active workers
    /// - Hive operations - Managed by queen (install/start/stop/list)
    ///
    /// This allows new operations to be added to rbee-hive without modifying queen-rbee.
    pub fn should_forward_to_hive(&self) -> bool {
        matches!(
            self,
            // TEAM-278: DELETED worker binary operations
            // Worker process operations (hive-local)
            // TEAM-284: Updated to match typed requests
            Operation::WorkerSpawn(_)
                | Operation::WorkerProcessList(_)
                | Operation::WorkerProcessGet(_)
                | Operation::WorkerProcessDelete(_)
                // Model operations (hive-local)
                | Operation::ModelDownload(_)
                | Operation::ModelList(_)
                | Operation::ModelGet(_)
                | Operation::ModelDelete(_)
        )
    }
}

// ============================================================================
// TEAM-312: DELETED OPERATION CONSTANTS MODULE (backwards compat trap)
// ============================================================================
//
// This module contained string constants like OP_HIVE_START, OP_WORKER_SPAWN, etc.
// marked "for backward compatibility with string-based code".
//
// ❌ PROBLEM: These constants were NEVER used anywhere in the codebase.
//    They created 28 lines of permanent technical debt that would need
//    to be maintained forever "just in case".
//
// ✅ SOLUTION: DELETED. The Operation enum's .name() method provides
//    the canonical string representation. No duplication needed.
//
// Pre-1.0 software is ALLOWED to break. See: RULE ZERO
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_hive_list() {
        let op = Operation::HiveList;
        let json = serde_json::to_string(&op).unwrap();
        assert_eq!(json, r#"{"operation":"hive_list"}"#);
    }

    // TEAM-278: DELETED tests for HiveInstall, HiveUninstall
    // TEAM-285: DELETED tests for HiveStart, HiveStop

    #[test]
    fn test_serialize_worker_spawn() {
        // TEAM-285: Updated to use typed request
        let op = Operation::WorkerSpawn(WorkerSpawnRequest {
            hive_id: "localhost".to_string(),
            model: "test-model".to_string(),
            worker: "cpu".to_string(),
            device: 0,
        });
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"worker_spawn"#));
        assert!(json.contains(r#""hive_id":"localhost"#));
        assert!(json.contains(r#""model":"test-model"#));
    }

    #[test]
    fn test_serialize_infer() {
        // TEAM-285: Updated to use typed request
        let op = Operation::Infer(InferRequest {
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
        });
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
        // TEAM-285: Updated to match typed request
        match op {
            Operation::WorkerSpawn(req) => {
                assert_eq!(req.hive_id, "localhost");
                assert_eq!(req.model, "test-model");
                assert_eq!(req.worker, "cpu");
                assert_eq!(req.device, 0);
            }
            _ => panic!("Wrong operation type"),
        }
    }

    #[test]
    fn test_operation_name() {
        assert_eq!(Operation::HiveList.name(), "hive_list");
        // TEAM-278: DELETED tests for deleted operations
        // TEAM-285: DELETED test for HiveStart
    }

    #[test]
    fn test_operation_hive_id() {
        // TEAM-285: Updated to use HiveGet instead of HiveStart
        let op = Operation::HiveGet { alias: "localhost".to_string() };
        assert_eq!(op.hive_id(), Some("localhost"));

        let op = Operation::HiveList;
        assert_eq!(op.hive_id(), None);

        // TEAM-278: DELETED test for HiveInstall
    }
}
