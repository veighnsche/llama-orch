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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    // Hive operations
    // TEAM-186: Renamed create→install, delete→uninstall
    // TEAM-186: Added hive_id with default "localhost" to all operations
    // TEAM-187: Added SshTest for pre-installation SSH validation
    SshTest {
        ssh_host: String,
        #[serde(default = "default_ssh_port")]
        ssh_port: u16,
        ssh_user: String,
    },
    HiveInstall {
        hive_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_host: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_port: Option<u16>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_user: Option<String>,
        #[serde(default = "default_port")]
        port: u16,
        /// TEAM-187: Optional path where to install/find the hive binary
        /// If None, defaults to standard location (git clone + cargo build)
        /// Assumes rustup is installed on all systems (fail fast if not)
        #[serde(skip_serializing_if = "Option::is_none")]
        binary_path: Option<String>,
    },
    HiveUninstall {
        hive_id: String,
        /// If true, only remove from catalog (for unreachable remote hives)
        #[serde(default)]
        catalog_only: bool,
    },
    HiveUpdate {
        hive_id: String,
        /// Updated SSH host (if changed)
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_host: Option<String>,
        /// Updated SSH port (if changed)
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_port: Option<u16>,
        /// Updated SSH user (if changed)
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_user: Option<String>,
        /// If true, refresh device capabilities from hive
        #[serde(default)]
        refresh_capabilities: bool,
    },
    HiveStart {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
    HiveStop {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
    HiveList,
    HiveGet {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
    /// TEAM-189: Check hive health endpoint status
    HiveStatus {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },

    // Worker operations
    WorkerSpawn {
        hive_id: String,
        model: String,
        worker: String,
        device: u32,
    },
    WorkerList {
        hive_id: String,
    },
    WorkerGet {
        hive_id: String,
        id: String,
    },
    WorkerDelete {
        hive_id: String,
        id: String,
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

fn default_port() -> u16 {
    8600
}

// TEAM-187: Default SSH port for SshTest operation
fn default_ssh_port() -> u16 {
    22
}

impl Operation {
    /// Get the operation name as a string (for logging/narration)
    pub fn name(&self) -> &'static str {
        match self {
            Operation::SshTest { .. } => "ssh_test",
            Operation::HiveInstall { .. } => "hive_install",
            Operation::HiveUninstall { .. } => "hive_uninstall",
            Operation::HiveUpdate { .. } => "hive_update",
            Operation::HiveStart { .. } => "hive_start",
            Operation::HiveStop { .. } => "hive_stop",
            Operation::HiveList => "hive_list",
            Operation::HiveGet { .. } => "hive_get",
            Operation::HiveStatus { .. } => "hive_status",
            Operation::WorkerSpawn { .. } => "worker_spawn",
            Operation::WorkerList { .. } => "worker_list",
            Operation::WorkerGet { .. } => "worker_get",
            Operation::WorkerDelete { .. } => "worker_delete",
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
            Operation::HiveInstall { hive_id, .. } => Some(hive_id),
            Operation::HiveUninstall { hive_id, .. } => Some(hive_id),
            Operation::HiveUpdate { hive_id, .. } => Some(hive_id),
            Operation::HiveStart { hive_id } => Some(hive_id),
            Operation::HiveStop { hive_id } => Some(hive_id),
            Operation::HiveGet { hive_id } => Some(hive_id),
            Operation::HiveStatus { hive_id } => Some(hive_id),
            Operation::WorkerSpawn { hive_id, .. } => Some(hive_id),
            Operation::WorkerList { hive_id } => Some(hive_id),
            Operation::WorkerGet { hive_id, .. } => Some(hive_id),
            Operation::WorkerDelete { hive_id, .. } => Some(hive_id),
            Operation::ModelDownload { hive_id, .. } => Some(hive_id),
            Operation::ModelList { hive_id } => Some(hive_id),
            Operation::ModelGet { hive_id, .. } => Some(hive_id),
            Operation::ModelDelete { hive_id, .. } => Some(hive_id),
            Operation::Infer { hive_id, .. } => Some(hive_id),
            _ => None,
        }
    }
}

// ============================================================================
// OPERATION CONSTANTS (for backward compatibility)
// ============================================================================

/// Operation name constants
///
/// TEAM-186: Kept for backward compatibility with string-based code
pub mod constants {
    // TEAM-186: Renamed create→install, delete→uninstall
    pub const OP_HIVE_INSTALL: &str = "hive_install";
    pub const OP_HIVE_UNINSTALL: &str = "hive_uninstall";
    pub const OP_HIVE_UPDATE: &str = "hive_update";
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
        // TEAM-186: Test localhost install
        let op = Operation::HiveInstall {
            hive_id: "localhost".to_string(),
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            port: 8600,
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"hive_install"#));
        assert!(json.contains(r#""hive_id":"localhost"#));
        assert!(json.contains(r#""port":8600"#));
        assert!(!json.contains(r#""ssh_host""#)); // Should be omitted
    }

    #[test]
    fn test_serialize_hive_install_remote() {
        // TEAM-186: Test remote SSH install
        let op = Operation::HiveInstall {
            hive_id: "hive-prod-01".to_string(),
            ssh_host: Some("192.168.1.100".to_string()),
            ssh_port: Some(22),
            ssh_user: Some("admin".to_string()),
            port: 8600,
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""ssh_host":"192.168.1.100"#));
        assert!(json.contains(r#""ssh_port":22"#));
        assert!(json.contains(r#""ssh_user":"admin"#));
    }

    #[test]
    fn test_serialize_hive_uninstall() {
        // TEAM-186: Test uninstall with catalog_only flag
        let op = Operation::HiveUninstall { hive_id: "localhost".to_string(), catalog_only: false };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"hive_uninstall"#));
        assert!(json.contains(r#""hive_id":"localhost"#));
    }

    #[test]
    fn test_serialize_hive_update() {
        // TEAM-186: Test update with refresh_capabilities
        let op = Operation::HiveUpdate {
            hive_id: "localhost".to_string(),
            ssh_host: Some("192.168.1.101".to_string()),
            ssh_port: None,
            ssh_user: None,
            refresh_capabilities: true,
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"hive_update"#));
        assert!(json.contains(r#""refresh_capabilities":true"#));
    }

    #[test]
    fn test_serialize_hive_start() {
        let op = Operation::HiveStart { hive_id: "localhost".to_string() };
        let json = serde_json::to_string(&op).unwrap();
        assert_eq!(json, r#"{"operation":"hive_start","hive_id":"localhost"}"#);
    }

    #[test]
    fn test_hive_start_defaults_to_localhost() {
        // TEAM-186: Test that hive_id defaults to "localhost"
        let json = r#"{"operation":"hive_start"}"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        match op {
            Operation::HiveStart { hive_id } => {
                assert_eq!(hive_id, "localhost");
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
        assert_eq!(Operation::HiveStart { hive_id: "test".to_string() }.name(), "hive_start");
        // TEAM-186: Test new operation names
        assert_eq!(
            Operation::HiveInstall {
                hive_id: "test".to_string(),
                ssh_host: None,
                ssh_port: None,
                ssh_user: None,
                port: 8600,
            }
            .name(),
            "hive_install"
        );
        assert_eq!(
            Operation::HiveUninstall { hive_id: "test".to_string(), catalog_only: false }.name(),
            "hive_uninstall"
        );
    }

    #[test]
    fn test_operation_hive_id() {
        let op = Operation::HiveStart { hive_id: "localhost".to_string() };
        assert_eq!(op.hive_id(), Some("localhost"));

        let op = Operation::HiveList;
        assert_eq!(op.hive_id(), None);

        // TEAM-186: Test hive_id extraction for new operations
        let op = Operation::HiveInstall {
            hive_id: "hive-prod".to_string(),
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            port: 8600,
        };
        assert_eq!(op.hive_id(), Some("hive-prod"));
    }
}
