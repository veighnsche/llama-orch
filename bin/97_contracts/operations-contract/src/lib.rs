//! Operations contract for queen-rbee ↔ rbee-hive communication
//!
//! TEAM-186: Created to ensure type safety between client (rbee-keeper) and server (queen-rbee)
//! TEAM-284: Renamed from rbee-operations to operations-contract
//! TEAM-CLEANUP: Reorganized for clarity and maintainability
//!
//! # Quick Reference
//!
//! ## Queen Operations (2)
//! - `Status` - Query hive and worker registries
//! - `Infer` - Schedule inference and route to worker
//!
//! ## Hive Operations (8)
//! **Worker Lifecycle:** WorkerSpawn, WorkerProcessList, WorkerProcessGet, WorkerProcessDelete  
//! **Model Management:** ModelDownload, ModelList, ModelGet, ModelDelete
//!
//! ## Diagnostic (2)
//! - `QueenCheck` - Test queen SSE streaming
//! - `HiveCheck` - Test hive SSE streaming
//!
//! **Total:** 12 operations
//!
//! # Architecture
//!
//! ```text
//! rbee-keeper
//!     ↓
//! Operation enum
//!     ↓
//!     ├─→ Queen Operations (http://localhost:7833/v1/jobs)
//!     │   - Status, Infer
//!     │
//!     └─→ Hive Operations (http://localhost:7835/v1/jobs)
//!         - Worker/Model lifecycle
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

/// Operation implementation methods (TEAM-CLEANUP: extracted from lib.rs)
pub mod operation_impl;

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
    // ═══════════════════════════════════════════════════════════════════════
    // QUEEN OPERATIONS (http://localhost:7833/v1/jobs)
    // ═══════════════════════════════════════════════════════════════════════
    /// Query all hives and workers from registry
    Status,

    /// Schedule inference and route to worker
    Infer(InferRequest),

    // Image Generation Operations (TEAM-397)
    // ───────────────────────────────────────────────────────────────────────
    /// Generate image from text prompt (Stable Diffusion)
    ImageGeneration(ImageGenerationRequest),
    /// Transform image (img2img)
    ImageTransform(ImageTransformRequest),
    /// Inpaint image with mask
    ImageInpaint(ImageInpaintRequest),

    // RHAI Script Management
    // ───────────────────────────────────────────────────────────────────────
    /// Save a RHAI script
    RhaiScriptSave {
        name: String,
        content: String,
        id: Option<String>,
    },
    /// Test a RHAI script
    RhaiScriptTest {
        content: String,
    },
    /// Get a RHAI script by ID
    RhaiScriptGet {
        id: String,
    },
    /// List all RHAI scripts
    RhaiScriptList,
    /// Delete a RHAI script
    RhaiScriptDelete {
        id: String,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // HIVE OPERATIONS (http://localhost:7835/v1/jobs)
    // ═══════════════════════════════════════════════════════════════════════

    // Worker Lifecycle
    // ───────────────────────────────────────────────────────────────────────
    /// List available workers from catalog server (Hono)
    /// TEAM-388: Queries http://localhost:8787/workers
    WorkerCatalogList(WorkerCatalogListRequest),
    /// Get worker details from catalog server (Hono)
    /// TEAM-388: Queries http://localhost:8787/workers/:id
    WorkerCatalogGet(WorkerCatalogGetRequest),
    /// List installed worker binaries on hive (from worker catalog)
    WorkerListInstalled(WorkerListInstalledRequest),
    /// Get details of a specific installed worker binary
    /// TEAM-388: Shows details from ~/.cache/rbee/workers/
    WorkerInstalledGet(WorkerCatalogGetRequest),
    /// Install a worker binary from catalog (download PKGBUILD, build, install)
    WorkerInstall(WorkerInstallRequest),
    /// Remove an installed worker binary
    /// TEAM-388: Removes from ~/.cache/rbee/workers/
    WorkerRemove(WorkerRemoveRequest),
    /// Spawn a worker process on hive
    WorkerSpawn(WorkerSpawnRequest),
    /// List worker processes running on hive (local ps, not registry)
    WorkerProcessList(WorkerProcessListRequest),
    /// Get details of a worker process on hive (local ps, not registry)
    WorkerProcessGet(WorkerProcessGetRequest),
    /// Delete (kill) a worker process on hive
    WorkerProcessDelete(WorkerProcessDeleteRequest),

    // Model Management
    // ───────────────────────────────────────────────────────────────────────
    ModelDownload(ModelDownloadRequest),
    ModelList(ModelListRequest),
    ModelGet(ModelGetRequest),
    ModelDelete(ModelDeleteRequest),
    ModelLoad(ModelLoadRequest),
    ModelUnload(ModelUnloadRequest),

    // ═══════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════
    /// Deep narration test through queen job server
    QueenCheck,

    /// Deep narration test through hive job server
    HiveCheck {
        #[serde(default = "default_hive_id")]
        alias: String,
    },
}

fn default_hive_id() -> String {
    "localhost".to_string()
}

// Re-export TargetServer from operation_impl for convenience
pub use operation_impl::TargetServer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_status() {
        let op = Operation::Status;
        let json = serde_json::to_string(&op).unwrap();
        assert_eq!(json, r#"{"operation":"status"}"#);
    }

    #[test]
    fn test_serialize_worker_spawn() {
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
    fn test_deserialize_status() {
        let json = r#"{"operation":"status"}"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        assert_eq!(op, Operation::Status);
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
        assert_eq!(Operation::Status.name(), "status");
        assert_eq!(Operation::QueenCheck.name(), "queen_check");
    }

    #[test]
    fn test_operation_hive_id() {
        let op = Operation::HiveCheck { alias: "localhost".to_string() };
        assert_eq!(op.hive_id(), Some("localhost"));

        let op = Operation::Status;
        assert_eq!(op.hive_id(), None);
    }

    // TEAM-397/398: Tests for image generation operations
    #[test]
    fn test_serialize_image_generation() {
        let op = Operation::ImageGeneration(ImageGenerationRequest {
            hive_id: "localhost".to_string(),
            model: "stable-diffusion-v1-5".to_string(),
            prompt: "test prompt".to_string(),
            negative_prompt: None,
            steps: 20,
            guidance_scale: 7.5,
            width: 512,
            height: 512,
            seed: None,
            loras: vec![], // Empty loras vector
            worker_id: None,
        });
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains(r#""operation":"image_generation"#));
        assert!(json.contains(r#""prompt":"test prompt"#));
    }

    #[test]
    fn test_deserialize_image_generation() {
        let json = r#"{
            "operation": "image_generation",
            "hive_id": "localhost",
            "model": "stable-diffusion-v1-5",
            "prompt": "test",
            "steps": 20,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        match op {
            Operation::ImageGeneration(req) => {
                assert_eq!(req.hive_id, "localhost");
                assert_eq!(req.prompt, "test");
                assert_eq!(req.steps, 20);
            }
            _ => panic!("Wrong operation type"),
        }
    }

    #[test]
    fn test_image_operation_names() {
        let op1 = Operation::ImageGeneration(ImageGenerationRequest {
            hive_id: "localhost".to_string(),
            model: "sd".to_string(),
            prompt: "test".to_string(),
            negative_prompt: None,
            steps: 20,
            guidance_scale: 7.5,
            width: 512,
            height: 512,
            seed: None,
            loras: vec![], // Empty loras vector
            worker_id: None,
        });
        assert_eq!(op1.name(), "image_generation");

        let op2 = Operation::ImageTransform(ImageTransformRequest {
            hive_id: "localhost".to_string(),
            model: "sd".to_string(),
            prompt: "test".to_string(),
            negative_prompt: None,
            input_image: "base64".to_string(),
            strength: 0.8,
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
            loras: vec![], // Empty loras vector
            worker_id: None,
        });
        assert_eq!(op2.name(), "image_transform");

        let op3 = Operation::ImageInpaint(ImageInpaintRequest {
            hive_id: "localhost".to_string(),
            model: "sd".to_string(),
            prompt: "test".to_string(),
            negative_prompt: None,
            init_image: "base64".to_string(),
            mask_image: "mask".to_string(),
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
            loras: vec![], // Empty loras vector
            worker_id: None,
        });
        assert_eq!(op3.name(), "image_inpaint");
    }

    #[test]
    fn test_image_operation_target_server() {
        use crate::operation_impl::TargetServer;

        let op = Operation::ImageGeneration(ImageGenerationRequest {
            hive_id: "localhost".to_string(),
            model: "sd".to_string(),
            prompt: "test".to_string(),
            negative_prompt: None,
            steps: 20,
            guidance_scale: 7.5,
            width: 512,
            height: 512,
            seed: None,
            loras: vec![], // Empty loras vector
            worker_id: None,
        });
        assert_eq!(op.target_server(), TargetServer::Queen);
    }
}
