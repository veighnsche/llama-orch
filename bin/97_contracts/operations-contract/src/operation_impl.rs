//! Operation implementation methods
//!
//! TEAM-CLEANUP: Extracted from lib.rs to reduce clutter and improve maintainability

use super::*;

/// Target server for operation routing
///
/// TEAM-CLEANUP: Helps rbee-keeper determine which server to talk to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetServer {
    /// Send to queen-rbee (http://localhost:7833/v1/jobs)
    Queen,
    /// Send to rbee-hive (http://localhost:7835/v1/jobs)
    Hive,
}

impl Operation {
    /// Get the operation name as a string (for logging/narration)
    pub fn name(&self) -> &'static str {
        match self {
            // Queen operations
            Operation::Status => "status",
            Operation::Infer { .. } => "infer",
            
            // Hive operations - Worker lifecycle
            Operation::WorkerInstall { .. } => "worker_install", // TEAM-378: Worker binary installation
            Operation::WorkerSpawn { .. } => "worker_spawn",
            Operation::WorkerProcessList { .. } => "worker_process_list",
            Operation::WorkerProcessGet { .. } => "worker_process_get",
            Operation::WorkerProcessDelete { .. } => "worker_process_delete",
            
            // Hive operations - Model management
            Operation::ModelDownload { .. } => "model_download",
            Operation::ModelList { .. } => "model_list",
            Operation::ModelGet { .. } => "model_get",
            Operation::ModelDelete { .. } => "model_delete",
            Operation::ModelLoad { .. } => "model_load",
            Operation::ModelUnload { .. } => "model_unload",
            
            // RHAI script operations
            Operation::RhaiScriptSave { .. } => "rhai_script_save",
            Operation::RhaiScriptTest { .. } => "rhai_script_test",
            Operation::RhaiScriptGet { .. } => "rhai_script_get",
            Operation::RhaiScriptList => "rhai_script_list",
            Operation::RhaiScriptDelete { .. } => "rhai_script_delete",
            
            // Diagnostic operations
            Operation::QueenCheck => "queen_check",
            Operation::HiveCheck { .. } => "hive_check",
        }
    }

    /// Get the hive_id if the operation targets a specific hive
    pub fn hive_id(&self) -> Option<&str> {
        match self {
            // Operations with hive_id in typed requests
            Operation::WorkerInstall(req) => Some(&req.hive_id), // TEAM-378: Worker binary installation
            Operation::WorkerSpawn(req) => Some(&req.hive_id),
            Operation::WorkerProcessList(req) => Some(&req.hive_id),
            Operation::WorkerProcessGet(req) => Some(&req.hive_id),
            Operation::WorkerProcessDelete(req) => Some(&req.hive_id),
            Operation::ModelDownload(req) => Some(&req.hive_id),
            Operation::ModelList(req) => Some(&req.hive_id),
            Operation::ModelGet(req) => Some(&req.hive_id),
            Operation::ModelDelete(req) => Some(&req.hive_id),
            Operation::ModelLoad(req) => Some(&req.hive_id),
            Operation::ModelUnload(req) => Some(&req.hive_id),
            Operation::Infer(req) => Some(&req.hive_id),
            
            // Operations with alias field
            Operation::HiveCheck { alias } => Some(alias),
            
            // RHAI operations don't target specific hives
            Operation::RhaiScriptSave { .. }
            | Operation::RhaiScriptTest { .. }
            | Operation::RhaiScriptGet { .. }
            | Operation::RhaiScriptList
            | Operation::RhaiScriptDelete { .. } => None,
            
            // Operations without hive_id
            _ => None,
        }
    }

    /// Get the target server for this operation
    ///
    /// TEAM-CLEANUP: Replaced should_forward_to_hive() with target_server()
    /// 
    /// **Architecture:**
    /// - rbee-keeper talks directly to BOTH queen AND hive (NO proxying)
    /// - Queen handles: Status, Infer (orchestration)
    /// - Hive handles: Worker/Model lifecycle operations
    ///
    /// **Queen Operations (http://localhost:7833/v1/jobs):**
    /// - Status - Query registries
    /// - Infer - Scheduling and routing to workers
    ///
    /// **Hive Operations (http://localhost:7835/v1/jobs):**
    /// - Worker process operations - Spawn/list/kill worker processes
    /// - Model operations - Download/manage models
    ///
    /// This method helps rbee-keeper determine which server to send the operation to.
    pub fn target_server(&self) -> TargetServer {
        match self {
            // Queen operations (orchestration)
            Operation::Status | Operation::Infer(_) => TargetServer::Queen,
            
            // Hive operations (worker/model lifecycle)
            Operation::WorkerInstall(_) // TEAM-378: Worker binary installation
                | Operation::WorkerSpawn(_)
                | Operation::WorkerProcessList(_)
                | Operation::WorkerProcessGet(_)
                | Operation::WorkerProcessDelete(_)
                | Operation::ModelDownload(_)
                | Operation::ModelList(_)
                | Operation::ModelGet(_)
                | Operation::ModelDelete(_)
                | Operation::ModelLoad(_)
                | Operation::ModelUnload(_) => TargetServer::Hive,
            
            // RHAI operations go to queen (orchestration layer)
            Operation::RhaiScriptSave { .. }
                | Operation::RhaiScriptTest { .. }
                | Operation::RhaiScriptGet { .. }
                | Operation::RhaiScriptList
                | Operation::RhaiScriptDelete { .. } => TargetServer::Queen,
            
            // Everything else goes to queen
            _ => TargetServer::Queen,
        }
    }
}
