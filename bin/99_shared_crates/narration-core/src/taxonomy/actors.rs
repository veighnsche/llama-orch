// TEAM-300: Modular reorganization - Actor constants
//! Actor taxonomy for narration system
//!
//! Actors represent WHO performed an action in the system.

/// Core orchestration service
pub const ACTOR_ORCHESTRATORD: &str = "orchestratord";
/// GPU pool manager service
pub const ACTOR_POOL_MANAGERD: &str = "pool-managerd";
/// Worker daemon (inference service)
pub const ACTOR_WORKER_ORCD: &str = "worker-orcd";
/// Inference engine (llama.cpp, vLLM, etc.)
pub const ACTOR_INFERENCE_ENGINE: &str = "inference-engine";
/// VRAM residency manager
pub const ACTOR_VRAM_RESIDENCY: &str = "vram-residency";
/// Queen-rbee main service (TEAM-191: Added for queen-rbee operations)
pub const ACTOR_QUEEN_RBEE: &str = "👑 queen-rbee";
/// Queen router (job routing and operation dispatch) (TEAM-191: Added for job routing)
pub const ACTOR_QUEEN_ROUTER: &str = "👑 queen-router";

/// Extract service name from a module path string.
///
/// Used by the `#[narrate(...)]` macro to infer actor from module path.
///
/// # Examples
/// ```
/// use observability_narration_core::extract_service_name;
///
/// assert_eq!(extract_service_name("llama_orch::orchestratord::admission"), "orchestratord");
/// assert_eq!(extract_service_name("llama_orch::pool_managerd::spawn"), "pool-managerd");
/// assert_eq!(extract_service_name("llama_orch::worker_orcd::inference"), "worker-orcd");
/// assert_eq!(extract_service_name("unknown::path"), "unknown");
/// ```
pub fn extract_service_name(module_path: &str) -> &'static str {
    let parts: Vec<&str> = module_path.split("::").collect();

    // Look for known service names
    for part in &parts {
        match *part {
            "orchestratord" => return ACTOR_ORCHESTRATORD,
            "pool_managerd" => return ACTOR_POOL_MANAGERD,
            "worker_orcd" => return ACTOR_WORKER_ORCD,
            "vram_residency" => return ACTOR_VRAM_RESIDENCY,
            "inference_engine" => return ACTOR_INFERENCE_ENGINE,
            _ => continue,
        }
    }

    // Fallback: return "unknown"
    "unknown"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_service_name() {
        assert_eq!(
            extract_service_name("llama_orch::orchestratord::admission"),
            ACTOR_ORCHESTRATORD
        );
        assert_eq!(extract_service_name("llama_orch::pool_managerd::spawn"), ACTOR_POOL_MANAGERD);
        assert_eq!(extract_service_name("llama_orch::worker_orcd::inference"), ACTOR_WORKER_ORCD);
        assert_eq!(extract_service_name("llama_orch::vram_residency::seal"), ACTOR_VRAM_RESIDENCY);
        assert_eq!(extract_service_name("unknown::path"), "unknown");
    }
}
