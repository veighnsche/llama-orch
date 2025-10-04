/// Infer actor name from module path.
/// 
/// Extracts service name from module path:
/// - `llama_orch::orchestratord::*` → "orchestratord"
/// - `llama_orch::pool_managerd::*` → "pool-managerd"
/// - `llama_orch::worker_orcd::*` → "worker-orcd"
/// 
/// Falls back to "unknown" if pattern doesn't match.
pub fn infer_actor_from_module() -> String {
    // Get the module path at compile time
    // This is a simplified version - in real implementation, we'd use
    // proc_macro::Span::call_site() to get the actual module path
    
    // For now, return a placeholder that will be replaced with actual logic
    "unknown".to_string()
}

/// Extract service name from a module path string.
/// 
/// # Examples
/// ```
/// use observability_narration_macros::actor_inference::extract_service_name;
/// 
/// assert_eq!(extract_service_name("llama_orch::orchestratord::admission"), "orchestratord");
/// assert_eq!(extract_service_name("llama_orch::pool_managerd::spawn"), "pool-managerd");
/// assert_eq!(extract_service_name("llama_orch::worker_orcd::inference"), "worker-orcd");
/// ```
pub fn extract_service_name(module_path: &str) -> &str {
    let parts: Vec<&str> = module_path.split("::").collect();
    
    // Look for known service names
    for part in &parts {
        match *part {
            "orchestratord" => return "orchestratord",
            "pool_managerd" => return "pool-managerd",
            "worker_orcd" => return "worker-orcd",
            "vram_residency" => return "vram-residency",
            _ => continue,
        }
    }
    
    // Fallback: use second component if available
    if parts.len() >= 2 {
        parts[1]
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_service_name() {
        assert_eq!(extract_service_name("llama_orch::orchestratord::admission"), "orchestratord");
        assert_eq!(extract_service_name("llama_orch::pool_managerd::spawn"), "pool-managerd");
        assert_eq!(extract_service_name("llama_orch::worker_orcd::inference"), "worker-orcd");
        assert_eq!(extract_service_name("llama_orch::vram_residency::seal"), "vram-residency");
    }

    #[test]
    fn test_extract_service_name_fallback() {
        assert_eq!(extract_service_name("unknown_module"), "unknown_module");
        assert_eq!(extract_service_name("llama_orch::unknown"), "unknown");
    }
}
