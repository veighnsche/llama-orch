// TEAM-274: Worker process listing using local ps commands
// TEAM-276: Renamed from process_list.rs to list.rs for consistency
use anyhow::Result;
use observability_narration_core::NarrationFactory;

// TEAM-276: Import WorkerInfo from types
use crate::types::WorkerInfo;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

/// List all worker processes on this hive
///
/// TEAM-276: Renamed from list_worker_processes to list_workers for consistency
///
/// This uses local `ps` command to find processes matching worker patterns.
///
/// # Architecture
///
/// Hive is STATELESS - this operation scans local processes, not a registry.
/// It's useful for debugging but ActiveWorkerList (in queen) is the source of truth.
///
/// # Arguments
///
/// * `job_id` - Job ID for narration routing
///
/// # Returns
///
/// Vector of WorkerInfo for all found worker processes
#[cfg(unix)]
pub async fn list_workers(job_id: &str) -> Result<Vec<WorkerInfo>> {
    use tokio::process::Command;

    NARRATE
        .action("proc_list_start")
        .job_id(job_id)
        .human("🔍 Scanning for worker processes (local ps)")
        .emit();

    // Run ps command to find worker processes
    // ps aux format: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
    let output = Command::new("ps")
        .args(&["aux"])
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to run ps command: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("ps command failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut processes = Vec::new();

    // Parse ps output
    for line in stdout.lines().skip(1) {
        // Skip header
        // Look for worker processes (heuristic: contains "worker" or "llm")
        if !line.contains("worker") && !line.contains("llm") {
            continue;
        }

        // Skip the ps command itself and grep
        if line.contains(" ps aux") || line.contains(" grep ") {
            continue;
        }

        // Parse ps output: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 11 {
            continue;
        }

        let pid = parts[1].parse::<u32>().unwrap_or(0);
        if pid == 0 {
            continue;
        }

        let cpu_percent = parts[2].parse::<f32>().unwrap_or(0.0);
        let memory_kb = parts[5].parse::<u64>().unwrap_or(0);
        let elapsed = parts[9].to_string();
        let command = parts[10..].join(" ");

        // TEAM-276: Simplified to WorkerInfo (removed memory/cpu/elapsed - can add back if needed)
        processes.push(WorkerInfo {
            pid,
            command: command.clone(),
            args: vec![], // TODO: Parse args from command if needed
        });
    }

    NARRATE
        .action("proc_list_found")
        .job_id(job_id)
        .context(&processes.len().to_string())
        .human("Found {} worker process(es)")
        .emit();

    Ok(processes)
}

#[cfg(not(unix))]
pub async fn list_worker_processes(job_id: &str) -> Result<Vec<WorkerProcessInfo>> {
    NARRATE
        .action("proc_list_unsupported")
        .job_id(job_id)
        .human("⚠️  Process listing not implemented for this platform")
        .emit();

    Err(anyhow::anyhow!("Process listing not implemented for non-Unix platforms"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[cfg(unix)]
    async fn test_list_worker_processes() {
        // This test just ensures the function doesn't panic
        // It may return empty list if no workers are running
        let result = list_worker_processes("test-job").await;
        assert!(result.is_ok());
    }
}
