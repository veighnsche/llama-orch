// TEAM-274: Get worker process details by PID
// TEAM-276: Renamed from process_get.rs to get.rs for consistency
use anyhow::Result;
use observability_narration_core::NarrationFactory;

use crate::types::WorkerInfo;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

/// Get details of a specific worker process by PID
///
/// TEAM-276: Renamed from get_worker_process to get_worker for consistency
///
/// This uses local `ps` command to get process details.
///
/// # Arguments
///
/// * `job_id` - Job ID for narration routing
/// * `pid` - Process ID to query
///
/// # Returns
///
/// WorkerInfo for the specified PID, or error if not found
#[cfg(unix)]
pub async fn get_worker(job_id: &str, pid: u32) -> Result<WorkerInfo> {
    use tokio::process::Command;

    NARRATE
        .action("proc_get_start")
        .job_id(job_id)
        .context(&pid.to_string())
        .human("🔍 Getting process details for PID {}")
        .emit();

    // Run ps command for specific PID
    // ps -p PID -o pid,%cpu,%mem,rss,etime,command
    let output = Command::new("ps")
        .args(&["-p", &pid.to_string(), "-o", "pid,%cpu,%mem,rss,etime,command"])
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to run ps command: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        NARRATE
            .action("proc_get_not_found")
            .job_id(job_id)
            .context(&pid.to_string())
            .human("❌ Process {} not found")
            .emit();
        return Err(anyhow::anyhow!("Process {} not found: {}", pid, stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    if lines.len() < 2 {
        return Err(anyhow::anyhow!("Process {} not found", pid));
    }

    // Parse ps output (skip header)
    let line = lines[1];
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 6 {
        return Err(anyhow::anyhow!("Failed to parse ps output for PID {}", pid));
    }

    let cpu_percent = parts[1].parse::<f32>().unwrap_or(0.0);
    let memory_kb = parts[3].parse::<u64>().unwrap_or(0);
    let elapsed = parts[4].to_string();
    let command = parts[5..].join(" ");

    // TEAM-276: Simplified to WorkerInfo
    let info = WorkerInfo {
        pid,
        command: command.clone(),
        args: vec![], // TODO: Parse args from command if needed
    };

    NARRATE
        .action("proc_get_found")
        .job_id(job_id)
        .context(&pid.to_string())
        .context(&command)
        .human("✅ Process {}: {}")
        .emit();

    Ok(info)
}

#[cfg(not(unix))]
pub async fn get_worker_process(job_id: &str, pid: u32) -> Result<WorkerProcessInfo> {
    NARRATE
        .action("proc_get_unsupported")
        .job_id(job_id)
        .context(&pid.to_string())
        .human("⚠️  Process query not implemented for this platform")
        .emit();

    Err(anyhow::anyhow!("Process query not implemented for non-Unix platforms"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[cfg(unix)]
    async fn test_get_current_process() {
        // Test with current process PID (should always exist)
        let current_pid = std::process::id();
        let result = get_worker_process("test-job", current_pid).await;
        assert!(result.is_ok());

        let info = result.unwrap();
        assert_eq!(info.pid, current_pid);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_get_nonexistent_process() {
        // PID 999999 should not exist
        let result = get_worker_process("test-job", 999999).await;
        assert!(result.is_err());
    }
}
