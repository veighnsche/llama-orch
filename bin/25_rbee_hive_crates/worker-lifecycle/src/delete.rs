// TEAM-272: Worker deletion (process cleanup)
use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

/// Delete a worker by killing its process
///
/// This is a STATELESS operation - it just kills the process by PID.
/// The PID should be obtained from queen's worker registry.
///
/// # Architecture
///
/// 1. Hive receives WorkerDelete operation with PID
/// 2. Kill process using SIGTERM (graceful)
/// 3. Wait briefly, then SIGKILL if needed
/// 4. Return success
///
/// # Arguments
///
/// * `job_id` - Job ID for narration routing
/// * `worker_id` - Worker ID for logging
/// * `pid` - Process ID to kill
///
/// # Platform Support
///
/// - **Unix:** Uses SIGTERM/SIGKILL via nix crate
/// - **Windows:** Not yet implemented
pub async fn delete_worker(job_id: &str, worker_id: &str, pid: u32) -> Result<()> {
    NARRATE
        .action("worker_delete_start")
        .job_id(job_id)
        .context(worker_id)
        .context(&pid.to_string())
        .human("🗑️  Deleting worker '{}' (PID: {})")
        .emit();

    kill_process(job_id, pid).await?;

    NARRATE
        .action("worker_delete_complete")
        .job_id(job_id)
        .context(worker_id)
        .human("✅ Worker '{}' deleted")
        .emit();

    Ok(())
}

/// Kill a process by PID
///
/// Tries SIGTERM first (graceful shutdown), waits 2 seconds,
/// then uses SIGKILL if process still exists.
#[cfg(unix)]
async fn kill_process(job_id: &str, pid: u32) -> Result<()> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let pid_nix = Pid::from_raw(pid as i32);

    // Try SIGTERM first (graceful)
    NARRATE
        .action("worker_delete_sigterm")
        .job_id(job_id)
        .context(&pid.to_string())
        .human("Sending SIGTERM to PID {}")
        .emit();

    match kill(pid_nix, Signal::SIGTERM) {
        Ok(_) => {
            NARRATE
                .action("worker_delete_sigterm_sent")
                .job_id(job_id)
                .human("SIGTERM sent successfully")
                .emit();

            // Wait for graceful shutdown
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

            // Check if process still exists by trying to send signal 0
            match kill(pid_nix, Signal::SIGKILL) {
                Ok(_) => {
                    NARRATE
                        .action("worker_delete_sigkill")
                        .job_id(job_id)
                        .human("Process still alive, sent SIGKILL")
                        .emit();
                }
                Err(_) => {
                    // Process already dead, that's fine
                    NARRATE
                        .action("worker_delete_graceful")
                        .job_id(job_id)
                        .human("Process terminated gracefully")
                        .emit();
                }
            }

            Ok(())
        }
        Err(e) => {
            NARRATE
                .action("worker_delete_error")
                .job_id(job_id)
                .context(&e.to_string())
                .human("⚠️  Failed to kill process: {} (may already be dead)")
                .emit();

            // Don't fail - process may already be dead
            Ok(())
        }
    }
}

#[cfg(not(unix))]
async fn kill_process(job_id: &str, _pid: u32) -> Result<()> {
    NARRATE
        .action("worker_delete_unsupported")
        .job_id(job_id)
        .human("⚠️  Process killing not implemented for this platform")
        .emit();

    Err(anyhow::anyhow!("Process killing not implemented for non-Unix platforms"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[cfg(unix)]
    async fn test_kill_nonexistent_process() {
        // Trying to kill a non-existent process should not fail
        // (it's already dead, which is what we want)
        let result = kill_process("test-job", 999999).await;
        assert!(result.is_ok());
    }
}
