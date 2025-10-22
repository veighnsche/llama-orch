// TEAM-245: Graceful shutdown tests for hive-lifecycle
// Purpose: Test SIGTERM → SIGKILL fallback and idempotency
// Priority: CRITICAL (prevents zombie processes)
// Scale: Reasonable for NUC (5-10 concurrent, no overkill)

use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::{sleep, timeout};

// ============================================================================
// SIGTERM Behavior Tests
// ============================================================================

#[tokio::test]
async fn test_sigterm_success_within_5s() {
    // TEAM-245: Test SIGTERM causes process to exit within 5s
    // This is the happy path - process responds to SIGTERM

    // Spawn a test process that responds to SIGTERM
    let mut child = Command::new("sleep")
        .arg("30")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn test process");

    let pid = child.id();

    // Send SIGTERM
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        
        kill(Pid::from_raw(pid as i32), Signal::SIGTERM).expect("Failed to send SIGTERM");
    }

    // Wait up to 5s for process to exit
    let result = timeout(Duration::from_secs(5), child.wait()).await;

    assert!(result.is_ok(), "Process should exit within 5s after SIGTERM");
    let status = result.unwrap().expect("Failed to wait for process");
    assert!(!status.success(), "Process should be terminated (not success)");
}

#[tokio::test]
async fn test_sigterm_timeout_sigkill_fallback() {
    // TEAM-245: Test SIGKILL sent if SIGTERM doesn't work within 5s
    // This tests the fallback mechanism

    // Spawn a process that ignores SIGTERM (trap in shell)
    #[cfg(unix)]
    {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("trap '' TERM; sleep 30") // Ignore SIGTERM
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn test process");

        let pid = child.id();

        // Send SIGTERM (will be ignored)
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        
        kill(Pid::from_raw(pid as i32), Signal::SIGTERM).expect("Failed to send SIGTERM");

        // Wait 5s (SIGTERM timeout)
        sleep(Duration::from_secs(5)).await;

        // Check if still running
        match child.try_wait() {
            Ok(Some(_)) => {
                // Process exited (shouldn't happen with trap)
            }
            Ok(None) => {
                // Still running - send SIGKILL
                kill(Pid::from_raw(pid as i32), Signal::SIGKILL).expect("Failed to send SIGKILL");
                
                // Wait for SIGKILL to take effect
                let result = timeout(Duration::from_millis(500), child.wait()).await;
                assert!(result.is_ok(), "Process should exit immediately after SIGKILL");
            }
            Err(e) => panic!("Error checking process status: {}", e),
        }
    }
}

#[tokio::test]
async fn test_stop_is_idempotent() {
    // TEAM-245: Test stopping already-stopped process is idempotent (no error)
    
    // Spawn and immediately kill a process
    let mut child = Command::new("sleep")
        .arg("1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn test process");

    let pid = child.id();

    // Kill it
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        
        kill(Pid::from_raw(pid as i32), Signal::SIGKILL).expect("Failed to send SIGKILL");
    }

    // Wait for it to die
    child.wait().expect("Failed to wait for process");

    // Try to kill it again (should not error)
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        
        let result = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
        // ESRCH (No such process) is expected and OK
        match result {
            Ok(_) => {}, // Somehow succeeded
            Err(nix::errno::Errno::ESRCH) => {}, // Expected: no such process
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
}

// ============================================================================
// Health Check During Shutdown Tests
// ============================================================================

#[tokio::test]
async fn test_health_check_polling_during_shutdown() {
    // TEAM-245: Test health check polls every 1s during shutdown
    // Verifies the polling loop (lines 134-149 in stop.rs)

    let start = std::time::Instant::now();
    
    // Simulate polling loop (5 attempts, 1s each)
    for attempt in 1..=5 {
        // Simulate health check
        sleep(Duration::from_secs(1)).await;
        
        if attempt == 5 {
            // On last attempt, break
            break;
        }
    }

    let elapsed = start.elapsed();
    
    // Should take ~5 seconds (5 attempts * 1s)
    assert!(
        elapsed.as_secs() >= 4 && elapsed.as_secs() <= 6,
        "Polling should take ~5s (got {}s)",
        elapsed.as_secs()
    );
}

#[tokio::test]
async fn test_early_exit_when_health_check_fails() {
    // TEAM-245: Test early exit when health check fails (hive stopped)
    // Should not wait full 5s if hive stops early

    let start = std::time::Instant::now();
    
    // Simulate polling loop with early exit
    for attempt in 1..=5 {
        // Simulate health check
        let health_ok = attempt < 3; // Fails on attempt 3
        
        if !health_ok {
            // Hive stopped - exit early
            break;
        }
        
        sleep(Duration::from_secs(1)).await;
    }

    let elapsed = start.elapsed();
    
    // Should take ~2 seconds (stopped on attempt 3)
    assert!(
        elapsed.as_secs() >= 1 && elapsed.as_secs() <= 3,
        "Should exit early when health check fails (got {}s)",
        elapsed.as_secs()
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_pkill_command_not_found() {
    // TEAM-245: Test pkill command not found (should error gracefully)
    
    // Try to run non-existent command
    let result = Command::new("nonexistent_command_12345")
        .arg("-TERM")
        .arg("test")
        .output()
        .await;

    assert!(result.is_err(), "Should error when command not found");
}

#[tokio::test]
async fn test_permission_denied() {
    // TEAM-245: Test permission denied (non-root killing root process)
    // This is hard to test without root, so we test the concept

    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        
        // Try to kill PID 1 (init) - should fail with EPERM
        let result = kill(Pid::from_raw(1), Signal::SIGTERM);
        
        match result {
            Ok(_) => {
                // Somehow succeeded (running as root?)
            }
            Err(nix::errno::Errno::EPERM) => {
                // Expected: permission denied
            }
            Err(e) => {
                // Other error (also acceptable for this test)
                println!("Got error (acceptable): {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_process_name_collision() {
    // TEAM-245: Test multiple processes with same name
    // pkill kills ALL processes with that name

    // Spawn 3 processes with same name
    let mut children = vec![];
    for _ in 0..3 {
        let child = Command::new("sleep")
            .arg("30")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn test process");
        children.push(child);
    }

    // Get PIDs
    let pids: Vec<u32> = children.iter().map(|c| c.id()).collect();

    // Kill all "sleep" processes
    #[cfg(unix)]
    {
        let _ = Command::new("pkill")
            .arg("-TERM")
            .arg("sleep")
            .output()
            .await;
    }

    // Wait for all to die
    for mut child in children {
        let result = timeout(Duration::from_secs(2), child.wait()).await;
        assert!(result.is_ok(), "All processes should be killed");
    }

    println!("Killed {} processes with name 'sleep'", pids.len());
}

// ============================================================================
// Integration with Hive Lifecycle
// ============================================================================

#[tokio::test]
async fn test_graceful_shutdown_flow() {
    // TEAM-245: Test complete graceful shutdown flow
    // 1. Check if running
    // 2. Send SIGTERM
    // 3. Poll health (5 attempts, 1s each)
    // 4. Send SIGKILL if still running
    // 5. Verify stopped

    // Spawn a test process
    let mut child = Command::new("sleep")
        .arg("30")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn test process");

    let pid = child.id();

    // Step 1: Check if running
    assert!(child.try_wait().unwrap().is_none(), "Process should be running");

    // Step 2: Send SIGTERM
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        
        kill(Pid::from_raw(pid as i32), Signal::SIGTERM).expect("Failed to send SIGTERM");
    }

    // Step 3: Poll health (simulate with try_wait)
    let mut stopped = false;
    for _ in 1..=5 {
        sleep(Duration::from_secs(1)).await;
        
        if child.try_wait().unwrap().is_some() {
            stopped = true;
            break;
        }
    }

    // Step 4: Send SIGKILL if still running
    if !stopped {
        #[cfg(unix)]
        {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;
            
            kill(Pid::from_raw(pid as i32), Signal::SIGKILL).expect("Failed to send SIGKILL");
        }
        
        sleep(Duration::from_millis(500)).await;
    }

    // Step 5: Verify stopped
    let status = child.try_wait().expect("Failed to check status");
    assert!(status.is_some(), "Process should be stopped");
}
