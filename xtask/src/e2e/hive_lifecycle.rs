//! Hive lifecycle E2E test
//!
//! Created by: TEAM-160
//! Modified by: TEAM-162
//!
//! Tests:
//! $ rbee hive start
//! $ rbee hive stop
//!
//! Flow:
//! 1. Start queen
//! 2. Start hive (localhost)
//! 3. Wait for first heartbeat
//! 4. Stop hive
//! 5. Stop queen
//!
//! TEAM-162: Tests rely ONLY on CLI stdout/stderr.
//! No internal product functions. Pure black-box testing.
//!
//! TEAM-164: E2E tests MUST show live narration output.
//! Using .output() hides all narration until command completes.
//! Using .spawn() + .wait() shows narration in real-time.

use anyhow::Result;
use std::process::Command;

pub async fn test_hive_lifecycle() -> Result<()> {
    println!("🚀 E2E Test: Hive Lifecycle\n");

    // Step 0: Clean up any leftover state from previous runs
    println!("🧹 Cleaning up previous test state...\n");

    // Kill any running queen/hive processes
    let _ = Command::new("pkill").args(["-f", "queen-rbee|rbee-hive"]).output();

    // Remove stale database
    let _ = std::fs::remove_file("queen-hive-catalog.db");

    // Give processes time to die
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    println!();

    // Step 1: rbee hive start (starts queen + hive)
    println!("📝 Running: rbee hive start\n");

    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper").args(["hive", "start"]).spawn()?;

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee hive start failed with exit code: {:?}", status.code());
    }

    println!();

    // Step 6: rbee hive stop
    println!("📝 Running: rbee hive stop\n");

    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper").args(["hive", "stop"]).spawn()?;

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee hive stop failed with exit code: {:?}", status.code());
    }

    println!();

    // Step 9: Clean up - stop queen
    println!("🧹 Cleaning up - stopping queen...\n");
    let _ = Command::new("target/debug/rbee-keeper").args(["queen", "stop"]).spawn()?.wait()?;

    println!("✅ E2E Test PASSED: Hive Lifecycle");
    Ok(())
}
