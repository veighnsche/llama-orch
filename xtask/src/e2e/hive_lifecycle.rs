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

use anyhow::Result;
use std::process::Command;

pub async fn test_hive_lifecycle() -> Result<()> {
    println!("🚀 E2E Test: Hive Lifecycle\n");

    // Step 1: rbee hive start (starts queen + hive)
    println!("📝 Running: rbee hive start");
    let output = Command::new("target/debug/rbee-keeper")
        .args(["hive", "start"])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("rbee hive start failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);
    
    // Verify actual product output:
    // "✅ Queen is running"
    // "✅ Hive started on localhost:8600"
    if !stdout.contains("Queen is running") {
        anyhow::bail!("Expected 'Queen is running' in output, got: {}", stdout);
    }
    if !stdout.contains("Hive started on") {
        anyhow::bail!("Expected 'Hive started on' in output, got: {}", stdout);
    }
    println!();

    // Step 6: rbee hive stop
    println!("📝 Running: rbee hive stop");
    let output = Command::new("target/debug/rbee-keeper")
        .args(["hive", "stop"])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("rbee hive stop failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);
    
    // Verify actual product output: "✅ Hive stopped"
    if !stdout.contains("Hive stopped") {
        anyhow::bail!("Expected 'Hive stopped' in output, got: {}", stdout);
    }
    println!();

    // Step 9: Clean up - stop queen
    println!("🧹 Cleaning up - stopping queen...");
    let _ = Command::new("target/debug/rbee-keeper")
        .args(["queen", "stop"])
        .output()?;

    println!("✅ E2E Test PASSED: Hive Lifecycle");
    Ok(())
}
