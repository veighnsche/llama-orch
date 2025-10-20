//! Hive lifecycle E2E test
//!
//! Created by: TEAM-160
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

use anyhow::Result;
use std::process::Command;

use super::helpers;

pub async fn test_hive_lifecycle() -> Result<()> {
    println!("🚀 E2E Test: Hive Lifecycle\n");

    // Step 1: Build binaries
    helpers::build_binaries()?;
    println!();

    // Step 2: rbee hive start (starts queen + hive)
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

    // Step 3: Verify queen is running
    println!("🔍 Verifying queen is running...");
    let client = reqwest::Client::new();
    let response = client.get("http://localhost:8500/health").send().await?;

    if !response.status().is_success() {
        anyhow::bail!("Queen health check failed");
    }

    println!("✅ Queen is running");

    // Step 4: Verify hive is running
    println!("🔍 Verifying hive is running...");
    let response = client.get("http://localhost:8600/health").send().await?;

    if !response.status().is_success() {
        anyhow::bail!("Hive health check failed");
    }

    println!("✅ Hive is running");

    // Step 5: Wait for first heartbeat
    helpers::wait_for_first_heartbeat(8500, "localhost").await?;
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

    // Step 7: Verify hive is stopped
    println!("🔍 Verifying hive is stopped...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    let response = client.get("http://localhost:8600/health").send().await;
    if response.is_ok() {
        anyhow::bail!("Hive is still running after stop command");
    }

    println!("✅ Hive stopped successfully");

    // Step 8: Verify queen is still running
    println!("🔍 Verifying queen is still running...");
    let response = client.get("http://localhost:8500/health").send().await?;

    if !response.status().is_success() {
        anyhow::bail!("Queen stopped unexpectedly");
    }

    println!("✅ Queen still running (as expected)");
    println!();

    // Step 9: Clean up - stop queen
    println!("🧹 Cleaning up - stopping queen...");
    let _ = Command::new("target/debug/rbee-keeper")
        .args(["queen", "stop"])
        .output()?;

    println!("✅ E2E Test PASSED: Hive Lifecycle");
    Ok(())
}
