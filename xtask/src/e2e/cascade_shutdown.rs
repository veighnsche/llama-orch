//! Cascade shutdown E2E test
//!
//! Created by: TEAM-160
//!
//! Tests cascading shutdown:
//! 1. Start queen
//! 2. Start hive
//! 3. Wait for first heartbeat
//! 4. Stop queen (should cascade to hive)
//! 5. Verify both queen and hive are stopped

use anyhow::Result;
use std::process::Command;

use super::helpers;

pub async fn test_cascade_shutdown() -> Result<()> {
    println!("🚀 E2E Test: Cascade Shutdown\n");

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

    // Step 3: Verify both are running
    println!("🔍 Verifying queen and hive are running...");
    let client = reqwest::Client::new();
    
    let response = client.get("http://localhost:8500/health").send().await?;
    if !response.status().is_success() {
        anyhow::bail!("Queen health check failed");
    }
    println!("✅ Queen is running");

    let response = client.get("http://localhost:8600/health").send().await?;
    if !response.status().is_success() {
        anyhow::bail!("Hive health check failed");
    }
    println!("✅ Hive is running");

    // Step 4: Wait for first heartbeat
    helpers::wait_for_first_heartbeat(8500, "localhost").await?;
    println!();

    // Step 5: Stop queen (should cascade to hive)
    println!("📝 Running: rbee queen stop (should cascade to hive)");
    let output = Command::new("target/debug/rbee-keeper")
        .args(["queen", "stop"])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("rbee queen stop failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    // Step 6: Verify queen is stopped
    println!("🔍 Verifying queen is stopped...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    let response = client.get("http://localhost:8500/health").send().await;
    if response.is_ok() {
        anyhow::bail!("Queen is still running after stop command");
    }

    println!("✅ Queen stopped");

    // Step 7: Verify hive is also stopped (cascade)
    println!("🔍 Verifying hive is also stopped (cascade)...");
    let response = client.get("http://localhost:8600/health").send().await;
    if response.is_ok() {
        anyhow::bail!("Hive is still running - cascade shutdown failed!");
    }

    println!("✅ Hive stopped (cascade worked!)");
    println!();

    println!("✅ E2E Test PASSED: Cascade Shutdown");
    println!("   Queen stopped → Hive stopped automatically");
    Ok(())
}
