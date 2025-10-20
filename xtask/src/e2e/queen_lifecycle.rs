//! Queen lifecycle E2E test
//!
//! Created by: TEAM-160
//!
//! Tests:
//! $ rbee queen start
//! $ rbee queen stop

use anyhow::Result;
use std::process::Command;

use super::helpers;

pub async fn test_queen_lifecycle() -> Result<()> {
    println!("🚀 E2E Test: Queen Lifecycle\n");

    // Step 1: Build binaries
    helpers::build_binaries()?;
    println!();

    // Step 2: rbee queen start
    println!("📝 Running: rbee queen start");
    let output = Command::new("target/debug/rbee-keeper")
        .args(["queen", "start"])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("rbee queen start failed: {}", stderr);
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

    println!("✅ Queen is running and healthy");
    println!();

    // Step 4: rbee queen stop
    println!("📝 Running: rbee queen stop");
    let output = Command::new("target/debug/rbee-keeper")
        .args(["queen", "stop"])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("rbee queen stop failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    // Step 5: Verify queen is stopped
    println!("🔍 Verifying queen is stopped...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    let response = client.get("http://localhost:8500/health").send().await;
    if response.is_ok() {
        anyhow::bail!("Queen is still running after stop command");
    }

    println!("✅ Queen stopped successfully");
    println!();

    println!("✅ E2E Test PASSED: Queen Lifecycle");
    Ok(())
}
