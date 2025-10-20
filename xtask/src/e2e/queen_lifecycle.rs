//! Queen lifecycle E2E test
//!
//! Created by: TEAM-160
//! Modified by: TEAM-162
//!
//! Tests:
//! $ rbee queen start
//! $ rbee queen stop
//!
//! TEAM-162: Tests rely ONLY on CLI stdout/stderr.
//! No internal product functions. Pure black-box testing.

use anyhow::Result;
use std::process::Command;

pub async fn test_queen_lifecycle() -> Result<()> {
    println!("🚀 E2E Test: Queen Lifecycle\n");

    // Step 1: rbee queen start
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
    
    // Verify actual product output: "✅ Queen started on http://localhost:8500"
    if !stdout.contains("Queen started on") {
        anyhow::bail!("Expected 'Queen started on' in output, got: {}", stdout);
    }

    // Step 2: rbee queen stop
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
    
    // Verify actual product output: "✅ Queen stopped"
    if !stdout.contains("Queen stopped") {
        anyhow::bail!("Expected 'Queen stopped' in output, got: {}", stdout);
    }

    println!("✅ E2E Test PASSED: Queen Lifecycle");
    Ok(())
}
