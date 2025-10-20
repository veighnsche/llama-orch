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
//!
//! TEAM-164: E2E tests MUST show live narration output.
//! Using .output() hides all narration until command completes.
//! Using .spawn() + .wait() shows narration in real-time.

use anyhow::Result;
use std::process::Command;

pub async fn test_queen_lifecycle() -> Result<()> {
    println!("🚀 E2E Test: Queen Lifecycle\n");

    // Step 1: rbee queen start
    println!("📝 Running: rbee queen start\n");

    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper").args(["queen", "start"]).spawn()?;

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee queen start failed with exit code: {:?}", status.code());
    }

    println!();

    // Step 2: rbee queen stop
    println!("📝 Running: rbee queen stop\n");

    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper").args(["queen", "stop"]).spawn()?;

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee queen stop failed with exit code: {:?}", status.code());
    }

    println!();

    println!("✅ E2E Test PASSED: Queen Lifecycle");
    Ok(())
}
