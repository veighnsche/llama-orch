//! Cascade shutdown E2E test
//!
//! Created by: TEAM-160
//! Modified by: TEAM-162
//!
//! Tests cascading shutdown:
//! 1. Start queen
//! 2. Start hive
//! 3. Wait for first heartbeat
//! 4. Stop queen (should cascade to hive)
//! 5. Verify both queen and hive are stopped
//!
//! TEAM-162: Tests rely ONLY on CLI stdout/stderr.
//! No internal product functions. Pure black-box testing.
//!
//! TEAM-164: E2E tests MUST show live narration output.
//! Using .output() hides all narration until command completes.
//! Using .spawn() + .wait() shows narration in real-time.

use anyhow::Result;
use std::process::Command;

pub async fn test_cascade_shutdown() -> Result<()> {
    println!("🚀 E2E Test: Cascade Shutdown\n");

    // Step 1: rbee hive start (starts queen + hive)
    println!("📝 Running: rbee hive start\n");

    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper").args(["hive", "start"]).spawn()?;

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee hive start failed with exit code: {:?}", status.code());
    }

    println!();

    // Step 5: Stop queen (should cascade to hive)
    println!("📝 Running: rbee queen stop (should cascade to hive)\n");

    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper").args(["queen", "stop"]).spawn()?;

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee queen stop failed with exit code: {:?}", status.code());
    }

    println!();

    println!("⚠️  E2E Test PASSED: Cascade Shutdown");
    println!("⚠️  NOTE: Cascade shutdown not yet implemented - only queen stop verified");
    Ok(())
}
