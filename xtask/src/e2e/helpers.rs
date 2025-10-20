//! Helper functions for E2E tests
//!
//! Created by: TEAM-160

use anyhow::{Context, Result};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;

/// Build all required binaries
pub fn build_binaries() -> Result<()> {
    println!("🔨 Building binaries...");

    // Build queen-rbee
    let status = Command::new("cargo")
        .args(["build", "--bin", "queen-rbee"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .context("Failed to build queen-rbee")?;

    if !status.success() {
        anyhow::bail!("queen-rbee build failed");
    }

    // Build rbee-keeper
    let status = Command::new("cargo")
        .args(["build", "--bin", "rbee-keeper"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .context("Failed to build rbee-keeper")?;

    if !status.success() {
        anyhow::bail!("rbee-keeper build failed");
    }

    // Build rbee-hive
    let status = Command::new("cargo")
        .args(["build", "--bin", "rbee-hive"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .context("Failed to build rbee-hive")?;

    if !status.success() {
        anyhow::bail!("rbee-hive build failed");
    }

    println!("✅ All binaries built");
    Ok(())
}

/// Start queen-rbee daemon
pub fn start_queen(port: u16) -> Result<Child> {
    println!("👑 Starting queen-rbee on port {}...", port);

    let child = Command::new("target/debug/queen-rbee")
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn queen-rbee")?;

    Ok(child)
}

/// Wait for queen to be healthy
pub async fn wait_for_queen(port: u16) -> Result<()> {
    println!("⏳ Waiting for queen to be ready...");

    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);

    for attempt in 1..=20 {
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                println!("✅ Queen ready after {} attempts", attempt);
                return Ok(());
            }
            _ => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    }

    anyhow::bail!("Queen failed to start after 10s")
}

/// Start rbee-hive daemon
pub fn start_hive(port: u16, queen_url: &str) -> Result<Child> {
    println!("🐝 Starting rbee-hive on port {}...", port);

    let child = Command::new("target/debug/rbee-hive")
        .arg("--port")
        .arg(port.to_string())
        .arg("--queen-url")
        .arg(queen_url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn rbee-hive")?;

    Ok(child)
}

/// Wait for hive to be healthy
pub async fn wait_for_hive(port: u16) -> Result<()> {
    println!("⏳ Waiting for hive to be ready...");

    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);

    for attempt in 1..=20 {
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                println!("✅ Hive ready after {} attempts", attempt);
                return Ok(());
            }
            _ => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    }

    anyhow::bail!("Hive failed to start after 10s")
}

/// Wait for first heartbeat from hive
pub async fn wait_for_first_heartbeat(queen_port: u16, hive_id: &str) -> Result<()> {
    println!("⏳ Waiting for first heartbeat from hive {}...", hive_id);

    let client = reqwest::Client::new();
    let catalog_url = format!("http://localhost:{}/hives", queen_port);

    for attempt in 1..=40 {
        match client.get(&catalog_url).send().await {
            Ok(response) if response.status().is_success() => {
                if let Ok(hives) = response.json::<serde_json::Value>().await {
                    if let Some(hive) = hives.as_array().and_then(|arr| {
                        arr.iter().find(|h| h["id"].as_str() == Some(hive_id))
                    }) {
                        if hive["last_heartbeat_ms"].is_number() {
                            println!("✅ First heartbeat received after {} attempts", attempt);
                            return Ok(());
                        }
                    }
                }
            }
            _ => {}
        }
        sleep(Duration::from_millis(500)).await;
    }

    anyhow::bail!("No heartbeat received after 20s")
}

/// Kill a process
pub fn kill_process(child: &mut Child) -> Result<()> {
    child.kill().context("Failed to kill process")?;
    child.wait().context("Failed to wait for process")?;
    Ok(())
}
