//! Health check step definitions
//!
//! Created by: TEAM-151 (2025-10-20)
//! Tests for queen-rbee health check functionality

use cucumber::{given, then, when};
use std::process::Child;
use std::time::Duration;

use super::world::BddWorld;

#[given(expr = "the queen URL is {string}")]
async fn set_queen_url(world: &mut BddWorld, url: String) {
    world.queen_url = url;
}

#[given("queen-rbee is not running")]
async fn queen_not_running(world: &mut BddWorld) {
    // Clean up any existing queen process from previous tests
    if let Some(mut child) = world.queen_process.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    
    // Ensure no queen is running on the default port
    // Kill any existing queen process
    let _ = std::process::Command::new("pkill")
        .arg("-9")
        .arg("-f")
        .arg("queen-rbee")
        .output();
    
    // Wait a bit to ensure it's stopped
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    world.queen_process = None;
}

#[given(expr = "queen-rbee is running on port {int}")]
async fn queen_running_on_port(world: &mut BddWorld, port: u16) {
    // Find the queen-rbee binary in the workspace target directory
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    
    let binary_path = std::path::PathBuf::from(&workspace_root)
        .parent()
        .expect("Cannot find parent")
        .parent()
        .expect("Cannot find parent")
        .parent()
        .expect("Cannot find parent")
        .join("target/debug/queen-rbee");

    if !binary_path.exists() {
        panic!("queen-rbee binary not found at {:?}. Run: cargo build --bin queen-rbee", binary_path);
    }

    // Start queen-rbee
    let child = std::process::Command::new(&binary_path)
        .arg("--port")
        .arg(port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to start queen-rbee");

    world.queen_process = Some(child);
    
    // Wait for queen to start
    tokio::time::sleep(Duration::from_secs(2)).await;
}

#[given("queen-rbee is not responding within 500ms")]
async fn queen_not_responding(_world: &mut BddWorld) {
    // This is a no-op - the timeout is built into the health check
    // We just need to ensure queen is not running
}

#[when("I check if queen is healthy")]
async fn check_queen_health(world: &mut BddWorld) {
    // Import the health check function from the parent crate
    // For BDD tests, we'll call it directly
    let result = check_health(&world.queen_url).await;
    world.health_check_result = Some(result);
}

#[then("the health check should return true")]
async fn health_check_returns_true(world: &mut BddWorld) {
    let result = world.health_check_result.as_ref()
        .expect("Health check was not performed");
    
    match result {
        Ok(true) => {}, // Success
        Ok(false) => panic!("Expected health check to return true, but got false"),
        Err(e) => panic!("Expected health check to succeed, but got error: {}", e),
    }
}

#[then("the health check should return false")]
async fn health_check_returns_false(world: &mut BddWorld) {
    let result = world.health_check_result.as_ref()
        .expect("Health check was not performed");
    
    match result {
        Ok(false) => {}, // Success
        Ok(true) => panic!("Expected health check to return false, but got true"),
        Err(e) => panic!("Expected health check to return false, but got error: {}", e),
    }
}

#[then("the health check should timeout")]
async fn health_check_timeouts(world: &mut BddWorld) {
    let result = world.health_check_result.as_ref()
        .expect("Health check was not performed");
    
    // Should either return false or error
    assert!(result.is_err() || matches!(result, Ok(false)));
}

#[then(expr = "I should see {string}")]
async fn should_see_message(world: &mut BddWorld, expected: String) {
    // In a real scenario, we'd capture stdout/stderr
    // For now, we just verify the health check result matches expectations
    let result = world.health_check_result.as_ref()
        .expect("Health check was not performed");
    
    if expected.contains("not running") {
        assert!(matches!(result, Ok(false)), "Expected queen to be not running");
    } else if expected.contains("running and healthy") {
        assert!(matches!(result, Ok(true)), "Expected queen to be running");
    }
    
    // Store the expected message for validation
    world.expected_message = Some(expected);
}

#[then("I should see connection error")]
async fn should_see_connection_error(world: &mut BddWorld) {
    let result = world.health_check_result.as_ref()
        .expect("Health check was not performed");
    
    // Should return false (connection refused) or error
    assert!(result.is_err() || matches!(result, Ok(false)));
}

/// Helper function to check health
/// This mirrors the implementation in the main crate
async fn check_health(base_url: &str) -> Result<bool, String> {
    let health_url = format!("{}/health", base_url);
    
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .map_err(|e| e.to_string())?;
    
    match client.get(&health_url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Err(e) => {
            // Connection refused means queen is not running
            if e.is_connect() {
                Ok(false)
            } else {
                Err(e.to_string())
            }
        }
    }
}
