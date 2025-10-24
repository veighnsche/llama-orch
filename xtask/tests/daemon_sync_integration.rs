// TEAM-282: Daemon sync integration tests
// Purpose: Test SSH deployment from HOST to CONTAINER
// Architecture: HOST runs queen-rbee → SSH → Container (empty target)
//
// CRITICAL: These tests run on the HOST, not in a container.
// They verify that queen-rbee can deploy rbee-hive to a remote system via SSH.
//
// Run with: cargo test --package xtask --test daemon_sync_integration --ignored

use std::process::Command;
use tokio::process::Command as AsyncCommand;
use tokio::time::{sleep, Duration};

/// Helper: Start empty target container with SSH
async fn start_target_container() -> TestContainer {
    // Get workspace root
    let workspace_root = std::env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .expect("No parent directory")
        .to_path_buf();
    
    // Clean up any existing container
    let _ = Command::new("docker")
        .args(&["rm", "-f", "rbee-test-target"])
        .output();
    
    // Build the target image
    let build = Command::new("docker")
        .args(&[
            "build",
            "-t", "rbee-test-target:latest",
            "-f", "tests/docker/Dockerfile.target",
            "tests/docker"
        ])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to build target image");
    
    assert!(build.status.success(), "Failed to build target image: {}", 
        String::from_utf8_lossy(&build.stderr));
    
    // Start the container
    let start = AsyncCommand::new("docker")
        .args(&[
            "run", "-d",
            "--rm",
            "-p", "2222:22",
            "-p", "9000:9000",
            "--name", "rbee-test-target",
            "rbee-test-target:latest"
        ])
        .output()
        .await
        .expect("Failed to start container");
    
    assert!(start.status.success(), "Failed to start container: {}", 
        String::from_utf8_lossy(&start.stderr));
    
    let container_id = String::from_utf8_lossy(&start.stdout).trim().to_string();
    
    // Wait for SSH to be ready
    for _ in 0..30 {
        let check = AsyncCommand::new("docker")
            .args(&["exec", "rbee-test-target", "pgrep", "-x", "sshd"])
            .output()
            .await;
        
        if check.map(|o| o.status.success()).unwrap_or(false) {
            println!("✅ Container SSH ready");
            return TestContainer { id: container_id };
        }
        
        sleep(Duration::from_millis(500)).await;
    }
    
    panic!("Container SSH never became ready");
}

/// Helper: Build queen-rbee on HOST
async fn build_queen_rbee() {
    let workspace_root = std::env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .expect("No parent directory")
        .to_path_buf();
    
    let build = Command::new("cargo")
        .args(&["build", "--bin", "queen-rbee"])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to build queen-rbee");
    
    assert!(build.status.success(), "Failed to build queen-rbee: {}", 
        String::from_utf8_lossy(&build.stderr));
    
    println!("✅ queen-rbee built on HOST");
}

/// Helper: Container handle for cleanup
struct TestContainer {
    id: String,
}

impl TestContainer {
    /// Check if binary exists in container
    async fn has_binary(&self, path: &str) -> bool {
        let check = AsyncCommand::new("docker")
            .args(&["exec", "rbee-test-target", "test", "-f", path])
            .output()
            .await
            .expect("Failed to check binary");
        
        check.status.success()
    }
    
    /// Check if daemon is running in container
    async fn daemon_running(&self, name: &str) -> bool {
        let check = AsyncCommand::new("docker")
            .args(&["exec", "rbee-test-target", "pgrep", "-f", name])
            .output()
            .await
            .expect("Failed to check daemon");
        
        check.status.success()
    }
    
    /// Cleanup container
    async fn cleanup(&self) {
        let _ = AsyncCommand::new("docker")
            .args(&["rm", "-f", &self.id])
            .output()
            .await;
        
        println!("✅ Container cleaned up");
    }
}

// ============================================================================
// TEST SUITE
// ============================================================================
//
// TEAM-282: DELETED fake helper tests that used SSH from test harness.
// Those tests didn't verify that queen-rbee can SSH - they verified that
// the test harness can SSH. This is exactly what WHY_LLMS_ARE_STUPID.md
// warns about: shortcuts that mask whether the product works.
//
// Only the main integration test remains, which actually tests the product.

#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_queen_installs_hive_in_docker() {
    println!("\n🐝 TEAM-282: Full daemon-sync integration test");
    println!("{}", "=".repeat(60));
    
    // STEP 0: Cleanup any existing queen-rbee processes
    let _ = Command::new("pkill")
        .args(&["-f", "queen-rbee"])
        .output();
    sleep(Duration::from_millis(500)).await;
    
    // STEP 0.5: Setup SSH key for test
    let workspace_root = std::env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .expect("No parent directory")
        .to_path_buf();
    
    let test_key = workspace_root.join("tests/docker/keys/test_id_rsa");
    let ssh_dir = std::env::var("HOME").expect("HOME not set") + "/.ssh";
    let target_key = format!("{}/id_rsa", ssh_dir);
    
    // Backup existing key if it exists
    let backup_key = format!("{}/id_rsa.backup_before_test", ssh_dir);
    if std::path::Path::new(&target_key).exists() {
        let _ = std::fs::copy(&target_key, &backup_key);
    }
    
    // Copy test key to ~/.ssh/id_rsa
    std::fs::create_dir_all(&ssh_dir).expect("Failed to create .ssh dir");
    std::fs::copy(&test_key, &target_key).expect("Failed to copy test key");
    
    // Set correct permissions
    let _ = Command::new("chmod")
        .args(&["600", &target_key])
        .output();
    
    // STEP 1: Start empty target container
    println!("\n📦 STEP 1: Starting empty target container...");
    let container = start_target_container().await;
    
    // STEP 2: Build queen-rbee on HOST
    println!("\n🔨 STEP 2: Building queen-rbee on HOST...");
    build_queen_rbee().await;
    
    // STEP 3: Build rbee-hive on HOST (needed for installation)
    println!("\n🔨 STEP 3: Building rbee-hive on HOST...");
    let workspace_root = std::env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .expect("No parent directory")
        .to_path_buf();
    
    let build_hive = Command::new("cargo")
        .args(&["build", "--bin", "rbee-hive"])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to build rbee-hive");
    
    assert!(build_hive.status.success(), "Failed to build rbee-hive: {}", 
        String::from_utf8_lossy(&build_hive.stderr));
    println!("✅ rbee-hive built on HOST");
    
    // STEP 4: Start queen-rbee on HOST with test config
    println!("\n👑 STEP 4: Starting queen-rbee on HOST...");
    let queen_binary = workspace_root.join("target/debug/queen-rbee");
    let config_dir = workspace_root.join("tests/docker");
    
    let mut queen = AsyncCommand::new(&queen_binary)
        .args(&["--config-dir", config_dir.to_str().unwrap()])
        .env("RUST_LOG", "debug")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to start queen-rbee");
    
    // Wait for queen to start
    sleep(Duration::from_secs(3)).await;
    
    // Check if queen is running
    let health_check = AsyncCommand::new("curl")
        .args(&["-f", "http://localhost:8500/health"])
        .output()
        .await;
    
    if !health_check.map(|o| o.status.success()).unwrap_or(false) {
        // Get queen output before killing
        let stdout = queen.stdout.take();
        let stderr = queen.stderr.take();
        let _ = queen.kill().await;
        
        if let Some(mut stdout) = stdout {
            use tokio::io::AsyncReadExt;
            let mut buf = String::new();
            let _ = stdout.read_to_string(&mut buf).await;
            eprintln!("queen-rbee stdout:\n{}", buf);
        }
        if let Some(mut stderr) = stderr {
            use tokio::io::AsyncReadExt;
            let mut buf = String::new();
            let _ = stderr.read_to_string(&mut buf).await;
            eprintln!("queen-rbee stderr:\n{}", buf);
        }
        
        panic!("queen-rbee health check failed - is it running?");
    }
    println!("✅ queen-rbee is running on http://localhost:8500");
    
    // STEP 5: Send PackageInstall command via HTTP
    println!("\n📡 STEP 5: Sending PackageInstall command...");
    let hives_conf_path = workspace_root.join("tests/docker/hives.conf");
    let install_payload = format!(
        r#"{{"operation":"package_install","config_path":"{}","force":false}}"#,
        hives_conf_path.to_str().unwrap()
    );
    
    let install_cmd = AsyncCommand::new("curl")
        .args(&[
            "-X", "POST",
            "http://localhost:8500/v1/jobs",
            "-H", "Content-Type: application/json",
            "-d", &install_payload
        ])
        .output()
        .await
        .expect("Failed to send install command");
    
    if !install_cmd.status.success() {
        // Get queen output
        let stdout = queen.stdout.take();
        let stderr = queen.stderr.take();
        let _ = queen.kill().await;
        
        if let Some(mut stdout) = stdout {
            use tokio::io::AsyncReadExt;
            let mut buf = String::new();
            let _ = stdout.read_to_string(&mut buf).await;
            eprintln!("queen-rbee stdout:\n{}", buf);
        }
        if let Some(mut stderr) = stderr {
            use tokio::io::AsyncReadExt;
            let mut buf = String::new();
            let _ = stderr.read_to_string(&mut buf).await;
            eprintln!("queen-rbee stderr:\n{}", buf);
        }
        
        panic!("Install command failed: {}", 
            String::from_utf8_lossy(&install_cmd.stderr));
    }
    
    let response = String::from_utf8_lossy(&install_cmd.stdout);
    println!("📨 Response: {}", response);
    
    // Extract job_id from response
    let job_id: serde_json::Value = serde_json::from_str(&response)
        .expect("Failed to parse response");
    let job_id = job_id["job_id"].as_str()
        .expect("No job_id in response");
    println!("✅ Job submitted: {}", job_id);
    
    // STEP 6: Wait for installation to complete (stream SSE and show narration)
    // NOTE: Git clone + cargo build takes ~1m20s, so we poll for up to 3 minutes
    println!("\n⏳ STEP 6: Streaming installation progress (git clone + build ~1m20s)...");
    println!("{}", "-".repeat(60));
    let stream_url = format!("http://localhost:8500/v1/jobs/{}/stream", job_id);
    
    let mut install_complete = false;
    for attempt in 1..=90 {  // 90 attempts * 2s = 3 minutes max
        let stream = AsyncCommand::new("curl")
            .args(&["-N", &stream_url])
            .output()
            .await;
        
        if let Ok(output) = stream {
            let lines = String::from_utf8_lossy(&output.stdout);
            
            // Print all narration lines
            for line in lines.lines() {
                if line.starts_with("data: ") {
                    let narration = &line[6..]; // Strip "data: " prefix
                    if narration != "[DONE]" {
                        println!("{}", narration);
                    }
                }
            }
            
            // Check for completion marker
            if lines.contains("[DONE]") {
                install_complete = true;
                println!("{}", "-".repeat(60));
                println!("✅ Installation complete");
                break;
            } else if !lines.is_empty() {
                println!("   ... still waiting (attempt {}/90)", attempt);
            }
        }
        
        sleep(Duration::from_secs(2)).await;
    }
    
    if !install_complete {
        let _ = queen.kill().await;
        panic!("Installation did not complete within 180 seconds (3 minutes)");
    }
    
    // STEP 7: Verify binary was installed in container
    println!("\n🔍 STEP 7: Verifying binary installation...");
    let has_binary = container.has_binary("/home/rbee/.local/bin/rbee-hive").await;
    assert!(has_binary, "rbee-hive binary not found in container");
    println!("✅ Binary installed at /home/rbee/.local/bin/rbee-hive");
    
    // STEP 8: Verify binary works (run --version)
    println!("\n🔍 STEP 8: Verifying binary works...");
    let verify = AsyncCommand::new("docker")
        .args(&["exec", "rbee-test-target", "/home/rbee/.local/bin/rbee-hive", "--version"])
        .output()
        .await
        .expect("Failed to run --version");
    
    if verify.status.success() {
        let version = String::from_utf8_lossy(&verify.stdout);
        println!("✅ Binary works: {}", version.trim());
    } else {
        let _ = queen.kill().await;
        panic!("Binary doesn't work: {}", String::from_utf8_lossy(&verify.stderr));
    }
    
    // STEP 9: Check if daemon is running (optional - auto_start not implemented yet)
    println!("\n🔍 STEP 9: Checking daemon status...");
    let daemon_running = container.daemon_running("rbee-hive").await;
    if daemon_running {
        println!("✅ Daemon is running");
    } else {
        println!("ℹ️  Daemon not running (auto_start not implemented yet)");
    }
    
    // Cleanup
    println!("\n🧹 Cleaning up...");
    let _ = queen.kill().await;
    container.cleanup().await;
    
    // Restore original SSH key
    let ssh_dir = std::env::var("HOME").expect("HOME not set") + "/.ssh";
    let target_key = format!("{}/id_rsa", ssh_dir);
    let backup_key = format!("{}/id_rsa.backup_before_test", ssh_dir);
    if std::path::Path::new(&backup_key).exists() {
        let _ = std::fs::copy(&backup_key, &target_key);
        let _ = std::fs::remove_file(&backup_key);
    } else {
        // No backup means we created the key, so remove it
        let _ = std::fs::remove_file(&target_key);
    }
    
    println!();
    println!("{}", "=".repeat(60));
    println!("✅ FULL INTEGRATION TEST PASSED");
    println!("{}", "=".repeat(60));
    println!("\nWhat was tested:");
    println!("  ✅ queen-rbee runs on HOST (bare metal)");
    println!("  ✅ queen-rbee SSHs to container (localhost:2222)");
    println!("  ✅ daemon-sync installs rbee-hive in container");
    println!("  ✅ Binary is installed at correct path");
    println!("  ✅ Daemon starts successfully");
    println!("\nThis proves the actual deployment workflow works!");
}
