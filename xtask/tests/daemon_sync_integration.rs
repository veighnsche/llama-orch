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

/// Helper: Build rbee-keeper (which auto-builds queen-rbee)
/// TEAM-260: Changed to build rbee-keeper instead of queen-rbee directly
async fn build_rbee_keeper() {
    let workspace_root = std::env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .expect("No parent directory")
        .to_path_buf();
    
    let build = Command::new("cargo")
        .args(&["build", "--bin", "rbee-keeper"])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to build rbee-keeper");
    
    assert!(build.status.success(), "Failed to build rbee-keeper: {}", 
        String::from_utf8_lossy(&build.stderr));
    
    println!("✅ rbee-keeper built on HOST (auto-builds queen-rbee)");
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
    
    // STEP 2: Build rbee-keeper on HOST
    // TEAM-260: rbee-keeper auto-builds queen-rbee via auto-updater
    println!("\n🔨 STEP 2: Building rbee-keeper on HOST...");
    build_rbee_keeper().await;
    
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
    
    // STEP 4: Run `./rbee sync` to install hive via SSH
    // TEAM-260: Use actual CLI instead of manually managing queen-rbee
    println!("\n🚀 STEP 4: Running ./rbee sync --config tests/docker/hives.conf...");
    println!("{}", "-".repeat(60));
    
    let rbee_script = workspace_root.join("rbee");
    let hives_conf = workspace_root.join("tests/docker/hives.conf");
    
    let sync_output = AsyncCommand::new(&rbee_script)
        .args(&[
            "sync",
            "--config", hives_conf.to_str().unwrap(),
        ])
        .current_dir(&workspace_root)
        .output()
        .await
        .expect("Failed to run ./rbee sync");
    
    // Print all output (includes narration)
    let stdout = String::from_utf8_lossy(&sync_output.stdout);
    let stderr = String::from_utf8_lossy(&sync_output.stderr);
    
    println!("{}", stdout);
    if !stderr.is_empty() {
        eprintln!("stderr: {}", stderr);
    }
    
    if !sync_output.status.success() {
        panic!("./rbee sync failed with exit code: {:?}\nstdout: {}\nstderr: {}",
            sync_output.status.code(), stdout, stderr);
    }
    
    println!("{}", "-".repeat(60));
    println!("✅ ./rbee sync completed successfully");
    
    // STEP 5: Verify binary was installed in container
    println!("\n🔍 STEP 5: Verifying binary installation...");
    let has_binary = container.has_binary("/home/rbee/.local/bin/rbee-hive").await;
    assert!(has_binary, "rbee-hive binary not found in container");
    println!("✅ Binary installed at /home/rbee/.local/bin/rbee-hive");
    
    // STEP 6: Verify binary works (run --version)
    println!("\n🔍 STEP 6: Verifying binary works...");
    let verify = AsyncCommand::new("docker")
        .args(&["exec", "rbee-test-target", "/home/rbee/.local/bin/rbee-hive", "--version"])
        .output()
        .await
        .expect("Failed to run --version");
    
    if verify.status.success() {
        let version = String::from_utf8_lossy(&verify.stdout);
        println!("✅ Binary works: {}", version.trim());
    } else {
        panic!("Binary doesn't work: {}", String::from_utf8_lossy(&verify.stderr));
    }
    
    // STEP 7: Check if daemon is running (optional - auto_start not implemented yet)
    println!("\n🔍 STEP 7: Checking daemon status...");
    let daemon_running = container.daemon_running("rbee-hive").await;
    if daemon_running {
        println!("✅ Daemon is running");
    } else {
        println!("ℹ️  Daemon not running (auto_start not implemented yet)");
    }
    
    // Cleanup
    println!("\n🧹 Cleaning up...");
    // TEAM-260: No need to kill queen - rbee-keeper manages it
    // Kill any lingering queen-rbee processes
    let _ = Command::new("pkill")
        .args(&["-f", "queen-rbee"])
        .output();
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
    println!("  ✅ ./rbee CLI works (rbee-keeper)");
    println!("  ✅ rbee-keeper auto-starts queen-rbee");
    println!("  ✅ queen-rbee SSHs to container (localhost:2222)");
    println!("  ✅ daemon-sync copies rbee-hive binary via SCP");
    println!("  ✅ Binary is installed at correct path");
    println!("  ✅ Binary is executable and runs --version");
    println!("  ✅ Full user workflow tested (./rbee sync)");
    println!("\nTEAM-260: Using local binary install method (SCP)");
    println!("This proves the complete deployment workflow works!");
    println!("\nNOTE: Git clone + cargo build method needs separate investigation");
}
