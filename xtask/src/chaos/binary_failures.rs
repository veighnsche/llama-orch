// TEAM-252: Binary failure tests
// Purpose: Test behavior when binaries are missing or corrupt

use crate::integration::assertions::*;
use crate::integration::harness::TestHarness;
use std::fs;
use std::path::PathBuf;

#[tokio::test]
async fn test_binary_not_found() {
    // TEAM-252: Test rbee-hive binary not found

    let mut harness = TestHarness::new().await.unwrap();

    // Hide the binary (rename it temporarily)
    let binary_path = PathBuf::from("target/debug/rbee-hive");
    let hidden_path = PathBuf::from("target/debug/rbee-hive.hidden");

    if binary_path.exists() {
        fs::rename(&binary_path, &hidden_path).unwrap();
    }

    // Try to start hive
    let result = harness.run_command(&["hive", "start"]).await.unwrap();

    // Should fail with helpful error
    assert_failure(&result);
    assert_output_contains(&result, "binary not found");
    assert_output_contains(&result, "cargo build");

    // Restore binary
    if hidden_path.exists() {
        fs::rename(&hidden_path, &binary_path).unwrap();
    }

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_binary_not_found() {
    // TEAM-252: Test queen-rbee binary not found

    let mut harness = TestHarness::new().await.unwrap();

    // Hide the queen binary
    let binary_path = PathBuf::from("target/debug/queen-rbee");
    let hidden_path = PathBuf::from("target/debug/queen-rbee.hidden");

    if binary_path.exists() {
        fs::rename(&binary_path, &hidden_path).unwrap();
    }

    // Try to start queen
    let result = harness.run_command(&["queen", "start"]).await.unwrap();

    // Should fail with helpful error
    assert_failure(&result);
    assert_output_contains(&result, "binary not found");

    // Restore binary
    if hidden_path.exists() {
        fs::rename(&hidden_path, &binary_path).unwrap();
    }

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_keeper_binary_not_found() {
    // TEAM-252: Test rbee-keeper binary not found

    let binary_path = PathBuf::from("target/debug/rbee-keeper");
    let hidden_path = PathBuf::from("target/debug/rbee-keeper.hidden");

    if binary_path.exists() {
        fs::rename(&binary_path, &hidden_path).unwrap();
    }

    // Try to run any command (keeper is the CLI)
    let result = std::process::Command::new(&binary_path).args(&["queen", "start"]).output();

    // Should fail because binary doesn't exist
    assert!(result.is_err() || !result.unwrap().status.success());

    // Restore binary
    if hidden_path.exists() {
        fs::rename(&hidden_path, &binary_path).unwrap();
    }
}

#[tokio::test]
async fn test_binary_permission_denied() {
    // TEAM-252: Test binary with no execute permission

    let mut harness = TestHarness::new().await.unwrap();

    let binary_path = PathBuf::from("target/debug/rbee-hive");

    if binary_path.exists() {
        // Remove execute permission
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o644);
            fs::set_permissions(&binary_path, perms).unwrap();
        }

        // Try to start hive
        let result = harness.run_command(&["hive", "start"]).await.unwrap();

        // Should fail
        assert_failure(&result);

        // Restore permission
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o755);
            fs::set_permissions(&binary_path, perms).unwrap();
        }
    }

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_binary_corrupted() {
    // TEAM-252: Test corrupted binary

    let mut harness = TestHarness::new().await.unwrap();

    let binary_path = PathBuf::from("target/debug/rbee-hive");
    let backup_path = PathBuf::from("target/debug/rbee-hive.backup");

    if binary_path.exists() {
        // Backup original
        fs::copy(&binary_path, &backup_path).unwrap();

        // Corrupt the binary
        fs::write(&binary_path, "corrupted binary data").unwrap();

        // Try to start hive
        let result = harness.run_command(&["hive", "start"]).await.unwrap();

        // Should fail
        assert_failure(&result);

        // Restore original
        fs::copy(&backup_path, &binary_path).unwrap();
        fs::remove_file(&backup_path).unwrap();
    }

    harness.cleanup().await.unwrap();
}
