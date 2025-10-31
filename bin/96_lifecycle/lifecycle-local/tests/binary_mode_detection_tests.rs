//! Binary mode detection tests
//!
//! TEAM-378: Phase 2 - Tests for get_binary_mode() and is_release_binary()

use lifecycle_local::utils::binary::{get_binary_mode, is_release_binary};
use std::path::PathBuf;

#[test]
fn test_detect_debug_binary() {
    let path = PathBuf::from("target/debug/queen-rbee");
    if path.exists() {
        let mode = get_binary_mode(&path).expect("Failed to get binary mode");
        assert_eq!(mode, "debug", "Debug binary should return 'debug' mode");
        
        let is_release = is_release_binary(&path).expect("Failed to check if release");
        assert!(!is_release, "Debug binary should not be identified as release");
    } else {
        println!("⚠️  Skipping test: target/debug/queen-rbee not found");
    }
}

#[test]
fn test_detect_release_binary() {
    let path = PathBuf::from("target/release/queen-rbee");
    if path.exists() {
        let mode = get_binary_mode(&path).expect("Failed to get binary mode");
        assert_eq!(mode, "release", "Release binary should return 'release' mode");
        
        let is_release = is_release_binary(&path).expect("Failed to check if release");
        assert!(is_release, "Release binary should be identified as release");
    } else {
        println!("⚠️  Skipping test: target/release/queen-rbee not found");
    }
}

#[test]
fn test_missing_binary() {
    let path = PathBuf::from("/nonexistent/binary");
    let result = get_binary_mode(&path);
    assert!(result.is_err(), "Should fail for missing binary");
    
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("Failed to execute"),
        "Error should mention execution failure"
    );
}

#[test]
fn test_binary_without_build_info() {
    // Use a system binary that doesn't have --build-info
    let path = PathBuf::from("/bin/ls");
    let result = get_binary_mode(&path);
    assert!(result.is_err(), "Should fail for binary without --build-info");
    
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("does not support --build-info") || error_msg.contains("Invalid build mode"),
        "Error should mention --build-info support or invalid mode"
    );
}

#[test]
fn test_rbee_hive_debug() {
    let path = PathBuf::from("target/debug/rbee-hive");
    if path.exists() {
        let mode = get_binary_mode(&path).expect("Failed to get binary mode");
        assert_eq!(mode, "debug", "Debug rbee-hive should return 'debug' mode");
    } else {
        println!("⚠️  Skipping test: target/debug/rbee-hive not found");
    }
}

#[test]
fn test_rbee_hive_release() {
    let path = PathBuf::from("target/release/rbee-hive");
    if path.exists() {
        let mode = get_binary_mode(&path).expect("Failed to get binary mode");
        assert_eq!(mode, "release", "Release rbee-hive should return 'release' mode");
    } else {
        println!("⚠️  Skipping test: target/release/rbee-hive not found");
    }
}

#[test]
fn test_llm_worker_debug() {
    let path = PathBuf::from("target/debug/llm-worker-rbee");
    if path.exists() {
        let mode = get_binary_mode(&path).expect("Failed to get binary mode");
        assert_eq!(mode, "debug", "Debug llm-worker-rbee should return 'debug' mode");
    } else {
        println!("⚠️  Skipping test: target/debug/llm-worker-rbee not found");
    }
}

#[test]
fn test_llm_worker_release() {
    let path = PathBuf::from("target/release/llm-worker-rbee");
    if path.exists() {
        let mode = get_binary_mode(&path).expect("Failed to get binary mode");
        assert_eq!(mode, "release", "Release llm-worker-rbee should return 'release' mode");
    } else {
        println!("⚠️  Skipping test: target/release/llm-worker-rbee not found");
    }
}
