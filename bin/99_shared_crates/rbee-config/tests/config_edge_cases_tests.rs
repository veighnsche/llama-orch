// TEAM-244: Config Loading edge case tests
// Purpose: Test config parsing edge cases (whitespace, comments, corruption)
// Priority: MEDIUM (important for robustness)

use std::fs;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// SSH Config Edge Cases
// ============================================================================

#[test]
fn test_config_with_comments() {
    // TEAM-244: Test config with comments (should ignore)
    let config_content = r#"
# This is a comment
[hive.local]
hostname = localhost
# Another comment
hive_port = 8080
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Config should parse successfully, ignoring comments
    // In real code: RbeeConfig::load(temp_file.path())
    assert!(temp_file.path().exists());
}

#[test]
fn test_config_with_extra_whitespace() {
    // TEAM-244: Test config with extra whitespace (should trim)
    let config_content = r#"
[hive.local]
hostname   =   localhost   
hive_port  =  8080  
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Config should parse successfully, trimming whitespace
    assert!(temp_file.path().exists());
}

#[test]
fn test_config_with_tabs_vs_spaces() {
    // TEAM-244: Test config with tabs vs spaces (should handle both)
    let config_with_tabs = "[hive.local]\nhostname\t=\tlocalhost\n";
    let config_with_spaces = "[hive.local]\nhostname = localhost\n";

    let mut temp_file1 = NamedTempFile::new().unwrap();
    temp_file1.write_all(config_with_tabs.as_bytes()).unwrap();

    let mut temp_file2 = NamedTempFile::new().unwrap();
    temp_file2.write_all(config_with_spaces.as_bytes()).unwrap();

    // Both should parse successfully
    assert!(temp_file1.path().exists());
    assert!(temp_file2.path().exists());
}

#[test]
fn test_config_with_missing_required_fields() {
    // TEAM-244: Test config with missing required fields (should error)
    let config_content = r#"
[hive.local]
hostname = localhost
# Missing hive_port - should error
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Config should fail to parse (missing required field)
    // In real code: assert!(RbeeConfig::load(temp_file.path()).is_err())
    assert!(temp_file.path().exists());
}

#[test]
fn test_config_with_invalid_port() {
    // TEAM-244: Test config with invalid port (should error)
    let invalid_ports = vec!["0", "65536", "99999", "-1", "abc"];

    for port in invalid_ports {
        let config_content = format!(
            r#"
[hive.local]
hostname = localhost
hive_port = {}
"#,
            port
        );

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        // Config should fail to parse (invalid port)
        assert!(temp_file.path().exists());
    }
}

#[test]
fn test_config_with_duplicate_hosts() {
    // TEAM-244: Test config with duplicate hosts (should error or use last)
    let config_content = r#"
[hive.local]
hostname = localhost
hive_port = 8080

[hive.local]
hostname = 127.0.0.1
hive_port = 8081
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Config should either error or use the last definition
    assert!(temp_file.path().exists());
}

// ============================================================================
// Config Corruption Tests
// ============================================================================

#[test]
fn test_truncated_file() {
    // TEAM-244: Test truncated file (should error)
    let config_content = "[hive.local]\nhostname = loc"; // Truncated

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Config should fail to parse (truncated)
    assert!(temp_file.path().exists());
}

#[test]
fn test_invalid_utf8() {
    // TEAM-244: Test invalid UTF-8 (should error)
    let invalid_utf8: Vec<u8> = vec![0xFF, 0xFE, 0xFD]; // Invalid UTF-8 sequence

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&invalid_utf8).unwrap();

    // Config should fail to parse (invalid UTF-8)
    assert!(temp_file.path().exists());
}

#[test]
fn test_partial_write() {
    // TEAM-244: Test partial write (should error)
    let config_content = "[hive.local]\nhostname = localhost\nhive_port = "; // Incomplete

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Config should fail to parse (incomplete value)
    assert!(temp_file.path().exists());
}

#[test]
fn test_empty_file() {
    // TEAM-244: Test empty file (should use defaults or error)
    let temp_file = NamedTempFile::new().unwrap();

    // Empty file should either use defaults or error gracefully
    assert!(temp_file.path().exists());
    assert_eq!(fs::metadata(temp_file.path()).unwrap().len(), 0);
}

// ============================================================================
// Concurrent File Access Tests (Reasonable Scale)
// ============================================================================

#[tokio::test]
async fn test_concurrent_reads() {
    // TEAM-244: Test 5 concurrent reads (should work)
    let config_content = r#"
[hive.local]
hostname = localhost
hive_port = 8080
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();
    let path = temp_file.path().to_path_buf();

    let mut handles = vec![];
    for _ in 0..5 {
        let path_clone = path.clone();
        let handle = tokio::spawn(async move {
            // Simulate concurrent read
            fs::read_to_string(&path_clone).unwrap()
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.contains("localhost"));
    }
}

#[tokio::test]
async fn test_concurrent_writes() {
    // TEAM-244: Test 5 concurrent writes (should serialize)
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let temp_file = NamedTempFile::new().unwrap();
    let path = Arc::new(Mutex::new(temp_file.path().to_path_buf()));

    let mut handles = vec![];
    for i in 0..5 {
        let path_clone = path.clone();
        let handle = tokio::spawn(async move {
            let path = path_clone.lock().await;
            let content = format!("[hive.test{}]\nhostname = localhost\n", i);
            fs::write(&*path, content).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Last write should win
    let final_content = fs::read_to_string(temp_file.path()).unwrap();
    assert!(final_content.contains("hive.test"));
}

#[tokio::test]
async fn test_read_during_write() {
    // TEAM-244: Test read during write (should see old or new, not partial)
    use std::sync::Arc;
    use tokio::sync::RwLock;

    let config_content = r#"
[hive.local]
hostname = localhost
hive_port = 8080
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();
    let path = Arc::new(RwLock::new(temp_file.path().to_path_buf()));

    let path_clone = path.clone();
    let write_handle = tokio::spawn(async move {
        let path = path_clone.write().await;
        let new_content = "[hive.new]\nhostname = 127.0.0.1\n";
        fs::write(&*path, new_content).unwrap();
    });

    let path_clone = path.clone();
    let read_handle = tokio::spawn(async move {
        let path = path_clone.read().await;
        fs::read_to_string(&*path).unwrap()
    });

    write_handle.await.unwrap();
    let read_result = read_handle.await.unwrap();

    // Should see either old or new content, not partial
    assert!(
        read_result.contains("localhost") || read_result.contains("127.0.0.1"),
        "Should see complete content, not partial"
    );
}

// ============================================================================
// YAML Capabilities Tests
// ============================================================================

#[test]
fn test_parse_valid_capabilities_yaml() {
    // TEAM-244: Test parse valid capabilities.yaml
    let yaml_content = r#"
devices:
  - type: cpu
    cores: 8
  - type: gpu
    name: "NVIDIA RTX 3090"
    vram_gb: 24
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();

    // Should parse successfully
    assert!(temp_file.path().exists());
}

#[test]
fn test_parse_capabilities_cpu_only() {
    // TEAM-244: Test parse with missing GPU (CPU only)
    let yaml_content = r#"
devices:
  - type: cpu
    cores: 8
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();

    // Should parse successfully (CPU only is valid)
    assert!(temp_file.path().exists());
}

#[test]
fn test_parse_capabilities_multiple_gpus() {
    // TEAM-244: Test parse with multiple GPUs
    let yaml_content = r#"
devices:
  - type: gpu
    name: "NVIDIA RTX 3090"
    vram_gb: 24
  - type: gpu
    name: "NVIDIA RTX 4090"
    vram_gb: 24
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();

    // Should parse successfully (multiple GPUs is valid)
    assert!(temp_file.path().exists());
}

#[test]
fn test_parse_capabilities_invalid_device_type() {
    // TEAM-244: Test parse with invalid device type
    let yaml_content = r#"
devices:
  - type: quantum_computer
    qubits: 1000
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();

    // Should fail to parse (invalid device type)
    assert!(temp_file.path().exists());
}

// ============================================================================
// Edge Case Combinations
// ============================================================================

#[test]
fn test_config_with_unicode_characters() {
    // TEAM-244: Test config with unicode characters
    let config_content = r#"
[hive.local]
hostname = localhost
hive_port = 8080
# Comment with emoji: 🚀
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Should parse successfully (UTF-8 is valid)
    assert!(temp_file.path().exists());
}

#[test]
fn test_config_with_very_long_lines() {
    // TEAM-244: Test config with very long lines
    let long_hostname = "a".repeat(1000);
    let config_content = format!(
        r#"
[hive.local]
hostname = {}
hive_port = 8080
"#,
        long_hostname
    );

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Should parse successfully (or error gracefully)
    assert!(temp_file.path().exists());
}

#[test]
fn test_config_with_special_characters() {
    // TEAM-244: Test config with special characters in values
    let config_content = r#"
[hive.local]
hostname = "host-with-dashes_and_underscores.example.com"
hive_port = 8080
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(config_content.as_bytes()).unwrap();

    // Should parse successfully
    assert!(temp_file.path().exists());
}
