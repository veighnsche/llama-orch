//! Integration tests for rbee-config
//!
//! Created by: TEAM-193

use rbee_config::{DeviceInfo, DeviceType, HiveCapabilities, RbeeConfig};
use std::fs;
use tempfile::TempDir;

/// Create a complete test config directory
fn create_complete_config() -> TempDir {
    let dir = TempDir::new().unwrap();

    // Copy valid_hives.conf
    let hives_conf = include_str!("fixtures/valid_hives.conf");
    fs::write(dir.path().join("hives.conf"), hives_conf).unwrap();

    // Create config.toml
    let config_toml = r#"
[queen]
port = 8080
log_level = "info"

[runtime]
max_concurrent_operations = 10
"#;
    fs::write(dir.path().join("config.toml"), config_toml).unwrap();

    // Copy sample_capabilities.yaml
    let capabilities = include_str!("fixtures/sample_capabilities.yaml");
    fs::write(dir.path().join("capabilities.yaml"), capabilities).unwrap();

    dir
}

#[test]
fn test_load_complete_config() {
    let dir = create_complete_config();
    let config = RbeeConfig::load_from_dir(dir.path()).unwrap();

    // Verify queen config
    assert_eq!(config.queen.queen.port, 8080);
    assert_eq!(config.queen.queen.log_level, "info");
    assert_eq!(config.queen.runtime.max_concurrent_operations, 10);

    // Verify hives
    assert_eq!(config.hives.len(), 3);
    assert!(config.hives.contains("localhost"));
    assert!(config.hives.contains("workstation"));
    assert!(config.hives.contains("gpu-cloud"));

    // TEAM-278: Use new declarative API
    let localhost = config.hives.get_hive("localhost").unwrap();
    assert_eq!(localhost.hostname, "127.0.0.1");
    assert_eq!(localhost.ssh_port, 22);
    assert_eq!(localhost.ssh_user, "vince");
    assert_eq!(localhost.hive_port, 8081);

    // Verify capabilities
    assert_eq!(config.capabilities.len(), 2);
    let localhost_caps = config.capabilities.get("localhost").unwrap();
    assert_eq!(localhost_caps.devices.len(), 2);
    assert_eq!(localhost_caps.gpu_count(), 2);
    assert_eq!(localhost_caps.total_vram_gb(), 36);
}

#[test]
fn test_validation_with_complete_config() {
    let dir = create_complete_config();
    let config = RbeeConfig::load_from_dir(dir.path()).unwrap();

    let result = config.validate().unwrap();
    assert!(result.is_valid());

    // Should have warning about gpu-cloud not having capabilities
    assert!(result.has_warnings());
    assert!(result.warnings.iter().any(|w| w.contains("gpu-cloud")));
}

#[test]
fn test_update_and_save_capabilities() {
    let dir = create_complete_config();
    let mut config = RbeeConfig::load_from_dir(dir.path()).unwrap();

    // Add capabilities for gpu-cloud
    let caps = HiveCapabilities::new(
        "gpu-cloud".to_string(),
        vec![DeviceInfo {
            id: "GPU-0".to_string(),
            name: "NVIDIA H100".to_string(),
            vram_gb: 80,
            compute_capability: Some("9.0".to_string()),
            device_type: DeviceType::Gpu,
        }],
        "http://localhost:8081".to_string(),
    );

    config.capabilities.update_hive("gpu-cloud", caps);
    config.save_capabilities().unwrap();

    // Reload and verify
    let reloaded = RbeeConfig::load_from_dir(dir.path()).unwrap();
    assert_eq!(reloaded.capabilities.len(), 3);

    let gpu_cloud_caps = reloaded.capabilities.get("gpu-cloud").unwrap();
    assert_eq!(gpu_cloud_caps.devices.len(), 1);
    assert_eq!(gpu_cloud_caps.devices[0].name, "NVIDIA H100");

    // Validation should now pass without warnings
    let result = reloaded.validate().unwrap();
    assert!(result.is_valid());
    assert!(!result.has_warnings());
}

#[test]
fn test_load_invalid_hives_conf() {
    let dir = TempDir::new().unwrap();

    // Copy invalid_hives.conf
    let invalid_hives = include_str!("fixtures/invalid_hives.conf");
    fs::write(dir.path().join("hives.conf"), invalid_hives).unwrap();

    // Create minimal config.toml
    fs::write(dir.path().join("config.toml"), "[queen]\n").unwrap();

    // Should fail due to duplicate alias
    let result = RbeeConfig::load_from_dir(dir.path());
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("duplicate"));
}

#[test]
fn test_missing_config_files() {
    let dir = TempDir::new().unwrap();

    // Load with all files missing - should use defaults
    let config = RbeeConfig::load_from_dir(dir.path()).unwrap();

    // Queen config should have defaults
    assert_eq!(config.queen.queen.port, 8080);
    assert_eq!(config.queen.queen.log_level, "info");

    // Hives and capabilities should be empty
    assert!(config.hives.is_empty());
    assert!(config.capabilities.is_empty());
}

#[test]
fn test_hive_entry_access() {
    let dir = create_complete_config();
    let config = RbeeConfig::load_from_dir(dir.path()).unwrap();

    // TEAM-278: Test new declarative API
    let localhost = config.hives.get_hive("localhost").unwrap();
    assert_eq!(localhost.alias, "localhost");

    // Test all (direct access to hives vec)
    let all_hives = &config.hives.hives;
    assert_eq!(all_hives.len(), 3);

    // Test aliases
    let aliases = config.hives.aliases();
    assert_eq!(aliases.len(), 3);
    assert!(aliases.contains(&"localhost".to_string()));
}

#[test]
fn test_capabilities_operations() {
    let dir = create_complete_config();
    let mut config = RbeeConfig::load_from_dir(dir.path()).unwrap();

    // Test contains
    assert!(config.capabilities.contains("localhost"));
    assert!(!config.capabilities.contains("nonexistent"));

    // Test remove
    let removed = config.capabilities.remove("localhost");
    assert!(removed.is_some());
    assert!(!config.capabilities.contains("localhost"));
    assert_eq!(config.capabilities.len(), 1);

    // Save and reload
    config.save_capabilities().unwrap();
    let reloaded = RbeeConfig::load_from_dir(dir.path()).unwrap();
    assert_eq!(reloaded.capabilities.len(), 1);
    assert!(!reloaded.capabilities.contains("localhost"));
}

#[test]
fn test_yaml_header_preservation() {
    let dir = TempDir::new().unwrap();
    let mut config = RbeeConfig::load_from_dir(dir.path()).unwrap();

    let caps = HiveCapabilities::new(
        "test".to_string(),
        vec![DeviceInfo {
            id: "GPU-0".to_string(),
            name: "Test GPU".to_string(),
            vram_gb: 8,
            compute_capability: None,
            device_type: DeviceType::Gpu,
        }],
        "http://localhost:8081".to_string(),
    );

    config.capabilities.update_hive("test", caps);
    config.save_capabilities().unwrap();

    // Read file and verify header
    let content = fs::read_to_string(dir.path().join("capabilities.yaml")).unwrap();
    assert!(content.starts_with("# Auto-generated by queen-rbee"));
    assert!(content.contains("# DO NOT EDIT MANUALLY"));
}

#[test]
fn test_validation_errors() {
    let dir = TempDir::new().unwrap();

    // Create hive with invalid config
    let invalid_hives = r#"
Host invalid
    HostName 
    Port 0
    User 
    HivePort 0
"#;
    fs::write(dir.path().join("hives.conf"), invalid_hives).unwrap();
    fs::write(dir.path().join("config.toml"), "[queen]\n").unwrap();

    // Should fail during parsing due to missing required fields
    let result = RbeeConfig::load_from_dir(dir.path());
    assert!(result.is_err());
}
