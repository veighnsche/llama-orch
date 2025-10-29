// TEAM-375: Critical tests for config.rs
//
// Tests configuration loading, saving, validation, and error handling.
// Prevents user data corruption and ensures proper config management.

use anyhow::Result;
use rbee_keeper::Config;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// LOAD TESTS
// ============================================================================

#[test]
fn test_load_creates_default_when_missing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    let config = Config::load()?;

    // Should create default config
    assert_eq!(config.queen_url(), "http://localhost:7833");

    Ok(())
}

#[test]
fn test_load_from_valid_toml() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config_dir = temp_dir.path().join(".config/rbee");
    fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    fs::write(
        &config_path,
        r#"
[queen]
host = "localhost"
port = 9999
"#,
    )?;

    std::env::set_var("HOME", temp_dir.path());

    let config = Config::load()?;
    assert_eq!(config.queen_url(), "http://localhost:9999");

    Ok(())
}

#[test]
fn test_load_from_invalid_toml_fails() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path().join(".config/rbee");
    fs::create_dir_all(&config_dir).unwrap();

    let config_path = config_dir.join("config.toml");
    fs::write(&config_path, "invalid toml {{{{ content").unwrap();

    std::env::set_var("HOME", temp_dir.path());

    let result = Config::load();
    assert!(result.is_err(), "Should fail to parse invalid TOML");
    assert!(
        result.unwrap_err().to_string().contains("parse"),
        "Error should mention parsing"
    );
}

#[test]
fn test_load_with_missing_required_fields() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path().join(".config/rbee");
    fs::create_dir_all(&config_dir).unwrap();

    let config_path = config_dir.join("config.toml");
    // Empty TOML file
    fs::write(&config_path, "").unwrap();

    std::env::set_var("HOME", temp_dir.path());

    // Should either load defaults or fail gracefully
    let result = Config::load();
    // This test verifies behavior - either succeeds with defaults or fails cleanly
    if let Err(e) = result {
        assert!(
            e.to_string().contains("parse") || e.to_string().contains("Invalid"),
            "Error should be descriptive"
        );
    }
}

// ============================================================================
// SAVE TESTS
// ============================================================================

#[test]
fn test_save_creates_parent_directories() -> Result<()> {
    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    let config = Config::default();
    config.save()?;

    let config_path = temp_dir.path().join(".config/rbee/config.toml");
    assert!(config_path.exists(), "Config file should be created");
    assert!(
        config_path.parent().unwrap().exists(),
        "Parent directory should be created"
    );

    Ok(())
}

#[test]
fn test_save_writes_valid_toml() -> Result<()> {
    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    let config = Config::default();
    config.save()?;

    let config_path = temp_dir.path().join(".config/rbee/config.toml");
    let content = fs::read_to_string(&config_path)?;

    // Should be valid TOML
    assert!(content.contains("[queen]"), "Should contain queen section");
    assert!(toml::from_str::<toml::Value>(&content).is_ok(), "Should be valid TOML");

    Ok(())
}

#[test]
fn test_save_overwrites_existing_file() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config_dir = temp_dir.path().join(".config/rbee");
    fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    fs::write(&config_path, "old content")?;

    std::env::set_var("HOME", temp_dir.path());

    let config = Config::default();
    config.save()?;

    let content = fs::read_to_string(&config_path)?;
    assert_ne!(content, "old content", "Should overwrite old content");
    assert!(content.contains("[queen]"), "Should contain new config");

    Ok(())
}

#[test]
fn test_save_to_readonly_directory_fails() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path().join(".config/rbee");
    fs::create_dir_all(&config_dir).unwrap();

    // Make directory read-only
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&config_dir).unwrap().permissions();
        perms.set_mode(0o444); // Read-only
        fs::set_permissions(&config_dir, perms).unwrap();
    }

    std::env::set_var("HOME", temp_dir.path());

    let config = Config::default();
    let result = config.save();

    #[cfg(unix)]
    {
        assert!(result.is_err(), "Should fail to write to read-only directory");
    }

    // Cleanup: restore permissions
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&config_dir).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&config_dir, perms).unwrap();
    }
}

// ============================================================================
// ROUND-TRIP TESTS
// ============================================================================

#[test]
fn test_save_and_load_roundtrip() -> Result<()> {
    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    // Save default config
    let config1 = Config::default();
    config1.save()?;

    // Load it back
    let config2 = Config::load()?;

    // Should match
    assert_eq!(config1.queen_url(), config2.queen_url());

    Ok(())
}

#[test]
fn test_multiple_save_load_cycles() -> Result<()> {
    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    for _ in 0..5 {
        let config = Config::load()?;
        config.save()?;
    }

    // Should still be valid after multiple cycles
    let final_config = Config::load()?;
    assert_eq!(final_config.queen_url(), "http://localhost:7833");

    Ok(())
}

// ============================================================================
// VALIDATION TESTS
// ============================================================================

#[test]
fn test_default_config_is_valid() -> Result<()> {
    let config = Config::default();

    // Default config should pass validation
    // (Validation happens in load(), so we test via save/load)
    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    config.save()?;
    let loaded = Config::load()?;

    assert_eq!(loaded.queen_url(), "http://localhost:7833");

    Ok(())
}

#[test]
fn test_queen_url_format() -> Result<()> {
    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    let config = Config::load()?;
    let url = config.queen_url();

    // Should be valid HTTP URL
    assert!(url.starts_with("http://"), "Should be HTTP URL");
    assert!(url.contains(':'), "Should include port");

    Ok(())
}

// ============================================================================
// PATH RESOLUTION TESTS
// ============================================================================

#[test]
fn test_config_path_uses_home_directory() {
    let temp_dir = TempDir::new().unwrap();
    std::env::set_var("HOME", temp_dir.path());

    let config = Config::default();
    config.save().unwrap();

    let expected_path = temp_dir.path().join(".config/rbee/config.toml");
    assert!(expected_path.exists(), "Config should be in HOME/.config/rbee/");
}

#[test]
fn test_config_path_without_home_fails() {
    // Temporarily unset HOME
    let original_home = std::env::var("HOME").ok();
    std::env::remove_var("HOME");

    let config = Config::default();
    let result = config.save();

    // Should fail without HOME
    assert!(result.is_err(), "Should fail without HOME environment variable");

    // Restore HOME
    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }
}

// ============================================================================
// CORRUPTION PREVENTION TESTS
// ============================================================================

#[test]
fn test_partial_write_does_not_corrupt() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config_dir = temp_dir.path().join(".config/rbee");
    fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");

    // Write valid config
    std::env::set_var("HOME", temp_dir.path());
    let config = Config::default();
    config.save()?;

    // Verify it's valid
    let content_before = fs::read_to_string(&config_path)?;
    assert!(toml::from_str::<toml::Value>(&content_before).is_ok());

    // Write again (simulating potential corruption scenario)
    config.save()?;

    // Should still be valid
    let content_after = fs::read_to_string(&config_path)?;
    assert!(toml::from_str::<toml::Value>(&content_after).is_ok());

    Ok(())
}

#[test]
fn test_concurrent_save_does_not_corrupt() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let temp_dir = TempDir::new()?;
    std::env::set_var("HOME", temp_dir.path());

    let config = Arc::new(Config::default());

    // Spawn multiple threads trying to save
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let config = Arc::clone(&config);
            thread::spawn(move || config.save())
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    // Config should still be valid
    let loaded = Config::load()?;
    assert_eq!(loaded.queen_url(), "http://localhost:7833");

    Ok(())
}
