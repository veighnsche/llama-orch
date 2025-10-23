//! File-based configuration for rbee
//!
//! Created by: TEAM-193
//!
//! This crate provides file-based configuration management following Unix best practices.
//! It replaces the SQLite-based hive-catalog with human-editable config files.
//!
//! # Configuration Files
//!
//! All config files are stored in `~/.config/rbee/`:
//!
//! - `config.toml` - Queen-level settings (port, log level, etc.)
//! - `hives.conf` - SSH/hive definitions (SSH config style)
//! - `capabilities.yaml` - Auto-generated device capabilities cache
//!
//! # Usage
//!
//! ```rust,no_run
//! use rbee_config::RbeeConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load all config files
//! let config = RbeeConfig::load()?;
//!
//! // Access queen settings
//! println!("Queen port: {}", config.queen.queen.port);
//!
//! // TEAM-278: Access hive by alias (new declarative API)
//! if let Some(hive) = config.hives.get_hive("localhost") {
//!     println!("Hive: {}@{}", hive.ssh_user, hive.hostname);
//! }
//!
//! // Check capabilities
//! if let Some(caps) = config.capabilities.get("localhost") {
//!     println!("GPUs: {}", caps.gpu_count());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # SSH Config Style
//!
//! The `hives.conf` file uses SSH config syntax:
//!
//! ```text
//! Host localhost
//!     HostName 127.0.0.1
//!     Port 22
//!     User vince
//!     HivePort 8081
//!
//! Host workstation
//!     HostName 192.168.1.100
//!     User admin
//!     HivePort 8081
//!     BinaryPath /usr/local/bin/rbee-hive
//! ```

mod capabilities;
mod error;
// TEAM-278: DELETED old SSH-style hives_config module - REPLACED with declarative
// mod hives_config; // DELETED - v0.1.0 breaking change
mod queen_config;
mod validation;

// TEAM-278: Declarative configuration for lifecycle management
pub mod declarative;

pub use capabilities::{CapabilitiesCache, DeviceInfo, DeviceType, HiveCapabilities};
pub use error::{ConfigError, Result};
// TEAM-278: DELETED old SSH-style exports - REPLACED with declarative
// pub use hives_config::{HiveEntry, HivesConfig}; // DELETED
pub use queen_config::{QueenConfig, QueenSettings, RuntimeSettings};
pub use validation::{
    preflight_validation, validate_capabilities_sync, validate_hives_config, ValidationResult,
};

// TEAM-278: Export declarative types as primary API (no aliases needed)
pub use declarative::{HiveConfig, HivesConfig, WorkerConfig};

use std::path::{Path, PathBuf};

/// Main configuration structure
/// 
/// TEAM-278: BREAKING CHANGE - hives field now uses declarative::HivesConfig (Vec-based)
/// instead of old SSH-style HivesConfig (HashMap-based)
#[derive(Debug, Clone)]
pub struct RbeeConfig {
    /// Queen-level configuration
    pub queen: QueenConfig,
    /// Hives configuration (TEAM-278: Now uses declarative::HivesConfig)
    pub hives: HivesConfig,
    /// Capabilities cache
    pub capabilities: CapabilitiesCache,
}

impl RbeeConfig {
    /// Load all config files from ~/.config/rbee/
    ///
    /// # Errors
    ///
    /// Returns an error if config files cannot be read or parsed
    pub fn load() -> Result<Self> {
        let config_dir = Self::config_dir()?;
        Self::load_from_dir(&config_dir)
    }

    /// Load from specific directory
    ///
    /// TEAM-278: BREAKING CHANGE - now loads declarative TOML config instead of SSH config
    ///
    /// # Errors
    ///
    /// Returns an error if config files cannot be read or parsed
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let queen = QueenConfig::load(&dir.join("config.toml"))?;
        // TEAM-278: Load declarative TOML config (not SSH config)
        let hives = HivesConfig::load_from(&dir.join("hives.conf"))?;
        let capabilities = CapabilitiesCache::load(&dir.join("capabilities.yaml"))?;

        Ok(Self { queen, hives, capabilities })
    }

    /// Get config directory path (~/.config/rbee/)
    ///
    /// # Errors
    ///
    /// Returns an error if HOME environment variable is not set or directory cannot be created
    pub fn config_dir() -> Result<PathBuf> {
        let home = std::env::var("HOME").map_err(|_| {
            ConfigError::InvalidConfig("HOME environment variable not set".to_string())
        })?;

        let config_dir = PathBuf::from(home).join(".config").join("rbee");

        // Create directory if it doesn't exist
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir)?;
        }

        Ok(config_dir)
    }

    /// Validate all config files
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails critically
    pub fn validate(&self) -> Result<ValidationResult> {
        // TEAM-195: Validate queen config first
        self.queen.validate()?;

        preflight_validation(&self.hives, &self.capabilities)
    }

    /// Save capabilities cache
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn save_capabilities(&self) -> Result<()> {
        self.capabilities.save()
    }

    // TEAM-278: Helper methods for backward API compatibility
    // (but using new declarative HivesConfig internally)

    /// Get hive by alias
    pub fn get_hive(&self, alias: &str) -> Option<&HiveConfig> {
        self.hives.get_hive(alias)
    }

    /// Get all hives
    pub fn all_hives(&self) -> &[HiveConfig] {
        &self.hives.hives
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_config_dir() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Create config.toml
        let config_toml = r#"
[queen]
port = 8080
log_level = "info"

[runtime]
max_concurrent_operations = 10
"#;
        fs::write(dir.path().join("config.toml"), config_toml).unwrap();

        // TEAM-278: Create hives.conf (TOML format, not SSH config)
        let hives_conf = r#"
[[hive]]
alias = "localhost"
hostname = "127.0.0.1"
ssh_user = "vince"
ssh_port = 22
hive_port = 8081

[[hive]]
alias = "workstation"
hostname = "192.168.1.100"
ssh_user = "admin"
hive_port = 8081
"#;
        fs::write(dir.path().join("hives.conf"), hives_conf).unwrap();

        // Create empty capabilities.yaml
        let capabilities_yaml = r#"
last_updated: "2025-10-21T20:00:00Z"
hives: {}
"#;
        fs::write(dir.path().join("capabilities.yaml"), capabilities_yaml).unwrap();

        dir
    }

    #[test]
    fn test_load_from_dir() {
        let dir = create_test_config_dir();
        let config = RbeeConfig::load_from_dir(dir.path()).unwrap();

        assert_eq!(config.queen.queen.port, 8080);
        // TEAM-278: Use new declarative API
        assert_eq!(config.hives.len(), 2);
        assert!(config.get_hive("localhost").is_some());
        assert!(config.get_hive("workstation").is_some());
    }

    #[test]
    fn test_validate() {
        let dir = create_test_config_dir();
        let config = RbeeConfig::load_from_dir(dir.path()).unwrap();

        let result = config.validate().unwrap();
        assert!(result.is_valid());
        assert!(result.has_warnings()); // Hives have no capabilities cached
    }

    #[test]
    fn test_config_dir_creation() {
        let temp = TempDir::new().unwrap();
        let test_home = temp.path().to_str().unwrap();

        std::env::set_var("HOME", test_home);

        let config_dir = RbeeConfig::config_dir().unwrap();
        assert!(config_dir.exists());
        assert!(config_dir.ends_with(".config/rbee"));
    }

    #[test]
    fn test_save_capabilities() {
        let dir = create_test_config_dir();
        let mut config = RbeeConfig::load_from_dir(dir.path()).unwrap();

        let caps = HiveCapabilities::new(
            "localhost".to_string(),
            vec![DeviceInfo {
                id: "GPU-0".to_string(),
                name: "RTX 4090".to_string(),
                vram_gb: 24,
                compute_capability: Some("8.9".to_string()),
                device_type: capabilities::DeviceType::Gpu,
            }],
            "http://localhost:8081".to_string(),
        );

        config.capabilities.update_hive("localhost", caps);
        config.save_capabilities().unwrap();

        // Reload and verify
        let reloaded = RbeeConfig::load_from_dir(dir.path()).unwrap();
        assert!(reloaded.capabilities.contains("localhost"));
        let caps = reloaded.capabilities.get("localhost").unwrap();
        assert_eq!(caps.devices.len(), 1);
        // TEAM-278: Verify hives still load correctly
        assert_eq!(reloaded.hives.len(), 2);
    }
}
