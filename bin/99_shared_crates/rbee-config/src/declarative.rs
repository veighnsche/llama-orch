//! Declarative configuration for rbee lifecycle management
//!
//! Created by: TEAM-278
//!
//! This module provides declarative configuration support for managing hives and workers
//! through a TOML-based config file. This enables desired-state management where the
//! config file is the source of truth and `rbee sync` brings the system to that state.
//!
//! # Configuration File
//!
//! Config is stored at `~/.config/rbee/hives.conf` in TOML format:
//!
//! ```toml
//! [[hive]]
//! alias = "gpu-server-1"
//! hostname = "192.168.1.100"
//! ssh_user = "vince"
//! ssh_port = 22
//! hive_port = 8600
//! auto_start = true
//!
//! workers = [
//!     { type = "vllm", version = "latest" },
//!     { type = "llama-cpp", version = "latest" },
//! ]
//!
//! [[hive]]
//! alias = "local-hive"
//! hostname = "localhost"
//! ssh_user = "vince"
//! workers = [
//!     { type = "llama-cpp", version = "latest" },
//! ]
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use rbee_config::declarative::HivesConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load from default location (~/.config/rbee/hives.conf)
//! let config = HivesConfig::load()?;
//!
//! // Validate config
//! config.validate()?;
//!
//! // Access hives
//! for hive in &config.hives {
//!     println!("Hive: {} @ {}", hive.alias, hive.hostname);
//!     for worker in &hive.workers {
//!         println!("  Worker: {} ({})", worker.worker_type, worker.version);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level declarative configuration
///
/// Contains all hives and their workers. This is the desired state
/// that `rbee sync` will reconcile against actual system state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HivesConfig {
    /// List of hives to manage
    #[serde(rename = "hive")]
    pub hives: Vec<HiveConfig>,
}

/// Single hive configuration
///
/// Defines a hive and all workers that should be installed on it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HiveConfig {
    /// Unique alias for this hive
    pub alias: String,

    /// Hostname or IP address
    pub hostname: String,

    /// SSH username
    pub ssh_user: String,

    /// SSH port (default: 22)
    #[serde(default = "default_ssh_port")]
    pub ssh_port: u16,

    /// Hive HTTP API port (default: 8600)
    #[serde(default = "default_hive_port")]
    pub hive_port: u16,

    /// Optional path to rbee-hive binary on remote system
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,

    /// Workers to install on this hive
    #[serde(default)]
    pub workers: Vec<WorkerConfig>,

    /// Auto-start hive after installation (default: true)
    #[serde(default = "default_true")]
    pub auto_start: bool,
}

/// Worker configuration
///
/// Defines a worker binary that should be installed on a hive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerConfig {
    /// Worker type (e.g., "vllm", "llama-cpp", "comfyui")
    #[serde(rename = "type")]
    pub worker_type: String,

    /// Version to install (e.g., "latest", "0.5.0")
    pub version: String,

    /// Optional custom binary path (if not using standard download)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
}

// Default value functions for serde
fn default_ssh_port() -> u16 {
    22
}

fn default_hive_port() -> u16 {
    8600
}

fn default_true() -> bool {
    true
}

impl HivesConfig {
    /// Load from default location (~/.config/rbee/hives.conf)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Config directory cannot be determined
    /// - File cannot be read
    /// - TOML parsing fails
    pub fn load() -> Result<Self> {
        let config_dir = Self::config_dir()?;
        let path = config_dir.join("hives.conf");
        Self::load_from(&path)
    }

    /// Load from specific path
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be read
    /// - TOML parsing fails
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.exists() {
            // Return empty config if file doesn't exist
            return Ok(Self { hives: Vec::new() });
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::ReadError { path: path.to_path_buf(), source: e })?;

        let config: Self = toml::from_str(&content).map_err(|e| ConfigError::InvalidConfig(
            format!("Failed to parse TOML: {}", e),
        ))?;

        Ok(config)
    }

    /// Save to default location (~/.config/rbee/hives.conf)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Config directory cannot be determined
    /// - File cannot be written
    /// - TOML serialization fails
    pub fn save(&self) -> Result<()> {
        let config_dir = Self::config_dir()?;
        let path = config_dir.join("hives.conf");
        self.save_to(&path)
    }

    /// Save to specific path
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be written
    /// - TOML serialization fails
    pub fn save_to(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            ConfigError::InvalidConfig(format!("Failed to serialize TOML: {}", e))
        })?;

        std::fs::write(path, content)
            .map_err(|e| ConfigError::ReadError { path: path.to_path_buf(), source: e })?;

        Ok(())
    }

    /// Get config directory path (~/.config/rbee/)
    ///
    /// # Errors
    ///
    /// Returns an error if HOME environment variable is not set
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

    /// Validate configuration
    ///
    /// Checks:
    /// - Unique hive aliases
    /// - Valid hostnames
    /// - Valid ports
    /// - Valid worker types
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate aliases
        let mut seen_aliases = std::collections::HashSet::new();
        for hive in &self.hives {
            if !seen_aliases.insert(&hive.alias) {
                return Err(ConfigError::DuplicateAlias { alias: hive.alias.clone() });
            }
        }

        // Validate each hive
        for hive in &self.hives {
            hive.validate()?;
        }

        Ok(())
    }

    /// Get hive by alias
    pub fn get_hive(&self, alias: &str) -> Option<&HiveConfig> {
        self.hives.iter().find(|h| h.alias == alias)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.hives.is_empty()
    }

    /// Number of hives
    pub fn len(&self) -> usize {
        self.hives.len()
    }

    /// Get all hive aliases
    pub fn aliases(&self) -> Vec<String> {
        self.hives.iter().map(|h| h.alias.clone()).collect()
    }

    /// Check if hive with alias exists
    pub fn contains(&self, alias: &str) -> bool {
        self.hives.iter().any(|h| h.alias == alias)
    }

    /// Get hive by alias (mutable)
    pub fn get_mut(&mut self, alias: &str) -> Option<&mut HiveConfig> {
        self.hives.iter_mut().find(|h| h.alias == alias)
    }
}

impl HiveConfig {
    /// Validate hive configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Alias is empty
    /// - Hostname is empty
    /// - SSH user is empty
    /// - Ports are invalid
    /// - Workers are invalid
    pub fn validate(&self) -> Result<()> {
        // Check alias
        if self.alias.is_empty() {
            return Err(ConfigError::InvalidConfig("Hive alias cannot be empty".to_string()));
        }

        // Check hostname
        if self.hostname.is_empty() {
            return Err(ConfigError::InvalidConfig(
                format!("Hive '{}': hostname cannot be empty", self.alias),
            ));
        }

        // Check SSH user
        if self.ssh_user.is_empty() {
            return Err(ConfigError::InvalidConfig(
                format!("Hive '{}': ssh_user cannot be empty", self.alias),
            ));
        }

        // Check ports
        if self.ssh_port == 0 {
            return Err(ConfigError::InvalidConfig(
                format!("Hive '{}': ssh_port cannot be 0", self.alias),
            ));
        }

        if self.hive_port == 0 {
            return Err(ConfigError::InvalidConfig(
                format!("Hive '{}': hive_port cannot be 0", self.alias),
            ));
        }

        // Validate workers
        for worker in &self.workers {
            worker.validate(&self.alias)?;
        }

        Ok(())
    }

    /// Get base URL for hive HTTP API
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.hostname, self.hive_port)
    }
}

impl WorkerConfig {
    /// Validate worker configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Worker type is empty
    /// - Version is empty
    pub fn validate(&self, hive_alias: &str) -> Result<()> {
        if self.worker_type.is_empty() {
            return Err(ConfigError::InvalidConfig(
                format!("Hive '{}': worker type cannot be empty", hive_alias),
            ));
        }

        if self.version.is_empty() {
            return Err(ConfigError::InvalidConfig(
                format!("Hive '{}': worker version cannot be empty", hive_alias),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_valid_config() {
        let toml = r#"
[[hive]]
alias = "gpu-1"
hostname = "192.168.1.100"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600
auto_start = true

workers = [
    { type = "vllm", version = "latest" },
    { type = "llama-cpp", version = "0.5.0" },
]

[[hive]]
alias = "local"
hostname = "localhost"
ssh_user = "vince"

workers = [
    { type = "llama-cpp", version = "latest" },
]
"#;

        let config: HivesConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.hives.len(), 2);

        let gpu1 = &config.hives[0];
        assert_eq!(gpu1.alias, "gpu-1");
        assert_eq!(gpu1.hostname, "192.168.1.100");
        assert_eq!(gpu1.ssh_user, "vince");
        assert_eq!(gpu1.ssh_port, 22);
        assert_eq!(gpu1.hive_port, 8600);
        assert!(gpu1.auto_start);
        assert_eq!(gpu1.workers.len(), 2);
        assert_eq!(gpu1.workers[0].worker_type, "vllm");
        assert_eq!(gpu1.workers[0].version, "latest");

        let local = &config.hives[1];
        assert_eq!(local.alias, "local");
        assert_eq!(local.hostname, "localhost");
        assert_eq!(local.ssh_port, 22); // Default
        assert_eq!(local.hive_port, 8600); // Default
        assert!(local.auto_start); // Default
        assert_eq!(local.workers.len(), 1);
    }

    #[test]
    fn test_validate_duplicate_aliases() {
        let config = HivesConfig {
            hives: vec![
                HiveConfig {
                    alias: "duplicate".to_string(),
                    hostname: "host1".to_string(),
                    ssh_user: "user1".to_string(),
                    ssh_port: 22,
                    hive_port: 8600,
                    binary_path: None,
                    workers: vec![],
                    auto_start: true,
                },
                HiveConfig {
                    alias: "duplicate".to_string(),
                    hostname: "host2".to_string(),
                    ssh_user: "user2".to_string(),
                    ssh_port: 22,
                    hive_port: 8600,
                    binary_path: None,
                    workers: vec![],
                    auto_start: true,
                },
            ],
        };

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfigError::DuplicateAlias { alias } => assert_eq!(alias, "duplicate"),
            _ => panic!("Expected DuplicateAlias error"),
        }
    }

    #[test]
    fn test_validate_empty_fields() {
        let config = HivesConfig {
            hives: vec![HiveConfig {
                alias: "".to_string(),
                hostname: "host".to_string(),
                ssh_user: "user".to_string(),
                ssh_port: 22,
                hive_port: 8600,
                binary_path: None,
                workers: vec![],
                auto_start: true,
            }],
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_load_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        let toml = r#"
[[hive]]
alias = "test"
hostname = "localhost"
ssh_user = "testuser"

workers = [
    { type = "vllm", version = "latest" },
]
"#;
        file.write_all(toml.as_bytes()).unwrap();
        file.flush().unwrap();

        let config = HivesConfig::load_from(file.path()).unwrap();
        assert_eq!(config.hives.len(), 1);
        assert_eq!(config.hives[0].alias, "test");
        assert_eq!(config.hives[0].workers.len(), 1);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let path = Path::new("/nonexistent/hives.conf");
        let config = HivesConfig::load_from(path).unwrap();
        assert!(config.is_empty());
    }

    #[test]
    fn test_save_and_reload() {
        let config = HivesConfig {
            hives: vec![HiveConfig {
                alias: "test".to_string(),
                hostname: "localhost".to_string(),
                ssh_user: "vince".to_string(),
                ssh_port: 22,
                hive_port: 8600,
                binary_path: None,
                workers: vec![WorkerConfig {
                    worker_type: "vllm".to_string(),
                    version: "latest".to_string(),
                    binary_path: None,
                }],
                auto_start: true,
            }],
        };

        let mut file = NamedTempFile::new().unwrap();
        config.save_to(file.path()).unwrap();

        let reloaded = HivesConfig::load_from(file.path()).unwrap();
        assert_eq!(reloaded, config);
    }

    #[test]
    fn test_get_hive() {
        let config = HivesConfig {
            hives: vec![
                HiveConfig {
                    alias: "hive1".to_string(),
                    hostname: "host1".to_string(),
                    ssh_user: "user1".to_string(),
                    ssh_port: 22,
                    hive_port: 8600,
                    binary_path: None,
                    workers: vec![],
                    auto_start: true,
                },
                HiveConfig {
                    alias: "hive2".to_string(),
                    hostname: "host2".to_string(),
                    ssh_user: "user2".to_string(),
                    ssh_port: 22,
                    hive_port: 8600,
                    binary_path: None,
                    workers: vec![],
                    auto_start: true,
                },
            ],
        };

        assert!(config.get_hive("hive1").is_some());
        assert!(config.get_hive("hive2").is_some());
        assert!(config.get_hive("hive3").is_none());
    }

    #[test]
    fn test_base_url() {
        let hive = HiveConfig {
            alias: "test".to_string(),
            hostname: "192.168.1.100".to_string(),
            ssh_user: "vince".to_string(),
            ssh_port: 22,
            hive_port: 8600,
            binary_path: None,
            workers: vec![],
            auto_start: true,
        };

        assert_eq!(hive.base_url(), "http://192.168.1.100:8600");
    }
}
