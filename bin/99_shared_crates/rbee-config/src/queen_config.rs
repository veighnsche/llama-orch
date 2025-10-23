//! Queen-level configuration (config.toml)
//!
//! Created by: TEAM-193

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Queen-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct QueenConfig {
    /// Queen daemon settings
    #[serde(default)]
    pub queen: QueenSettings,

    /// Runtime settings
    #[serde(default)]
    pub runtime: RuntimeSettings,
}

/// Queen daemon settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueenSettings {
    /// Port for queen-rbee HTTP API
    #[serde(default = "default_queen_port")]
    pub port: u16,

    /// Log level (trace, debug, info, warn, error)
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

impl Default for QueenSettings {
    fn default() -> Self {
        Self { port: default_queen_port(), log_level: default_log_level() }
    }
}

/// Runtime settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSettings {
    /// Maximum concurrent operations
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_operations: usize,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self { max_concurrent_operations: default_max_concurrent() }
    }
}


impl QueenConfig {
    /// Validate queen configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the port is 0
    pub fn validate(&self) -> Result<()> {
        // TEAM-195: u16 max is 65535, so only check for 0
        if self.queen.port == 0 {
            return Err(ConfigError::InvalidConfig(format!(
                "Invalid queen port: {} (must be 1-65535)",
                self.queen.port
            )));
        }
        Ok(())
    }

    /// Load from config.toml file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            // Return default config if file doesn't exist
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::ReadError { path: path.to_path_buf(), source: e })?;

        toml::from_str(&content)
            .map_err(|e| ConfigError::TomlParseError { path: path.to_path_buf(), source: e })
    }

    /// Save to config.toml file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or serialization fails
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            ConfigError::InvalidConfig(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(path, content)
            .map_err(|e| ConfigError::WriteError { path: path.to_path_buf(), source: e })
    }
}

// Default value functions
const fn default_queen_port() -> u16 {
    8080
}

fn default_log_level() -> String {
    "info".to_string()
}

const fn default_max_concurrent() -> usize {
    10
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = QueenConfig::default();
        assert_eq!(config.queen.port, 8080);
        assert_eq!(config.queen.log_level, "info");
        assert_eq!(config.runtime.max_concurrent_operations, 10);
    }

    #[test]
    fn test_load_nonexistent_returns_default() {
        let path = Path::new("/nonexistent/config.toml");
        let config = QueenConfig::load(path).unwrap();
        assert_eq!(config.queen.port, 8080);
    }

    #[test]
    fn test_load_and_save() {
        let mut file = NamedTempFile::new().unwrap();
        let content = r#"
[queen]
port = 9090
log_level = "debug"

[runtime]
max_concurrent_operations = 20
"#;
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();

        let config = QueenConfig::load(file.path()).unwrap();
        assert_eq!(config.queen.port, 9090);
        assert_eq!(config.queen.log_level, "debug");
        assert_eq!(config.runtime.max_concurrent_operations, 20);

        // Test save
        let save_file = NamedTempFile::new().unwrap();
        config.save(save_file.path()).unwrap();

        let loaded = QueenConfig::load(save_file.path()).unwrap();
        assert_eq!(loaded.queen.port, 9090);
    }

    #[test]
    fn test_partial_config() {
        let mut file = NamedTempFile::new().unwrap();
        let content = r#"
[queen]
port = 7070
"#;
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();

        let config = QueenConfig::load(file.path()).unwrap();
        assert_eq!(config.queen.port, 7070);
        assert_eq!(config.queen.log_level, "info"); // default
        assert_eq!(config.runtime.max_concurrent_operations, 10); // default
    }

    // TEAM-195: Validation tests
    #[test]
    fn test_validate_valid_config() {
        let config = QueenConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_port_zero() {
        let mut config = QueenConfig::default();
        config.queen.port = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_valid_port_max() {
        let mut config = QueenConfig::default();
        config.queen.port = 65535; // Max valid u16
        assert!(config.validate().is_ok());
    }
}
