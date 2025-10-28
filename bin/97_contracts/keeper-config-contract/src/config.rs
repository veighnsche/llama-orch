//! Keeper configuration types
//!
//! TEAM-315: Extracted from rbee-keeper for stability

use serde::{Deserialize, Serialize};

use crate::validation::ValidationError;

/// Keeper configuration
///
/// Loaded from ~/.config/rbee/config.toml
///
/// # Example
///
/// ```rust
/// use keeper_config_contract::KeeperConfig;
///
/// let config = KeeperConfig::default();
/// assert_eq!(config.queen_port, 7833);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeeperConfig {
    /// Queen HTTP port
    #[serde(default = "default_queen_port")]
    pub queen_port: u16,
}

fn default_queen_port() -> u16 {
    7833
}

impl Default for KeeperConfig {
    fn default() -> Self {
        Self { queen_port: default_queen_port() }
    }
}

impl KeeperConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Get queen URL based on configured port
    pub fn queen_url(&self) -> String {
        format!("http://localhost:{}", self.queen_port)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Port must be in valid range (>= 1024, u16 max is 65535 so no upper check needed)
        if self.queen_port < 1024 {
            return Err(ValidationError::InvalidPort {
                port: self.queen_port,
                reason: "Port must be >= 1024 (non-privileged)".to_string(),
            });
        }

        Ok(())
    }

    /// Parse from TOML string
    pub fn from_toml(toml_str: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_str)
    }

    /// Serialize to TOML string
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KeeperConfig::default();
        assert_eq!(config.queen_port, 7833);
    }

    #[test]
    fn test_queen_url() {
        let config = KeeperConfig { queen_port: 8080 };
        assert_eq!(config.queen_url(), "http://localhost:8080");
    }

    #[test]
    fn test_validation_valid() {
        let config = KeeperConfig { queen_port: 7833 };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_port_too_low() {
        let config = KeeperConfig { queen_port: 80 };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = KeeperConfig { queen_port: 7833 };

        let toml_str = config.to_toml().unwrap();
        let parsed = KeeperConfig::from_toml(&toml_str).unwrap();

        assert_eq!(config, parsed);
    }

    #[test]
    fn test_toml_with_defaults() {
        let toml_str = ""; // Empty config should use defaults
        let config = KeeperConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.queen_port, 7833);
    }
}
