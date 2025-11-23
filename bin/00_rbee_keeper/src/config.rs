//! Configuration management for rbee-keeper
//!
//! Loads config from ~/.config/rbee/config.toml
//!
//! TEAM-216: Investigated - Complete behavior inventory created
//! TEAM-315: Use KeeperConfig from keeper-config-contract

use anyhow::{Context, Result};
use std::fs;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

// TEAM-315: Use KeeperConfig from contract
pub use keeper_config_contract::KeeperConfig;

// TEAM-315: Wrapper to add I/O operations
#[derive(Debug, Clone)]
pub struct Config(KeeperConfig);

impl Deref for Config {
    type Target = KeeperConfig;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Config {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for Config {
    fn default() -> Self {
        Self(KeeperConfig::default())
    }
}

impl Config {
    /// Load config from ~/.config/rbee/config.toml
    /// Creates default config if file doesn't exist
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            // Create default config
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }

        let contents = fs::read_to_string(&config_path).context("Failed to read config file")?;

        let keeper_config =
            KeeperConfig::from_toml(&contents).context("Failed to parse config file")?;

        // Validate
        keeper_config.validate().context("Invalid configuration")?;

        Ok(Self(keeper_config))
    }

    /// Save config to ~/.config/rbee/config.toml
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        let contents = self.0.to_toml().context("Failed to serialize config")?;

        fs::write(&config_path, contents).context("Failed to write config file")?;

        Ok(())
    }

    /// Get the path to the config file: ~/.config/rbee/config.toml
    pub fn config_path() -> Result<PathBuf> {
        // For testing, respect HOME environment variable if set
        if let Ok(home) = std::env::var("HOME") {
            let home_path = PathBuf::from(home);

            // For test isolation, include thread ID in path when running tests
            let thread_id = std::thread::current().id();
            let thread_suffix = format!("{:?}", thread_id).replace(['(', ')', ' ', '-'], "_");

            let config_path = if std::env::var("RUST_TEST_THREADS").is_ok() || cfg!(test) {
                // Use thread-specific config directory for tests
                home_path
                    .join(".config")
                    .join("rbee")
                    .join(format!("test_{}", thread_suffix))
                    .join("config.toml")
            } else {
                // Normal config path for production
                home_path.join(".config").join("rbee").join("config.toml")
            };

            return Ok(config_path);
        }

        let config_dir = dirs::config_dir().context("Failed to get config directory")?;
        Ok(config_dir.join("rbee").join("config.toml"))
    }
}
