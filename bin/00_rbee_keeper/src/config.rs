//! Configuration management for rbee-keeper
//!
//! Loads config from ~/.config/rbee/config.toml
//!
//! TEAM-216: Investigated - Complete behavior inventory created

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_queen_port")]
    pub queen_port: u16,
    
    // TEAM-312: DELETED backwards compatibility fields (pool, paths, remote)
    // These were never used and created permanent technical debt.
    // Pre-1.0 software is ALLOWED to break. See: RULE ZERO
}

fn default_queen_port() -> u16 {
    7833
}

impl Default for Config {
    fn default() -> Self {
        Self {
            queen_port: default_queen_port(),
        }
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

        let config: Config = toml::from_str(&contents).context("Failed to parse config file")?;

        Ok(config)
    }

    /// Save config to ~/.config/rbee/config.toml
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        let contents = toml::to_string_pretty(self).context("Failed to serialize config")?;

        fs::write(&config_path, contents).context("Failed to write config file")?;

        Ok(())
    }

    /// Get the path to the config file: ~/.config/rbee/config.toml
    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().context("Failed to get config directory")?;

        Ok(config_dir.join("rbee").join("config.toml"))
    }

    /// Get queen URL based on configured port
    pub fn queen_url(&self) -> String {
        format!("http://localhost:{}", self.queen_port)
    }
}
