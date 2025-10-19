//! Configuration file loading
//!
//! Created by: TEAM-036
//! Implements XDG Base Directory specification

use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub pool: PoolConfig,
    pub paths: PathsConfig,
    #[serde(default)]
    pub remote: Option<RemoteConfig>,
}

#[derive(Debug, Deserialize)]
pub struct PoolConfig {
    pub name: String,
    pub listen_addr: String,
}

#[derive(Debug, Deserialize)]
pub struct PathsConfig {
    pub models_dir: PathBuf,
    pub catalog_db: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct RemoteConfig {
    /// Custom binary path on remote machines
    pub binary_path: Option<String>,
    /// Git repository directory on remote machines
    pub git_repo_dir: Option<PathBuf>,
}

impl Config {
    /// Load config from standard locations
    /// Priority: RBEE_CONFIG env var > ~/.config/rbee/config.toml > /etc/rbee/config.toml
    pub fn load() -> Result<Self> {
        let config_path = if let Ok(path) = std::env::var("RBEE_CONFIG") {
            PathBuf::from(path)
        } else if let Some(home) = dirs::home_dir() {
            let user_config = home.join(".config/rbee/config.toml");
            if user_config.exists() {
                user_config
            } else {
                PathBuf::from("/etc/rbee/config.toml")
            }
        } else {
            PathBuf::from("/etc/rbee/config.toml")
        };

        let contents = std::fs::read_to_string(&config_path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
}
