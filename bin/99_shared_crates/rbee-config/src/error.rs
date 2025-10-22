//! Error types for rbee-config
//!
//! Created by: TEAM-193

use std::path::PathBuf;

/// Config error type
/// TEAM-209: Added missing variant documentation
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Config directory not found
    #[error("Config directory not found: {0}")]
    DirectoryNotFound(PathBuf),

    /// Failed to read config file
    #[error("Failed to read config file {path}: {source}")]
    ReadError {
        /// Path to the config file
        path: PathBuf,
        /// IO error source
        source: std::io::Error
    },

    /// Failed to write config file
    #[error("Failed to write config file {path}: {source}")]
    WriteError {
        /// Path to the config file
        path: PathBuf,
        /// IO error source
        source: std::io::Error
    },

    #[error("Failed to parse TOML config {path}: {source}")]
    TomlParseError { path: PathBuf, source: toml::de::Error },

    #[error("Failed to parse YAML config {path}: {source}")]
    YamlParseError { path: PathBuf, source: serde_yaml::Error },

    #[error("Failed to serialize YAML: {0}")]
    YamlSerializeError(#[from] serde_yaml::Error),

    #[error("Invalid hives.conf syntax at line {line}: {message}")]
    InvalidSyntax { line: usize, message: String },

    #[error("Missing required field '{field}' for host '{host}'")]
    MissingField { host: String, field: String },

    #[error("Invalid port number '{value}' for host '{host}'")]
    InvalidPort { host: String, value: String },

    #[error("Duplicate hive alias: '{alias}'")]
    DuplicateAlias { alias: String },

    #[error("Hive not found: '{alias}'")]
    HiveNotFound { alias: String },

    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ConfigError>;
