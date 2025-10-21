//! Error types for rbee-config
//!
//! Created by: TEAM-193

use std::path::PathBuf;

/// Config error type
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Config directory not found: {0}")]
    DirectoryNotFound(PathBuf),

    #[error("Failed to read config file {path}: {source}")]
    ReadError {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Failed to write config file {path}: {source}")]
    WriteError {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Failed to parse TOML config {path}: {source}")]
    TomlParseError {
        path: PathBuf,
        source: toml::de::Error,
    },

    #[error("Failed to parse YAML config {path}: {source}")]
    YamlParseError {
        path: PathBuf,
        source: serde_yaml::Error,
    },

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
