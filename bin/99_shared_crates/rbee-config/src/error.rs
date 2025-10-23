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
        source: std::io::Error,
    },

    /// Failed to write config file
    #[error("Failed to write config file {path}: {source}")]
    WriteError {
        /// Path to the config file
        path: PathBuf,
        /// IO error source
        source: std::io::Error,
    },

    /// Failed to parse TOML config
    #[error("Failed to parse TOML config {path}: {source}")]
    TomlParseError {
        /// Path to the config file
        path: PathBuf,
        /// TOML deserialization error
        source: toml::de::Error
    },

    /// Failed to parse YAML config
    #[error("Failed to parse YAML config {path}: {source}")]
    YamlParseError {
        /// Path to the config file
        path: PathBuf,
        /// YAML deserialization error
        source: serde_yaml::Error
    },

    /// Failed to serialize YAML
    #[error("Failed to serialize YAML: {0}")]
    YamlSerializeError(#[from] serde_yaml::Error),

    /// Invalid hives.conf syntax
    #[error("Invalid hives.conf syntax at line {line}: {message}")]
    InvalidSyntax {
        /// Line number where the error occurred
        line: usize,
        /// Error message
        message: String
    },

    /// Missing required field in config
    #[error("Missing required field '{field}' for host '{host}'")]
    MissingField {
        /// Host name
        host: String,
        /// Missing field name
        field: String
    },

    /// Invalid port number
    #[error("Invalid port number '{value}' for host '{host}'")]
    InvalidPort {
        /// Host name
        host: String,
        /// Invalid port value
        value: String
    },

    /// Duplicate hive alias
    #[error("Duplicate hive alias: '{alias}'")]
    DuplicateAlias {
        /// Duplicate alias name
        alias: String
    },

    /// Hive not found
    #[error("Hive not found: '{alias}'")]
    HiveNotFound {
        /// Hive alias that was not found
        alias: String
    },

    /// Invalid configuration
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias for config operations
pub type Result<T> = std::result::Result<T, ConfigError>;
