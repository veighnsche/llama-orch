//! Error types for validation failures

use thiserror::Error;

/// Validation error types
///
/// All errors contain only metadata (lengths, types), never input content.
/// This prevents sensitive data leakage in logs.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum ValidationError {
    /// String is empty
    #[error("string is empty")]
    Empty,

    /// String exceeds maximum length
    #[error("string too long: {actual} chars (max {max})")]
    TooLong { actual: usize, max: usize },

    /// String contains invalid characters
    #[error("invalid characters found: {found}")]
    InvalidCharacters { found: String },

    /// String contains null byte
    #[error("null byte found in string")]
    NullByte,

    /// Path traversal sequence detected
    #[error("path traversal sequence detected")]
    PathTraversal,

    /// Wrong length (for hex strings, etc.)
    #[error("wrong length: {actual} (expected {expected})")]
    WrongLength { actual: usize, expected: usize },

    /// Invalid hex character
    #[error("invalid hex character: {char}")]
    InvalidHex { char: char },

    /// Value out of range
    #[error("value out of range: {value} (expected {min}..{max})")]
    OutOfRange { value: String, min: String, max: String },

    /// Control character found
    #[error("control character found: {char:?}")]
    ControlCharacter { char: char },

    /// ANSI escape sequence detected
    #[error("ANSI escape sequence detected")]
    AnsiEscape,

    /// Shell metacharacter found
    #[error("shell metacharacter found: {char}")]
    ShellMetacharacter { char: char },

    /// Path outside allowed root
    #[error("path outside allowed directory")]
    PathOutsideRoot,

    /// I/O error during validation
    #[error("I/O error: {0}")]
    Io(String),
}

/// Result type for validation operations
pub type Result<T> = std::result::Result<T, ValidationError>;
