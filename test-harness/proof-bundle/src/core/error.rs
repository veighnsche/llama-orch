//! Error types for proof bundle operations

use std::fmt;

/// Errors that can occur during proof bundle generation
#[derive(Debug)]
pub enum ProofBundleError {
    /// No tests were found in the output
    NoTestsFound {
        /// Package name that was tested
        package: String,
        /// Hint for the user
        hint: String,
    },
    
    /// Failed to run cargo test
    CargoTestFailed {
        /// Exit code from cargo
        exit_code: Option<i32>,
        /// Error message
        message: String,
    },
    
    /// Failed to parse test output
    ParseError {
        /// What we were trying to parse
        context: String,
        /// The error that occurred
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    /// Failed to extract metadata from source
    MetadataExtractionFailed {
        /// File that failed
        file: String,
        /// Error message
        message: String,
    },
    
    /// Cannot generate reports with invalid data
    CannotGenerateReports {
        /// Reason why
        reason: String,
    },
    
    /// I/O error
    Io {
        /// What operation failed
        operation: String,
        /// The I/O error
        source: std::io::Error,
    },
    
    /// Generic error
    Other(String),
}

impl fmt::Display for ProofBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTestsFound { package, hint } => {
                write!(f, "No tests found for package '{}'. {}", package, hint)
            }
            Self::CargoTestFailed { exit_code, message } => {
                write!(f, "cargo test failed")?;
                if let Some(code) = exit_code {
                    write!(f, " with exit code {}", code)?;
                }
                write!(f, ": {}", message)
            }
            Self::ParseError { context, source } => {
                write!(f, "Failed to parse {}: {}", context, source)
            }
            Self::MetadataExtractionFailed { file, message } => {
                write!(f, "Failed to extract metadata from {}: {}", file, message)
            }
            Self::CannotGenerateReports { reason } => {
                write!(f, "Cannot generate reports: {}", reason)
            }
            Self::Io { operation, source } => {
                write!(f, "I/O error during {}: {}", operation, source)
            }
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ProofBundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ParseError { source, .. } => Some(source.as_ref()),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ProofBundleError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            operation: "unknown".to_string(),
            source: err,
        }
    }
}

impl From<serde_json::Error> for ProofBundleError {
    fn from(err: serde_json::Error) -> Self {
        Self::ParseError {
            context: "JSON".to_string(),
            source: Box::new(err),
        }
    }
}
