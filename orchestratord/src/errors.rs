//! Typed error envelopes and helpers (planning-only).

use serde::{Deserialize, Serialize};

/// Error codes enumerated for API error envelopes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    AdmissionReject,
    QueueFullDropLru,
    Internal,
    Unavailable,
    BadRequest,
}

/// Error envelope returned by handlers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub code: ErrorCode,
    pub message: String,
    pub engine: Option<String>,
    /// Optional advisory label surfaced on 429 bodies.
    pub policy_label: Option<String>,
}

impl ErrorEnvelope {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            engine: None,
            policy_label: None,
        }
    }
}

/// Map admission errors to envelopes.
/// Planning-only: returns a placeholder Internal error.
pub fn map_admission_error(_err: &str) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCode::Internal, "not implemented")
}
