//! streaming — SSE token streaming
//!
//! Streams inference tokens to clients via Server-Sent Events (SSE).
//!
//! # Key Responsibilities
//!
//! - SSE connection management
//! - Token streaming (event: token)
//! - Progress updates (event: progress)
//! - Error propagation (event: error)
//! - Connection cleanup on client disconnect
//!
//! # Example
//!
//! ```rust
//! use streaming::{StreamManager, StreamEvent};
//!
//! let manager = StreamManager::new();
//! let stream_id = manager.create_stream(job_id);
//!
//! // Send token
//! manager.send_event(stream_id, StreamEvent::Token {
//!     text: "Hello".to_string(),
//!     index: 0,
//! });
//! ```

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StreamError {
    #[error("stream not found: {0}")]
    NotFound(String),
    #[error("stream closed")]
    Closed,
}

pub type Result<T> = std::result::Result<T, StreamError>;

/// Stream event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum StreamEvent {
    Token { text: String, index: u32 },
    Progress { message: String, percent: f32 },
    Error { message: String },
    End { tokens_out: u32, decode_ms: u64 },
}

/// Stream manager
pub struct StreamManager {
    // Future: Track active streams
}

impl StreamManager {
    pub fn new() -> Self {
        Self {}
    }

    pub fn create_stream(&self, _job_id: &str) -> String {
        // Generate stream ID
        "stream-123".to_string()
    }

    pub fn send_event(&self, _stream_id: &str, event: StreamEvent) -> Result<()> {
        tracing::debug!(event = ?event, "Stream event");
        Ok(())
    }
}

impl Default for StreamManager {
    fn default() -> Self {
        Self::new()
    }
}
