//! Server-Sent Events (SSE) streaming for inference results
//!
//! This module provides SSE event types and streaming logic for real-time
//! token delivery during inference.
//!
//! # Spec References
//! - M0-W-1310: SSE event types
//! - M0-W-1311: Event ordering
//! - M0-W-1312: UTF-8 safety

use serde::{Deserialize, Serialize};

/// Reason for inference termination
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Reached max_tokens limit
    MaxTokens,
    /// Matched a stop sequence
    StopSequence,
    /// Error occurred
    Error,
    /// Request cancelled
    Cancelled,
}

/// Inference SSE event types
///
/// Events are emitted in strict order:
/// 1. `Started` - Inference began
/// 2. `Token` (0 or more) - Generated tokens
/// 3. `Metrics` (optional) - Performance metrics
/// 4. Terminal event - Either `End` OR `Error` (never both)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    /// Inference started
    Started {
        /// Job ID
        job_id: String,
        /// Model name/path
        model: String,
        /// ISO 8601 timestamp
        started_at: String,
    },

    /// Token generated
    Token {
        /// Token text (UTF-8 safe)
        t: String,
        /// Token index (0-based)
        i: u32,
    },

    /// Performance metrics (optional, emitted periodically)
    Metrics {
        /// Tokens per second
        tokens_per_sec: f32,
        /// VRAM usage in bytes
        vram_bytes: u64,
    },

    /// Inference completed successfully
    End {
        /// Total tokens generated
        tokens_out: u32,
        /// Decode time in milliseconds
        decode_time_ms: u64,
        /// Reason for stopping
        stop_reason: StopReason,
        /// Stop sequence that was matched (if stop_reason = StopSequence)
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_sequence_matched: Option<String>,
    },

    /// Inference failed
    Error {
        /// Error code
        code: String,
        /// Human-readable error message
        message: String,
    },
}

/// Error codes for SSE error events
pub mod error_codes {
    /// VRAM out of memory
    pub const VRAM_OOM: &str = "VRAM_OOM";

    /// Job cancelled by client
    pub const CANCELLED: &str = "CANCELLED";

    /// Job timed out
    pub const TIMEOUT: &str = "TIMEOUT";

    /// Invalid request parameters
    pub const INVALID_REQUEST: &str = "INVALID_REQUEST";

    /// Inference failed (CUDA error, model error, etc.)
    pub const INFERENCE_FAILED: &str = "INFERENCE_FAILED";
}

impl InferenceEvent {
    /// Check if this is a terminal event (End or Error)
    pub fn is_terminal(&self) -> bool {
        matches!(self, InferenceEvent::End { .. } | InferenceEvent::Error { .. })
    }

    /// Get event type name for SSE event field
    pub fn event_name(&self) -> &'static str {
        match self {
            InferenceEvent::Started { .. } => "started",
            InferenceEvent::Token { .. } => "token",
            InferenceEvent::Metrics { .. } => "metrics",
            InferenceEvent::End { .. } => "end",
            InferenceEvent::Error { .. } => "error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_started_event_serialization() {
        let event = InferenceEvent::Started {
            job_id: "test-123".to_string(),
            model: "Qwen2.5-0.5B".to_string(),
            started_at: "2025-10-04T12:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"started\""));
        assert!(json.contains("test-123"));
        assert!(json.contains("Qwen2.5-0.5B"));
    }

    #[test]
    fn test_token_event_serialization() {
        let event = InferenceEvent::Token { t: "Hello".to_string(), i: 5 };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"token\""));
        assert!(json.contains("\"t\":\"Hello\""));
        assert!(json.contains("\"i\":5"));
    }

    #[test]
    fn test_metrics_event_serialization() {
        let event = InferenceEvent::Metrics {
            tokens_per_sec: 42.5,
            vram_bytes: 1024 * 1024 * 512, // 512 MB
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"metrics\""));
        assert!(json.contains("tokens_per_sec"));
        assert!(json.contains("vram_bytes"));
    }

    #[test]
    fn test_end_event_serialization() {
        let event = InferenceEvent::End { 
            tokens_out: 100, 
            decode_time_ms: 5000,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"end\""));
        assert!(json.contains("\"tokens_out\":100"));
        assert!(json.contains("\"decode_time_ms\":5000"));
        assert!(json.contains("max_tokens"));
    }

    #[test]
    fn test_error_event_serialization() {
        let event = InferenceEvent::Error {
            code: error_codes::VRAM_OOM.to_string(),
            message: "Out of VRAM".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("VRAM_OOM"));
        assert!(json.contains("Out of VRAM"));
    }

    #[test]
    fn test_is_terminal() {
        let started = InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-04T12:00:00Z".to_string(),
        };
        assert!(!started.is_terminal());

        let token = InferenceEvent::Token { t: "test".to_string(), i: 0 };
        assert!(!token.is_terminal());

        let metrics = InferenceEvent::Metrics { tokens_per_sec: 10.0, vram_bytes: 1024 };
        assert!(!metrics.is_terminal());

        let end = InferenceEvent::End { 
            tokens_out: 10, 
            decode_time_ms: 1000,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        };
        assert!(end.is_terminal());

        let error = InferenceEvent::Error { code: "TEST".to_string(), message: "test".to_string() };
        assert!(error.is_terminal());
    }

    #[test]
    fn test_event_names() {
        let started = InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-04T12:00:00Z".to_string(),
        };
        assert_eq!(started.event_name(), "started");

        let token = InferenceEvent::Token { t: "test".to_string(), i: 0 };
        assert_eq!(token.event_name(), "token");

        let end = InferenceEvent::End { 
            tokens_out: 10, 
            decode_time_ms: 1000,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        };
        assert_eq!(end.event_name(), "end");
    }

    #[test]
    fn test_token_with_emoji() {
        let event = InferenceEvent::Token { t: "👋🌍".to_string(), i: 0 };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("👋🌍"));
    }

    #[test]
    fn test_token_with_cjk() {
        let event = InferenceEvent::Token { t: "世界".to_string(), i: 0 };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("世界"));
    }

    #[test]
    fn test_stop_reason_serialization() {
        let max_tokens = StopReason::MaxTokens;
        let json = serde_json::to_string(&max_tokens).unwrap();
        assert_eq!(json, "\"max_tokens\"");

        let stop_seq = StopReason::StopSequence;
        let json = serde_json::to_string(&stop_seq).unwrap();
        assert_eq!(json, "\"stop_sequence\"");

        let error = StopReason::Error;
        let json = serde_json::to_string(&error).unwrap();
        assert_eq!(json, "\"error\"");

        let cancelled = StopReason::Cancelled;
        let json = serde_json::to_string(&cancelled).unwrap();
        assert_eq!(json, "\"cancelled\"");
    }

    #[test]
    fn test_end_event_with_stop_sequence() {
        let event = InferenceEvent::End {
            tokens_out: 25,
            decode_time_ms: 750,
            stop_reason: StopReason::StopSequence,
            stop_sequence_matched: Some("\n\n".to_string()),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"end\""));
        assert!(json.contains("\"stop_reason\":\"stop_sequence\""));
        assert!(json.contains("\"stop_sequence_matched\":\"\\n\\n\""));
    }

    #[test]
    fn test_end_event_omits_none_stop_sequence() {
        let event = InferenceEvent::End {
            tokens_out: 100,
            decode_time_ms: 2000,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"stop_reason\":\"max_tokens\""));
        assert!(!json.contains("stop_sequence_matched"));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
