//! Inference result tracking with stop reason
//!
//! This module provides a unified result type that tracks why inference
//! terminated, enabling proper response construction and debugging.
//!
//! # Spec References
//! - M0-W-1422: Stop sequences
//! - M0-W-1300: HTTP API extension

use serde::{Deserialize, Serialize};

/// Stop reason for inference termination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StopReason {
    /// Max tokens reached
    MaxTokens,
    /// EOS token generated
    Eos,
    /// Stop sequence matched
    StopSequence,
    /// Inference cancelled
    Cancelled,
    /// Error occurred
    Error,
}

/// Complete inference result with termination reason
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Generated tokens (UTF-8 strings)
    pub tokens: Vec<String>,

    /// Generated token IDs
    pub token_ids: Vec<u32>,

    /// Why inference terminated
    pub stop_reason: StopReason,

    /// Which stop sequence matched (if stop_reason = StopSequence)
    pub stop_sequence_matched: Option<String>,

    /// Actual seed used (generated if not provided)
    pub seed: u64,

    /// Total decode time in milliseconds
    pub decode_time_ms: u64,
}

impl InferenceResult {
    /// Create result for max_tokens termination
    pub fn max_tokens(
        tokens: Vec<String>,
        token_ids: Vec<u32>,
        seed: u64,
        decode_time_ms: u64,
    ) -> Self {
        Self {
            tokens,
            token_ids,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
            seed,
            decode_time_ms,
        }
    }

    /// Create result for stop sequence termination
    pub fn stop_sequence(
        tokens: Vec<String>,
        token_ids: Vec<u32>,
        seed: u64,
        decode_time_ms: u64,
        matched_sequence: String,
    ) -> Self {
        Self {
            tokens,
            token_ids,
            stop_reason: StopReason::StopSequence,
            stop_sequence_matched: Some(matched_sequence),
            seed,
            decode_time_ms,
        }
    }

    /// Create result for cancellation
    pub fn cancelled(
        tokens: Vec<String>,
        token_ids: Vec<u32>,
        seed: u64,
        decode_time_ms: u64,
    ) -> Self {
        Self {
            tokens,
            token_ids,
            stop_reason: StopReason::Cancelled,
            stop_sequence_matched: None,
            seed,
            decode_time_ms,
        }
    }

    /// Create result for error
    pub fn error(tokens: Vec<String>, token_ids: Vec<u32>, seed: u64, decode_time_ms: u64) -> Self {
        Self {
            tokens,
            token_ids,
            stop_reason: StopReason::Error,
            stop_sequence_matched: None,
            seed,
            decode_time_ms,
        }
    }

    /// Get total tokens generated
    pub fn token_count(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Check if inference completed successfully (not error/cancelled)
    pub fn is_success(&self) -> bool {
        matches!(self.stop_reason, StopReason::MaxTokens | StopReason::Eos | StopReason::StopSequence)
    }

    /// Get human-readable stop reason description
    pub fn stop_reason_description(&self) -> String {
        match &self.stop_reason {
            StopReason::MaxTokens => "Reached max_tokens limit".to_string(),
            StopReason::Eos => "End of sequence token generated".to_string(),
            StopReason::StopSequence => {
                if let Some(seq) = &self.stop_sequence_matched {
                    format!("Matched stop sequence: {:?}", seq)
                } else {
                    "Matched stop sequence".to_string()
                }
            }
            StopReason::Error => "Inference error occurred".to_string(),
            StopReason::Cancelled => "Request cancelled by client".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_tokens_result() {
        let result = InferenceResult::max_tokens(
            vec!["Hello".to_string(), " world".to_string()],
            vec![100, 200],
            42,
            1000,
        );

        assert_eq!(result.token_count(), 2);
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
        assert!(result.stop_sequence_matched.is_none());
        assert!(result.is_success());
        assert_eq!(result.seed, 42);
    }

    #[test]
    fn test_stop_sequence_result() {
        let result = InferenceResult::stop_sequence(
            vec!["Line 1".to_string(), "\n\n".to_string()],
            vec![100, 200],
            42,
            500,
            "\n\n".to_string(),
        );

        assert_eq!(result.token_count(), 2);
        assert_eq!(result.stop_reason, StopReason::StopSequence);
        assert_eq!(result.stop_sequence_matched, Some("\n\n".to_string()));
        assert!(result.is_success());
    }

    #[test]
    fn test_cancelled_result() {
        let result = InferenceResult::cancelled(vec!["Partial".to_string()], vec![100], 42, 250);

        assert_eq!(result.token_count(), 1);
        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert!(!result.is_success());
    }

    #[test]
    fn test_error_result() {
        let result = InferenceResult::error(vec![], vec![], 42, 0);

        assert_eq!(result.token_count(), 0);
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(!result.is_success());
    }

    #[test]
    fn test_stop_reason_descriptions() {
        let max_tokens = InferenceResult::max_tokens(vec![], vec![], 42, 0);
        assert!(max_tokens.stop_reason_description().contains("max_tokens"));

        let stop_seq = InferenceResult::stop_sequence(vec![], vec![], 42, 0, "\n\n".to_string());
        assert!(stop_seq.stop_reason_description().contains("stop sequence"));
        assert!(stop_seq.stop_reason_description().contains("\\n\\n"));

        let cancelled = InferenceResult::cancelled(vec![], vec![], 42, 0);
        assert!(cancelled.stop_reason_description().contains("cancelled"));

        let error = InferenceResult::error(vec![], vec![], 42, 0);
        assert!(error.stop_reason_description().contains("error"));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
