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

    /// Which stop sequence matched (if `stop_reason` = `StopSequence`)
    pub stop_sequence_matched: Option<String>,

    /// Actual seed used (generated if not provided)
    pub seed: u64,

    /// Total decode time in milliseconds
    pub decode_time_ms: u64,
}

impl InferenceResult {
    /// Create result for `max_tokens` termination
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
        matches!(
            self.stop_reason,
            StopReason::MaxTokens | StopReason::Eos | StopReason::StopSequence
        )
    }

    /// Get human-readable stop reason description
    pub fn stop_reason_description(&self) -> String {
        match &self.stop_reason {
            StopReason::MaxTokens => "Reached max_tokens limit".to_string(),
            StopReason::Eos => "End of sequence token generated".to_string(),
            StopReason::StopSequence => {
                if let Some(seq) = &self.stop_sequence_matched {
                    format!("Matched stop sequence: {seq:?}")
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

    #[test]
    fn test_eos_result() {
        // EOS is not directly constructible but we can test via StopReason
        let mut result = InferenceResult::max_tokens(vec!["Hello".to_string()], vec![100], 42, 100);
        result.stop_reason = StopReason::Eos;

        assert_eq!(result.stop_reason, StopReason::Eos);
        assert!(result.is_success());
        assert!(result.stop_reason_description().contains("End of sequence"));
    }

    #[test]
    fn test_token_count_consistency() {
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let token_ids = vec![1, 2, 3];
        let result = InferenceResult::max_tokens(tokens.clone(), token_ids.clone(), 42, 100);

        assert_eq!(result.token_count(), 3);
        assert_eq!(result.tokens.len(), 3);
        assert_eq!(result.token_ids.len(), 3);
    }

    #[test]
    fn test_empty_result() {
        let result = InferenceResult::error(vec![], vec![], 42, 0);
        assert_eq!(result.token_count(), 0);
        assert!(result.tokens.is_empty());
        assert!(result.token_ids.is_empty());
    }

    #[test]
    fn test_large_token_count() {
        let tokens: Vec<String> = (0..1000).map(|i| format!("token{}", i)).collect();
        let token_ids: Vec<u32> = (0..1000).collect();
        let result = InferenceResult::max_tokens(tokens, token_ids, 42, 5000);

        assert_eq!(result.token_count(), 1000);
    }

    #[test]
    fn test_stop_sequence_without_match() {
        let result = InferenceResult::stop_sequence(
            vec!["test".to_string()],
            vec![100],
            42,
            100,
            "".to_string(),
        );

        assert_eq!(result.stop_reason, StopReason::StopSequence);
        assert_eq!(result.stop_sequence_matched, Some("".to_string()));
    }

    #[test]
    fn test_decode_time_tracking() {
        let result = InferenceResult::max_tokens(vec![], vec![], 42, 12345);
        assert_eq!(result.decode_time_ms, 12345);
    }

    #[test]
    fn test_seed_tracking() {
        let result = InferenceResult::max_tokens(vec![], vec![], 999, 100);
        assert_eq!(result.seed, 999);
    }

    #[test]
    fn test_stop_reason_serialization() {
        // Test that StopReason serializes to SCREAMING_SNAKE_CASE
        let json = serde_json::to_string(&StopReason::MaxTokens).unwrap();
        assert_eq!(json, "\"MAX_TOKENS\"");

        let json = serde_json::to_string(&StopReason::Eos).unwrap();
        assert_eq!(json, "\"EOS\"");

        let json = serde_json::to_string(&StopReason::StopSequence).unwrap();
        assert_eq!(json, "\"STOP_SEQUENCE\"");

        let json = serde_json::to_string(&StopReason::Cancelled).unwrap();
        assert_eq!(json, "\"CANCELLED\"");

        let json = serde_json::to_string(&StopReason::Error).unwrap();
        assert_eq!(json, "\"ERROR\"");
    }

    #[test]
    fn test_stop_reason_deserialization() {
        let reason: StopReason = serde_json::from_str("\"MAX_TOKENS\"").unwrap();
        assert_eq!(reason, StopReason::MaxTokens);

        let reason: StopReason = serde_json::from_str("\"EOS\"").unwrap();
        assert_eq!(reason, StopReason::Eos);

        let reason: StopReason = serde_json::from_str("\"STOP_SEQUENCE\"").unwrap();
        assert_eq!(reason, StopReason::StopSequence);

        let reason: StopReason = serde_json::from_str("\"CANCELLED\"").unwrap();
        assert_eq!(reason, StopReason::Cancelled);

        let reason: StopReason = serde_json::from_str("\"ERROR\"").unwrap();
        assert_eq!(reason, StopReason::Error);
    }

    #[test]
    fn test_is_success_classification() {
        // Success cases
        let max_tokens = InferenceResult::max_tokens(vec![], vec![], 42, 0);
        assert!(max_tokens.is_success());

        let mut eos = InferenceResult::max_tokens(vec![], vec![], 42, 0);
        eos.stop_reason = StopReason::Eos;
        assert!(eos.is_success());

        let stop_seq = InferenceResult::stop_sequence(vec![], vec![], 42, 0, "END".to_string());
        assert!(stop_seq.is_success());

        // Failure cases
        let cancelled = InferenceResult::cancelled(vec![], vec![], 42, 0);
        assert!(!cancelled.is_success());

        let error = InferenceResult::error(vec![], vec![], 42, 0);
        assert!(!error.is_success());
    }

    #[test]
    fn test_stop_sequence_description_with_match() {
        let result = InferenceResult::stop_sequence(vec![], vec![], 42, 0, "###".to_string());
        let desc = result.stop_reason_description();
        assert!(desc.contains("stop sequence"));
        assert!(desc.contains("###"));
    }

    #[test]
    fn test_stop_sequence_description_without_match() {
        let mut result = InferenceResult::max_tokens(vec![], vec![], 42, 0);
        result.stop_reason = StopReason::StopSequence;
        result.stop_sequence_matched = None;

        let desc = result.stop_reason_description();
        assert!(desc.contains("stop sequence"));
        assert!(!desc.contains(":"));
    }

    #[test]
    fn test_unicode_tokens() {
        let tokens = vec!["Hello".to_string(), " 世界".to_string(), "🌍".to_string()];
        let token_ids = vec![100, 200, 300];
        let result = InferenceResult::max_tokens(tokens.clone(), token_ids, 42, 100);

        assert_eq!(result.token_count(), 3);
        assert_eq!(result.tokens[1], " 世界");
        assert_eq!(result.tokens[2], "🌍");
    }

    #[test]
    fn test_partial_generation_on_error() {
        let partial_tokens = vec!["Hello".to_string(), " wor".to_string()];
        let partial_ids = vec![100, 200];
        let result = InferenceResult::error(partial_tokens.clone(), partial_ids.clone(), 42, 50);

        assert_eq!(result.token_count(), 2);
        assert_eq!(result.tokens, partial_tokens);
        assert_eq!(result.token_ids, partial_ids);
        assert!(!result.is_success());
    }

    #[test]
    fn test_partial_generation_on_cancellation() {
        let partial_tokens = vec!["Partial".to_string()];
        let partial_ids = vec![100];
        let result =
            InferenceResult::cancelled(partial_tokens.clone(), partial_ids.clone(), 42, 25);

        assert_eq!(result.token_count(), 1);
        assert_eq!(result.tokens, partial_tokens);
        assert_eq!(result.token_ids, partial_ids);
        assert!(!result.is_success());
    }
}

// ---
// Verified by Testing Team 🔍
