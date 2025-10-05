//! Inference execution coordinator
//!
//! This module coordinates the full inference pipeline:
//! 1. Configuration validation
//! 2. Stop sequence tokenization
//! 3. Token generation loop with all sampling parameters
//! 4. Stop sequence detection
//! 5. Result construction with stop reason
//!
//! # Spec References
//! - M0-W-1421: Advanced sampling parameters
//! - M0-W-1422: Stop sequences
//! - M0-W-1300: HTTP API extension

use worker_common::inference_result::StopReason;
use worker_common::inference_result::InferenceResult;
use worker_common::sampling_config::SamplingConfig;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Inference execution state
///
/// Tracks generation progress and determines when to stop.
#[derive(Debug)]
pub struct InferenceExecutor {
    /// Sampling configuration
    config: SamplingConfig,

    /// Generated tokens (UTF-8 strings)
    tokens: Vec<String>,

    /// Generated token IDs
    token_ids: Vec<u32>,

    /// Start time for performance tracking
    start_time: Instant,

    /// Whether inference should stop
    should_stop: bool,

    /// Stop reason (determined during execution)
    stop_reason: Option<StopReason>,

    /// Matched stop sequence (if applicable)
    stop_sequence_matched: Option<String>,
}

impl InferenceExecutor {
    /// Create new executor with configuration
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            tokens: Vec::new(),
            token_ids: Vec::new(),
            start_time: Instant::now(),
            should_stop: false,
            stop_reason: None,
            stop_sequence_matched: None,
        }
    }

    /// Add generated token and check stop conditions
    ///
    /// Returns true if inference should continue, false if should stop.
    pub fn add_token(&mut self, token: String, token_id: u32) -> bool {
        self.tokens.push(token);
        self.token_ids.push(token_id);

        // Check max_tokens
        if self.tokens.len() >= self.config.max_tokens as usize {
            debug!(
                tokens_generated = self.tokens.len(),
                max_tokens = self.config.max_tokens,
                "Reached max_tokens limit"
            );
            self.stop_reason = Some(StopReason::MaxTokens);
            self.should_stop = true;
            return false;
        }

        // Check stop sequences (if configured)
        if self.config.has_stop_sequences() {
            if let Some(matched) = self.check_stop_sequences() {
                info!(
                    tokens_generated = self.tokens.len(),
                    matched_sequence = ?matched,
                    "Matched stop sequence"
                );
                self.stop_reason = Some(StopReason::StopSequence);
                self.stop_sequence_matched = Some(matched);
                self.should_stop = true;
                return false;
            }
        }

        true // Continue generation
    }

    /// Check if generated sequence matches any stop sequence
    ///
    /// Returns the matched stop string if found.
    fn check_stop_sequences(&self) -> Option<String> {
        let num_generated = self.token_ids.len();

        for (seq_idx, stop_str) in self.config.stop_strings.iter().enumerate() {
            // Get tokenized version (if available)
            if seq_idx >= self.config.stop_sequences.len() {
                continue;
            }

            let stop_tokens = &self.config.stop_sequences[seq_idx];
            let seq_len = stop_tokens.len();

            // Need at least seq_len tokens to match
            if num_generated < seq_len {
                continue;
            }

            // Check if last seq_len tokens match stop sequence
            let mut match_found = true;
            for i in 0..seq_len {
                let gen_idx = num_generated - seq_len + i;
                if self.token_ids[gen_idx] != stop_tokens[i] {
                    match_found = false;
                    break;
                }
            }

            if match_found {
                return Some(stop_str.clone());
            }
        }

        None
    }

    /// Mark inference as cancelled
    pub fn cancel(&mut self) {
        warn!(tokens_generated = self.tokens.len(), "Inference cancelled by client");
        self.stop_reason = Some(StopReason::Cancelled);
        self.should_stop = true;
    }

    /// Mark inference as error
    pub fn error(&mut self) {
        warn!(tokens_generated = self.tokens.len(), "Inference terminated due to error");
        self.stop_reason = Some(StopReason::Error);
        self.should_stop = true;
    }

    /// Check if inference should stop
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// Get current token count
    pub fn token_count(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Build final inference result
    pub fn finalize(self) -> InferenceResult {
        let decode_time_ms = self.start_time.elapsed().as_millis() as u64;

        let stop_reason = self.stop_reason.unwrap_or(StopReason::MaxTokens);

        match stop_reason {
            StopReason::MaxTokens => InferenceResult::max_tokens(
                self.tokens,
                self.token_ids,
                self.config.seed,
                decode_time_ms,
            ),
            StopReason::Eos => InferenceResult::max_tokens(
                self.tokens,
                self.token_ids,
                self.config.seed,
                decode_time_ms,
            ),
            StopReason::StopSequence => InferenceResult::stop_sequence(
                self.tokens,
                self.token_ids,
                self.config.seed,
                decode_time_ms,
                self.stop_sequence_matched.unwrap_or_default(),
            ),
            StopReason::Cancelled => InferenceResult::cancelled(
                self.tokens,
                self.token_ids,
                self.config.seed,
                decode_time_ms,
            ),
            StopReason::Error => InferenceResult::error(
                self.tokens,
                self.token_ids,
                self.config.seed,
                decode_time_ms,
            ),
        }
    }

    /// Get reference to sampling configuration
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Get generated tokens so far (for history buffer)
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worker_http::validation::ExecuteRequest;

    fn make_config(max_tokens: u32, stop: Vec<String>) -> SamplingConfig {
        let req = ExecuteRequest {
            job_id: "test".to_string(),
            prompt: "Hello".to_string(),
            max_tokens,
            temperature: 0.7,
            seed: Some(42),
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            stop,
            min_p: 0.0,
        };
        SamplingConfig::from_request(&req)
    }

    #[test]
    fn test_max_tokens_termination() {
        let config = make_config(3, vec![]);
        let mut executor = InferenceExecutor::new(config);

        assert!(executor.add_token("Hello".to_string(), 100));
        assert!(executor.add_token(" world".to_string(), 200));
        assert!(!executor.add_token("!".to_string(), 300)); // Reaches max_tokens

        assert!(executor.should_stop());

        let result = executor.finalize();
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
        assert_eq!(result.token_count(), 3);
    }

    #[test]
    fn test_stop_sequence_detection() {
        let mut config = make_config(100, vec!["\n\n".to_string()]);
        // Simulate tokenized stop sequence: [13, 13] (newline tokens)
        config.stop_sequences = vec![vec![13, 13]];

        let mut executor = InferenceExecutor::new(config);

        assert!(executor.add_token("Line 1".to_string(), 100));
        assert!(executor.add_token("\n".to_string(), 13));
        assert!(!executor.add_token("\n".to_string(), 13)); // Matches stop sequence

        assert!(executor.should_stop());

        let result = executor.finalize();
        assert_eq!(result.stop_reason, StopReason::StopSequence);
        assert_eq!(result.stop_sequence_matched, Some("\n\n".to_string()));
    }

    #[test]
    fn test_partial_stop_sequence_no_match() {
        let mut config = make_config(100, vec!["END".to_string()]);
        // Simulate tokenized: [5, 6, 7] (3 tokens)
        config.stop_sequences = vec![vec![5, 6, 7]];

        let mut executor = InferenceExecutor::new(config);

        assert!(executor.add_token("E".to_string(), 5));
        assert!(executor.add_token("N".to_string(), 6));
        // Don't add token 7, so sequence incomplete
        assert!(executor.add_token("X".to_string(), 8));

        assert!(!executor.should_stop());
    }

    #[test]
    fn test_cancellation() {
        let config = make_config(100, vec![]);
        let mut executor = InferenceExecutor::new(config);

        executor.add_token("Hello".to_string(), 100);
        executor.cancel();

        assert!(executor.should_stop());

        let result = executor.finalize();
        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert_eq!(result.token_count(), 1);
    }

    #[test]
    fn test_error_termination() {
        let config = make_config(100, vec![]);
        let mut executor = InferenceExecutor::new(config);

        executor.add_token("Hello".to_string(), 100);
        executor.error();

        assert!(executor.should_stop());

        let result = executor.finalize();
        assert_eq!(result.stop_reason, StopReason::Error);
    }

    #[test]
    fn test_multiple_stop_sequences() {
        let mut config = make_config(100, vec!["\n\n".to_string(), "END".to_string()]);
        config.stop_sequences = vec![
            vec![13, 13],     // \n\n
            vec![20, 21, 22], // END
        ];

        let mut executor = InferenceExecutor::new(config);

        // Match second sequence
        executor.add_token("E".to_string(), 20);
        executor.add_token("N".to_string(), 21);
        let should_continue = executor.add_token("D".to_string(), 22);

        assert!(!should_continue);
        assert!(executor.should_stop());

        let result = executor.finalize();
        assert_eq!(result.stop_reason, StopReason::StopSequence);
        assert_eq!(result.stop_sequence_matched, Some("END".to_string()));
    }

    #[test]
    fn test_token_count_tracking() {
        let config = make_config(100, vec![]);
        let mut executor = InferenceExecutor::new(config);

        assert_eq!(executor.token_count(), 0);

        executor.add_token("A".to_string(), 1);
        assert_eq!(executor.token_count(), 1);

        executor.add_token("B".to_string(), 2);
        assert_eq!(executor.token_count(), 2);
    }

    #[test]
    fn test_token_ids_access() {
        let config = make_config(100, vec![]);
        let mut executor = InferenceExecutor::new(config);

        executor.add_token("A".to_string(), 100);
        executor.add_token("B".to_string(), 200);

        let ids = executor.token_ids();
        assert_eq!(ids, &[100, 200]);
    }
}

// ---
// Built by Foundation-Alpha 🏗️
