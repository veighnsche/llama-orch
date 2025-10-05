//! Unified sampling configuration
//!
//! This module provides a single configuration type that encapsulates
//! all sampling parameters, ensuring consistency between HTTP API,
//! CUDA kernels, and inference execution.
//!
//! # Spec References
//! - M0-W-1421: Advanced sampling parameters
//! - M0-W-1422: Stop sequences
//! - M0-W-1300: HTTP API extension

use crate::http::validation::ExecuteRequest;

/// Complete sampling configuration for inference
///
/// This struct bridges the HTTP API layer and CUDA kernel layer,
/// providing a single source of truth for all sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for sampling (0.0-2.0)
    /// - 0.0 = greedy (argmax)
    /// - 1.0 = no scaling
    /// - >1.0 = more random
    pub temperature: f32,

    /// Top-P (nucleus) sampling (0.0-1.0)
    /// - 1.0 = disabled (no filtering)
    /// - <1.0 = keep tokens with cumulative prob <= top_p
    pub top_p: f32,

    /// Top-K sampling (0 = disabled)
    /// - 0 = disabled (no filtering)
    /// - >0 = keep only top k tokens
    pub top_k: u32,

    /// Repetition penalty (1.0 = disabled)
    /// - 1.0 = disabled (no penalty)
    /// - >1.0 = penalize repeated tokens
    /// - <1.0 = encourage repeated tokens
    pub repetition_penalty: f32,

    /// Min-P sampling (0.0 = disabled)
    /// - 0.0 = disabled (no filtering)
    /// - >0.0 = filter tokens with prob < min_p * max_prob
    pub min_p: f32,

    /// Stop sequences (up to 4)
    /// Tokenized stop sequences for pattern matching
    pub stop_sequences: Vec<Vec<u32>>,

    /// Original stop strings (for response)
    pub stop_strings: Vec<String>,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl SamplingConfig {
    /// Create from HTTP request
    ///
    /// Converts ExecuteRequest into SamplingConfig, generating seed if needed.
    /// Stop sequences are NOT tokenized here - that happens in the inference layer.
    pub fn from_request(req: &ExecuteRequest) -> Self {
        Self {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            repetition_penalty: req.repetition_penalty,
            min_p: req.min_p,
            stop_sequences: vec![], // Tokenized later
            stop_strings: req.stop.clone(),
            seed: req.seed.unwrap_or_else(|| {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
            }),
            max_tokens: req.max_tokens,
        }
    }

    /// Check if any advanced parameters are enabled
    pub fn has_advanced_sampling(&self) -> bool {
        self.top_p < 1.0 || self.top_k > 0 || self.repetition_penalty != 1.0 || self.min_p > 0.0
    }

    /// Check if stop sequences are configured
    pub fn has_stop_sequences(&self) -> bool {
        !self.stop_strings.is_empty()
    }

    /// Check if greedy sampling (temperature = 0.0)
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Get sampling mode description for logging
    pub fn sampling_mode(&self) -> String {
        if self.is_greedy() {
            "greedy".to_string()
        } else if self.has_advanced_sampling() {
            let mut modes = vec![];
            if self.top_p < 1.0 {
                modes.push(format!("top_p={:.2}", self.top_p));
            }
            if self.top_k > 0 {
                modes.push(format!("top_k={}", self.top_k));
            }
            if self.repetition_penalty != 1.0 {
                modes.push(format!("rep_penalty={:.2}", self.repetition_penalty));
            }
            if self.min_p > 0.0 {
                modes.push(format!("min_p={:.2}", self.min_p));
            }
            format!("stochastic(temp={:.2}, {})", self.temperature, modes.join(", "))
        } else {
            format!("stochastic(temp={:.2})", self.temperature)
        }
    }

    /// Validate configuration consistency
    ///
    /// Checks for conflicting or nonsensical parameter combinations.
    pub fn validate_consistency(&self) -> Result<(), String> {
        // Warn if top_k and top_p both very restrictive
        if self.top_k > 0 && self.top_k < 10 && self.top_p < 0.5 {
            return Err(format!(
                "Very restrictive sampling: top_k={} and top_p={:.2} may produce poor results",
                self.top_k, self.top_p
            ));
        }

        // Warn if min_p very high with low temperature
        if self.min_p > 0.5 && self.temperature < 0.5 {
            return Err(format!(
                "Conflicting parameters: min_p={:.2} with temperature={:.2} may filter all tokens",
                self.min_p, self.temperature
            ));
        }

        Ok(())
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            min_p: 0.0,
            stop_sequences: vec![],
            stop_strings: vec![],
            seed: 0,
            max_tokens: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        min_p: f32,
        stop: Vec<String>,
    ) -> ExecuteRequest {
        ExecuteRequest {
            job_id: "test".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 100,
            temperature,
            seed: Some(42),
            top_p,
            top_k,
            repetition_penalty,
            stop,
            min_p,
        }
    }

    #[test]
    fn test_from_request_basic() {
        let req = make_request(0.7, 1.0, 0, 1.0, 0.0, vec![]);
        let config = SamplingConfig::from_request(&req);

        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.min_p, 0.0);
        assert_eq!(config.seed, 42);
        assert_eq!(config.max_tokens, 100);
    }

    #[test]
    fn test_from_request_advanced() {
        let req =
            make_request(0.7, 0.9, 50, 1.1, 0.05, vec!["\n\n".to_string(), "END".to_string()]);
        let config = SamplingConfig::from_request(&req);

        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.repetition_penalty, 1.1);
        assert_eq!(config.min_p, 0.05);
        assert_eq!(config.stop_strings.len(), 2);
    }

    #[test]
    fn test_seed_generation_when_none() {
        let mut req = make_request(1.0, 1.0, 0, 1.0, 0.0, vec![]);
        req.seed = None;

        let config1 = SamplingConfig::from_request(&req);
        let config2 = SamplingConfig::from_request(&req);

        // Seeds should be different (time-based)
        assert_ne!(config1.seed, 0);
        assert_ne!(config2.seed, 0);
    }

    #[test]
    fn test_has_advanced_sampling() {
        let req_basic = make_request(0.7, 1.0, 0, 1.0, 0.0, vec![]);
        let config_basic = SamplingConfig::from_request(&req_basic);
        assert!(!config_basic.has_advanced_sampling());

        let req_top_p = make_request(0.7, 0.9, 0, 1.0, 0.0, vec![]);
        let config_top_p = SamplingConfig::from_request(&req_top_p);
        assert!(config_top_p.has_advanced_sampling());

        let req_top_k = make_request(0.7, 1.0, 50, 1.0, 0.0, vec![]);
        let config_top_k = SamplingConfig::from_request(&req_top_k);
        assert!(config_top_k.has_advanced_sampling());

        let req_penalty = make_request(0.7, 1.0, 0, 1.2, 0.0, vec![]);
        let config_penalty = SamplingConfig::from_request(&req_penalty);
        assert!(config_penalty.has_advanced_sampling());

        let req_min_p = make_request(0.7, 1.0, 0, 1.0, 0.05, vec![]);
        let config_min_p = SamplingConfig::from_request(&req_min_p);
        assert!(config_min_p.has_advanced_sampling());
    }

    #[test]
    fn test_has_stop_sequences() {
        let req_no_stop = make_request(0.7, 1.0, 0, 1.0, 0.0, vec![]);
        let config_no_stop = SamplingConfig::from_request(&req_no_stop);
        assert!(!config_no_stop.has_stop_sequences());

        let req_with_stop = make_request(0.7, 1.0, 0, 1.0, 0.0, vec!["\n\n".to_string()]);
        let config_with_stop = SamplingConfig::from_request(&req_with_stop);
        assert!(config_with_stop.has_stop_sequences());
    }

    #[test]
    fn test_is_greedy() {
        let req_greedy = make_request(0.0, 1.0, 0, 1.0, 0.0, vec![]);
        let config_greedy = SamplingConfig::from_request(&req_greedy);
        assert!(config_greedy.is_greedy());

        let req_stochastic = make_request(0.7, 1.0, 0, 1.0, 0.0, vec![]);
        let config_stochastic = SamplingConfig::from_request(&req_stochastic);
        assert!(!config_stochastic.is_greedy());
    }

    #[test]
    fn test_sampling_mode_descriptions() {
        let greedy = make_request(0.0, 1.0, 0, 1.0, 0.0, vec![]);
        let config = SamplingConfig::from_request(&greedy);
        assert_eq!(config.sampling_mode(), "greedy");

        let basic = make_request(0.7, 1.0, 0, 1.0, 0.0, vec![]);
        let config = SamplingConfig::from_request(&basic);
        assert!(config.sampling_mode().contains("stochastic"));
        assert!(config.sampling_mode().contains("temp=0.70"));

        let advanced = make_request(0.7, 0.9, 50, 1.1, 0.05, vec![]);
        let config = SamplingConfig::from_request(&advanced);
        let mode = config.sampling_mode();
        assert!(mode.contains("top_p=0.90"));
        assert!(mode.contains("top_k=50"));
        assert!(mode.contains("rep_penalty=1.10"));
        assert!(mode.contains("min_p=0.05"));
    }

    #[test]
    fn test_validate_consistency_ok() {
        let req = make_request(0.7, 0.9, 50, 1.1, 0.05, vec![]);
        let config = SamplingConfig::from_request(&req);
        assert!(config.validate_consistency().is_ok());
    }

    #[test]
    fn test_validate_consistency_restrictive_sampling() {
        let req = make_request(0.7, 0.3, 5, 1.0, 0.0, vec![]);
        let config = SamplingConfig::from_request(&req);
        let result = config.validate_consistency();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("restrictive"));
    }

    #[test]
    fn test_validate_consistency_conflicting_min_p() {
        let req = make_request(0.3, 1.0, 0, 1.0, 0.6, vec![]);
        let config = SamplingConfig::from_request(&req);
        let result = config.validate_consistency();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Conflicting"));
    }

    #[test]
    fn test_default_config() {
        let config = SamplingConfig::default();

        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.min_p, 0.0);
        assert_eq!(config.stop_sequences.len(), 0);
        assert_eq!(config.max_tokens, 100);
    }
}

// ---
// Built by Foundation-Alpha 🏗️
