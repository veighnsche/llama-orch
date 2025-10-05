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

    fn make_config(
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        min_p: f32,
        stop_strings: Vec<String>,
        seed: u64,
        max_tokens: u32,
    ) -> SamplingConfig {
        SamplingConfig {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_p,
            stop_sequences: vec![],
            stop_strings,
            seed,
            max_tokens,
        }
    }

    #[test]
    fn test_has_advanced_sampling() {
        let config_basic = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config_basic.has_advanced_sampling());

        let config_top_p = make_config(0.7, 0.9, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(config_top_p.has_advanced_sampling());

        let config_top_k = make_config(0.7, 1.0, 50, 1.0, 0.0, vec![], 42, 100);
        assert!(config_top_k.has_advanced_sampling());

        let config_penalty = make_config(0.7, 1.0, 0, 1.2, 0.0, vec![], 42, 100);
        assert!(config_penalty.has_advanced_sampling());

        let config_min_p = make_config(0.7, 1.0, 0, 1.0, 0.05, vec![], 42, 100);
        assert!(config_min_p.has_advanced_sampling());
    }

    #[test]
    fn test_has_stop_sequences() {
        let config_no_stop = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config_no_stop.has_stop_sequences());

        let config_with_stop =
            make_config(0.7, 1.0, 0, 1.0, 0.0, vec!["\n\n".to_string()], 42, 100);
        assert!(config_with_stop.has_stop_sequences());
    }

    #[test]
    fn test_is_greedy() {
        let config_greedy = make_config(0.0, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(config_greedy.is_greedy());

        let config_stochastic = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config_stochastic.is_greedy());
    }

    #[test]
    fn test_sampling_mode_greedy() {
        let config = make_config(0.0, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert_eq!(config.sampling_mode(), "greedy");
    }

    #[test]
    fn test_sampling_mode_basic_stochastic() {
        let config = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        let mode = config.sampling_mode();
        assert!(mode.contains("stochastic"));
        assert!(mode.contains("temp=0.70"));
    }

    #[test]
    fn test_sampling_mode_advanced() {
        let config = make_config(0.7, 0.9, 50, 1.1, 0.05, vec![], 42, 100);
        let mode = config.sampling_mode();
        assert!(mode.contains("top_p=0.90"));
        assert!(mode.contains("top_k=50"));
        assert!(mode.contains("rep_penalty=1.10"));
        assert!(mode.contains("min_p=0.05"));
    }

    #[test]
    fn test_validate_consistency_ok() {
        let config = make_config(0.7, 0.9, 50, 1.1, 0.05, vec![], 42, 100);
        assert!(config.validate_consistency().is_ok());
    }

    #[test]
    fn test_validate_consistency_restrictive_sampling() {
        let config = make_config(0.7, 0.3, 5, 1.0, 0.0, vec![], 42, 100);
        let result = config.validate_consistency();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("restrictive"));
    }

    #[test]
    fn test_validate_consistency_conflicting_min_p() {
        let config = make_config(0.3, 1.0, 0, 1.0, 0.6, vec![], 42, 100);
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

    #[test]
    fn test_temperature_range() {
        let config_zero = make_config(0.0, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(config_zero.is_greedy());

        let config_low = make_config(0.1, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config_low.is_greedy());

        let config_high = make_config(2.0, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config_high.is_greedy());
    }

    #[test]
    fn test_top_p_disabled() {
        let config = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config.has_advanced_sampling());
    }

    #[test]
    fn test_top_k_disabled() {
        let config = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config.has_advanced_sampling());
    }

    #[test]
    fn test_repetition_penalty_disabled() {
        let config = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config.has_advanced_sampling());
    }

    #[test]
    fn test_min_p_disabled() {
        let config = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert!(!config.has_advanced_sampling());
    }

    #[test]
    fn test_multiple_stop_sequences() {
        let stops = vec!["\n\n".to_string(), "END".to_string(), "###".to_string()];
        let config = make_config(0.7, 1.0, 0, 1.0, 0.0, stops.clone(), 42, 100);
        assert!(config.has_stop_sequences());
        assert_eq!(config.stop_strings.len(), 3);
        assert_eq!(config.stop_strings, stops);
    }

    #[test]
    fn test_seed_values() {
        let config1 = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 100);
        assert_eq!(config1.seed, 42);

        let config2 = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 999, 100);
        assert_eq!(config2.seed, 999);

        let config3 = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 0, 100);
        assert_eq!(config3.seed, 0);
    }

    #[test]
    fn test_max_tokens_values() {
        let config1 = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 1);
        assert_eq!(config1.max_tokens, 1);

        let config2 = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 1000);
        assert_eq!(config2.max_tokens, 1000);

        let config3 = make_config(0.7, 1.0, 0, 1.0, 0.0, vec![], 42, 4096);
        assert_eq!(config3.max_tokens, 4096);
    }

    #[test]
    fn test_validate_consistency_edge_cases() {
        // Very low top_k with low top_p should error
        let config1 = make_config(0.7, 0.3, 5, 1.0, 0.0, vec![], 42, 100);
        assert!(config1.validate_consistency().is_err());

        // Very low top_k with moderate top_p should be ok
        let config2 = make_config(0.7, 0.8, 5, 1.0, 0.0, vec![], 42, 100);
        assert!(config2.validate_consistency().is_ok());

        // High min_p with moderate temperature should be ok
        let config3 = make_config(0.8, 1.0, 0, 1.0, 0.5, vec![], 42, 100);
        assert!(config3.validate_consistency().is_ok());

        // All defaults should be ok
        let config4 = SamplingConfig::default();
        assert!(config4.validate_consistency().is_ok());
    }

    #[test]
    fn test_sampling_mode_with_single_advanced_param() {
        let config_top_p = make_config(0.7, 0.9, 0, 1.0, 0.0, vec![], 42, 100);
        let mode = config_top_p.sampling_mode();
        assert!(mode.contains("top_p=0.90"));
        assert!(!mode.contains("top_k"));

        let config_top_k = make_config(0.7, 1.0, 50, 1.0, 0.0, vec![], 42, 100);
        let mode = config_top_k.sampling_mode();
        assert!(mode.contains("top_k=50"));
        assert!(!mode.contains("top_p"));
    }

    #[test]
    fn test_clone_config() {
        let config = make_config(0.7, 0.9, 50, 1.1, 0.05, vec!["END".to_string()], 42, 100);
        let cloned = config.clone();

        assert_eq!(config.temperature, cloned.temperature);
        assert_eq!(config.top_p, cloned.top_p);
        assert_eq!(config.top_k, cloned.top_k);
        assert_eq!(config.repetition_penalty, cloned.repetition_penalty);
        assert_eq!(config.min_p, cloned.min_p);
        assert_eq!(config.seed, cloned.seed);
        assert_eq!(config.max_tokens, cloned.max_tokens);
        assert_eq!(config.stop_strings, cloned.stop_strings);
    }

    #[test]
    fn test_debug_format() {
        let config = make_config(0.7, 0.9, 50, 1.1, 0.05, vec![], 42, 100);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("SamplingConfig"));
    }
}

// ---
// Verified by Testing Team 🔍
