//! Sampling configuration and `LogitsProcessor` creation
//!
//! Created by: TEAM-015 (refactored from `candle_backend.rs`)
//! Original code by: TEAM-014

use crate::common::SamplingConfig;
use candle_transformers::generation::{LogitsProcessor, Sampling};

/// Create `LogitsProcessor` from `SamplingConfig`
///
/// TEAM-014: Use Candle's battle-tested `LogitsProcessor` instead of custom sampling
pub fn create_logits_processor(config: &SamplingConfig) -> LogitsProcessor {
    let temperature =
        if config.temperature == 0.0 { None } else { Some(f64::from(config.temperature)) };

    let sampling = match (temperature, config.top_k, config.top_p) {
        (None, _, _) => Sampling::ArgMax,
        (Some(temp), 0, p) if p >= 1.0 => Sampling::All { temperature: temp },
        (Some(temp), 0, p) => Sampling::TopP { p: f64::from(p), temperature: temp },
        (Some(temp), k, p) if p >= 1.0 => Sampling::TopK { k: k as usize, temperature: temp },
        (Some(temp), k, p) => {
            Sampling::TopKThenTopP { k: k as usize, p: f64::from(p), temperature: temp }
        }
    };

    LogitsProcessor::from_sampling(config.seed, sampling)
}
