//! Sampling configuration and LogitsProcessor creation
//!
//! Created by: TEAM-015 (refactored from candle_backend.rs)
//! Original code by: TEAM-014

use candle_transformers::generation::{LogitsProcessor, Sampling};
use worker_common::SamplingConfig;

/// Create LogitsProcessor from SamplingConfig
///
/// TEAM-014: Use Candle's battle-tested LogitsProcessor instead of custom sampling
pub fn create_logits_processor(config: &SamplingConfig) -> LogitsProcessor {
    let temperature = if config.temperature == 0.0 {
        None
    } else {
        Some(config.temperature as f64)
    };

    let sampling = match (temperature, config.top_k, config.top_p) {
        (None, _, _) => Sampling::ArgMax,
        (Some(temp), 0, p) if p >= 1.0 => Sampling::All { temperature: temp },
        (Some(temp), 0, p) => Sampling::TopP { p: p as f64, temperature: temp },
        (Some(temp), k, p) if p >= 1.0 => Sampling::TopK { k: k as usize, temperature: temp },
        (Some(temp), k, p) => Sampling::TopKThenTopP { k: k as usize, p: p as f64, temperature: temp },
    };

    LogitsProcessor::from_sampling(config.seed, sampling)
}
