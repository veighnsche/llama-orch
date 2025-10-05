//! capability-matcher — Model Capability Descriptor (MCD) / Engine Capability Profile (ECP) matching
//!
//! Validates that a model's required capabilities are supported by the worker's engine.
//!
//! # Security Properties
//!
//! - Explicit capability matching (no silent fallbacks)
//! - Deterministic compatibility checks
//! - Prevents loading incompatible models
//! - Clear error messages on mismatch
//!
//! # Example
//!
//! ```rust
//! use capability_matcher::{ModelCapabilityDescriptor, EngineCapabilityProfile, CapabilityMatcher};
//!
//! let mcd = ModelCapabilityDescriptor {
//!     model_id: "meta-llama/Llama-3.1-8B".to_string(),
//!     positional: "rope_llama".to_string(),
//!     attention: "gqa".to_string(),
//!     quant: vec!["q4_0".to_string()],
//!     context_max: 8192,
//!     vocab_size: 128256,
//! };
//!
//! let ecp = EngineCapabilityProfile {
//!     worker_id: "worker-gpu-0".to_string(),
//!     supports_positional: vec!["rope_llama".to_string(), "rope_neox".to_string()],
//!     supports_attention: vec!["mha".to_string(), "gqa".to_string()],
//!     supports_quant: vec!["q4_0".to_string(), "q8_0".to_string()],
//!     max_context: 16384,
//!     vram_bytes: 24_000_000_000,
//! };
//!
//! let matcher = CapabilityMatcher::new();
//! matcher.check_compatibility(&mcd, &ecp)?; // Returns Ok if compatible
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
pub enum CapabilityError {
    #[error("incompatible positional encoding: model requires {required}, worker supports {supported:?}")]
    IncompatiblePositional { required: String, supported: Vec<String> },
    #[error("incompatible attention mechanism: model requires {required}, worker supports {supported:?}")]
    IncompatibleAttention { required: String, supported: Vec<String> },
    #[error("incompatible quantization: model requires one of {required:?}, worker supports {supported:?}")]
    IncompatibleQuant { required: Vec<String>, supported: Vec<String> },
    #[error("context too large: model requires {required}, worker max is {max}")]
    ContextTooLarge { required: usize, max: usize },
    #[error("invalid MCD: {0}")]
    InvalidMcd(String),
    #[error("invalid ECP: {0}")]
    InvalidEcp(String),
}

pub type Result<T> = std::result::Result<T, CapabilityError>;

/// Model Capability Descriptor (MCD)
/// Embedded in model artifacts or provided as sidecar JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilityDescriptor {
    pub model_id: String,
    pub positional: String, // e.g., "rope_llama", "rope_neox", "alibi"
    pub attention: String,  // e.g., "mha", "gqa", "mqa"
    pub quant: Vec<String>, // e.g., ["q4_0", "q8_0"]
    pub context_max: usize,
    pub vocab_size: usize,
}

/// Engine Capability Profile (ECP)
/// Advertised by worker at startup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilityProfile {
    pub worker_id: String,
    pub supports_positional: Vec<String>,
    pub supports_attention: Vec<String>,
    pub supports_quant: Vec<String>,
    pub max_context: usize,
    pub vram_bytes: usize,
}

/// Capability matcher
pub struct CapabilityMatcher;

impl CapabilityMatcher {
    pub fn new() -> Self {
        Self
    }

    /// Check if MCD ⊆ ECP (model capabilities are subset of engine capabilities)
    pub fn check_compatibility(
        &self,
        mcd: &ModelCapabilityDescriptor,
        ecp: &EngineCapabilityProfile,
    ) -> Result<()> {
        // Validate positional encoding
        if !ecp.supports_positional.contains(&mcd.positional) {
            return Err(CapabilityError::IncompatiblePositional {
                required: mcd.positional.clone(),
                supported: ecp.supports_positional.clone(),
            });
        }

        // Validate attention mechanism
        if !ecp.supports_attention.contains(&mcd.attention) {
            return Err(CapabilityError::IncompatibleAttention {
                required: mcd.attention.clone(),
                supported: ecp.supports_attention.clone(),
            });
        }

        // Validate quantization (model needs at least one supported format)
        let has_compatible_quant = mcd.quant.iter().any(|q| ecp.supports_quant.contains(q));
        if !has_compatible_quant {
            return Err(CapabilityError::IncompatibleQuant {
                required: mcd.quant.clone(),
                supported: ecp.supports_quant.clone(),
            });
        }

        // Validate context size
        if mcd.context_max > ecp.max_context {
            return Err(CapabilityError::ContextTooLarge {
                required: mcd.context_max,
                max: ecp.max_context,
            });
        }

        tracing::info!(
            model_id = %mcd.model_id,
            worker_id = %ecp.worker_id,
            "Model capabilities compatible with worker"
        );

        Ok(())
    }

    /// Parse MCD from JSON string
    pub fn parse_mcd(&self, json: &str) -> Result<ModelCapabilityDescriptor> {
        serde_json::from_str(json).map_err(|e| CapabilityError::InvalidMcd(e.to_string()))
    }

    /// Parse ECP from JSON string
    pub fn parse_ecp(&self, json: &str) -> Result<EngineCapabilityProfile> {
        serde_json::from_str(json).map_err(|e| CapabilityError::InvalidEcp(e.to_string()))
    }
}

impl Default for CapabilityMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mcd() -> ModelCapabilityDescriptor {
        ModelCapabilityDescriptor {
            model_id: "meta-llama/Llama-3.1-8B".to_string(),
            positional: "rope_llama".to_string(),
            attention: "gqa".to_string(),
            quant: vec!["q4_0".to_string()],
            context_max: 8192,
            vocab_size: 128256,
        }
    }

    fn create_test_ecp() -> EngineCapabilityProfile {
        EngineCapabilityProfile {
            worker_id: "worker-gpu-0".to_string(),
            supports_positional: vec!["rope_llama".to_string(), "rope_neox".to_string()],
            supports_attention: vec!["mha".to_string(), "gqa".to_string()],
            supports_quant: vec!["q4_0".to_string(), "q8_0".to_string()],
            max_context: 16384,
            vram_bytes: 24_000_000_000,
        }
    }

    #[test]
    fn test_compatible_capabilities() {
        let matcher = CapabilityMatcher::new();
        let mcd = create_test_mcd();
        let ecp = create_test_ecp();

        assert!(matcher.check_compatibility(&mcd, &ecp).is_ok());
    }

    #[test]
    fn test_incompatible_positional() {
        let matcher = CapabilityMatcher::new();
        let mut mcd = create_test_mcd();
        mcd.positional = "alibi".to_string(); // Not supported
        let ecp = create_test_ecp();

        let result = matcher.check_compatibility(&mcd, &ecp);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CapabilityError::IncompatiblePositional { .. }));
    }

    #[test]
    fn test_incompatible_attention() {
        let matcher = CapabilityMatcher::new();
        let mut mcd = create_test_mcd();
        mcd.attention = "mqa".to_string(); // Not supported
        let ecp = create_test_ecp();

        let result = matcher.check_compatibility(&mcd, &ecp);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CapabilityError::IncompatibleAttention { .. }));
    }

    #[test]
    fn test_context_too_large() {
        let matcher = CapabilityMatcher::new();
        let mut mcd = create_test_mcd();
        mcd.context_max = 32768; // Exceeds worker max
        let ecp = create_test_ecp();

        let result = matcher.check_compatibility(&mcd, &ecp);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CapabilityError::ContextTooLarge { .. }));
    }
}
