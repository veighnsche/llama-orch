//! Common types, errors, and utilities for llama-orch workers
//!
//! This crate provides shared functionality used by all worker implementations:
//! - Error types
//! - Sampling configuration
//! - Inference result types
//! - Ready callback logic
//!
//! # Example
//!
//! ```no_run
//! use worker_common::{SamplingConfig, callback};
//!
//! let config = SamplingConfig {
//!     temperature: 0.7,
//!     top_p: 0.9,
//!     ..Default::default()
//! };
//!
//! callback::send_ready("http://pool:9200/ready", "worker-1", 16_000_000_000, "vram-only").await?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// TODO: Extract from worker-orcd/src/
// - error.rs (2071 bytes)
// - sampling_config.rs (11308 bytes)
// - inference_result.rs (5880 bytes)
// - startup.rs (989 bytes) → callback.rs

pub mod placeholder {
    //! Placeholder module until extraction is complete
    
    pub fn version() -> &'static str {
        "0.1.0"
    }
}
