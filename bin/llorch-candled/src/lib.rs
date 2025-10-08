//! llorch-candled - Candle-based Llama-2 inference library
//!
//! This library implements Llama-2 inference using a hybrid approach:
//! - Pure ndarray for CPU (checkpoint validation)
//! - Candle kernels for CUDA acceleration (optional)
//!
//! Architecture follows CANDLE_INTEGRATION_HANDOFF.md:
//! - Use Candle's kernels, NOT the framework
//! - Keep checkpoint-driven validation
//! - Maintain educational value
//!
//! Created by: TEAM-000 (Foundation)

pub mod backend;
pub mod cache;
pub mod device; // TEAM-007: Multi-backend device initialization
pub mod error;
pub mod layers;
pub mod model;
pub mod tensor;

// Re-export commonly used types
pub use backend::CandleInferenceBackend;
pub use cache::{Cache, KvCache};
pub use error::LlorchError;
