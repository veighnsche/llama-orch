//! llorch-candled - Candle-based Llama-2 inference library
//!
//! TEAM-009 rewrite: Uses candle-transformers::models::llama::Llama directly
//! instead of custom layer implementations.
//!
//! Architecture:
//! - SafeTensors model loading via VarBuilder
//! - HuggingFace tokenizers integration
//! - Multi-backend support (CPU, CUDA, Accelerate)
//! - Worker integration via InferenceBackend trait
//!
//! Created by: TEAM-000 (Foundation)
//! Modified by: TEAM-010 (Removed all deprecated modules)

pub mod backend;
pub mod device; // TEAM-007: Multi-backend device initialization
pub mod error;

// Re-export commonly used types
pub use backend::CandleInferenceBackend;
pub use error::LlorchError;
