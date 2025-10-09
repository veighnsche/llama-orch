//! Inference backend implementation
//!
//! Created by: TEAM-000
//! Refactored by: TEAM-015 (split into focused modules)
//! Modified by: TEAM-017 (added multi-model support with enum pattern)

mod inference;
pub mod models;
mod sampling;
mod tokenizer_loader;

pub use inference::CandleInferenceBackend;
