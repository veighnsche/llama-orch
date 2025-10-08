//! Inference backend implementation
//!
//! Created by: TEAM-000
//! Refactored by: TEAM-015 (split into focused modules)

mod inference;
mod model_loader;
mod sampling;

pub use inference::CandleInferenceBackend;
